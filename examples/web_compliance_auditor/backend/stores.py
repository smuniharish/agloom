"""PostgreSQL persistence for the auditor.

The feedback and refresh classes implement the persistence contracts of
feedback-manager and refresh-engine. The stores share one psycopg connection
pool; asynchronous contracts run their queries in worker threads, which keeps
them independent of the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import datetime
from types import MappingProxyType
from typing import Any
from uuid import UUID

import psycopg
from feedback_manager import (
    FeedbackConflictError,
    FeedbackEvent,
    FeedbackNotFoundError,
    FeedbackQuery,
    FeedbackStatus,
    FeedbackStoreError,
    validate_transition,
)
from feedback_manager.contracts.store import FeedbackStore
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from refresh_engine import (
    ResourceFingerprint,
    ResourceState,
    StateStoreABC,
    StateTransactionABC,
)
from refresh_engine import StateStoreError as RefreshStoreError

from .models import Audit, ComplianceRecord

TABLES = {
    "policies": "compliance_policies",
    "refresh_states": "compliance_refresh_state",
    "audits": "compliance_audits",
    "register": "compliance_register",
    "feedback": "compliance_feedback",
}


def open_pool(database_url: str, *, max_size: int = 10) -> ConnectionPool:
    """Open an autocommit connection pool with pgvector types registered."""
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True},
        configure=register_vector,
        open=False,
        name="web-compliance-auditor",
    )
    pool.open(wait=True, timeout=30)
    return pool


def table_counts(pool: ConnectionPool) -> dict[str, int]:
    """Return the number of rows of every application table."""
    counts: dict[str, int] = {}
    with pool.connection() as connection:
        for name, table in TABLES.items():
            row = connection.execute("SELECT to_regclass(%s)", (table,)).fetchone()
            exists = row is not None and row[0] is not None
            counted = (
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()
                if exists
                else None
            )
            counts[name] = int(counted[0]) if counted else 0
    return counts


class PostgresFeedbackStore(FeedbackStore):
    """Durable `FeedbackManager` storage."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the feedback table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS compliance_feedback (
                    feedback_id UUID PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS compliance_feedback_target
                    ON compliance_feedback (target_type, target_id, status);
                """)

    async def create(self, feedback: FeedbackEvent) -> FeedbackEvent:
        """Store an event, or return the one stored with its idempotency key."""
        return await asyncio.to_thread(self._create, feedback)

    async def get(self, feedback_id: UUID) -> FeedbackEvent | None:
        """Return the event with `feedback_id`, if any."""
        return await asyncio.to_thread(self._get, feedback_id)

    async def transition(
        self,
        feedback_id: UUID,
        status: FeedbackStatus,
        *,
        expected: FeedbackStatus,
        resolution: Mapping[str, Any] | None = None,
    ) -> FeedbackEvent:
        """Atomically move an event from `expected` to `status`."""
        return await asyncio.to_thread(
            self._transition, feedback_id, status, expected, resolution
        )

    async def query(self, query: FeedbackQuery) -> Sequence[FeedbackEvent]:
        """Return the matching events in creation order."""
        return await asyncio.to_thread(self._query, query)

    def _create(self, feedback: FeedbackEvent) -> FeedbackEvent:
        key = feedback.idempotency_key
        with self._pool.connection() as connection, connection.transaction():
            if key is not None and (existing := self._by_key(connection, key)):
                return existing
            try:
                inserted = connection.execute(
                    """
                    INSERT INTO compliance_feedback (feedback_id, idempotency_key,
                        target_type, target_id, status, created_at, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING feedback_id
                    """,
                    (
                        feedback.feedback_id,
                        key,
                        str(feedback.target.type),
                        feedback.target.id,
                        feedback.status.value,
                        feedback.created_at,
                        Jsonb(feedback.model_dump(mode="json")),
                    ),
                ).fetchone()
            except psycopg.errors.UniqueViolation as error:
                raise FeedbackStoreError(
                    "a feedback event with this ID already exists",
                    feedback_id=feedback.feedback_id,
                ) from error
        if inserted is None and key is not None:
            with self._pool.connection() as connection:
                existing = self._by_key(connection, key)
            if existing is not None:
                return existing
        return feedback

    @staticmethod
    def _by_key(connection: psycopg.Connection[Any], key: str) -> FeedbackEvent | None:
        row = connection.execute(
            "SELECT payload FROM compliance_feedback WHERE idempotency_key = %s",
            (key,),
        ).fetchone()
        return FeedbackEvent.model_validate(row[0]) if row else None

    def _get(self, feedback_id: UUID) -> FeedbackEvent | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM compliance_feedback WHERE feedback_id = %s",
                (feedback_id,),
            ).fetchone()
        return FeedbackEvent.model_validate(row[0]) if row else None

    def _transition(
        self,
        feedback_id: UUID,
        status: FeedbackStatus,
        expected: FeedbackStatus,
        resolution: Mapping[str, Any] | None,
    ) -> FeedbackEvent:
        with self._pool.connection() as connection, connection.transaction():
            row = connection.execute(
                """
                SELECT payload FROM compliance_feedback
                WHERE feedback_id = %s FOR UPDATE
                """,
                (feedback_id,),
            ).fetchone()
            if row is None:
                raise FeedbackNotFoundError(
                    "unknown feedback event", feedback_id=feedback_id
                )
            current = FeedbackEvent.model_validate(row[0])
            if current.status != expected:
                raise FeedbackConflictError(
                    f"feedback is {current.status}, not the expected {expected}",
                    feedback_id=feedback_id,
                    current_status=current.status.value,
                    requested_status=status.value,
                )
            if validate_transition(feedback_id, expected, status).idempotent:
                return current
            updated = current.with_status(
                status, resolution=dict(resolution) if resolution is not None else None
            )
            connection.execute(
                """
                UPDATE compliance_feedback SET status = %s, payload = %s
                WHERE feedback_id = %s
                """,
                (
                    updated.status.value,
                    Jsonb(updated.model_dump(mode="json")),
                    feedback_id,
                ),
            )
        return updated

    def _query(self, query: FeedbackQuery) -> list[FeedbackEvent]:
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("idempotency_key", query.idempotency_key),
            ("target_type", query.target_type),
            ("target_id", query.target_id),
            ("status", query.status.value if query.status is not None else None),
        ):
            if value is not None:
                clauses.append(f"{column} = %s")
                values.append(str(value))
        if query.created_after is not None:
            clauses.append("created_at >= %s")
            values.append(query.created_after)
        if query.created_before is not None:
            clauses.append("created_at < %s")
            values.append(query.created_before)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if query.newest_first else "ASC"
        with self._pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT payload FROM compliance_feedback {where}
                ORDER BY created_at {order}, feedback_id {order}
                """,
                values,
            ).fetchall()
        matches: list[FeedbackEvent] = []
        for row in rows:
            event = FeedbackEvent.model_validate(row[0])
            if query.matches(event):
                matches.append(event)
                if len(matches) == query.limit:
                    break
        return matches


class PostgresRefreshStore(StateStoreABC):
    """Durable `RefreshEngine` fingerprints, so restarts refresh only changes."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the refresh state table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS compliance_refresh_state (
                    resource_id TEXT PRIMARY KEY,
                    payload JSONB NOT NULL
                )
                """)

    async def load_all(self) -> Mapping[str, ResourceState]:
        """Return the last committed state of every resource."""
        return await asyncio.to_thread(self._load_all)

    def transaction(self) -> PostgresRefreshTransaction:
        """Create a transaction that applies its changes atomically."""
        return PostgresRefreshTransaction(self._pool)

    def _load_all(self) -> Mapping[str, ResourceState]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                "SELECT resource_id, payload FROM compliance_refresh_state"
            ).fetchall()
        return MappingProxyType({row[0]: _resource_state(row[1]) for row in rows})


class PostgresRefreshTransaction(StateTransactionABC):
    """Applies staged refresh state changes in one database transaction."""

    def __init__(self, pool: ConnectionPool) -> None:
        super().__init__()
        self._pool = pool

    async def apply_changes(
        self,
        puts: Mapping[str, ResourceState],
        deletes: AbstractSet[str],
        /,
    ) -> None:
        """Persist the staged upserts and deletions atomically."""
        try:
            await asyncio.to_thread(self._apply, dict(puts), set(deletes))
        except psycopg.Error as error:
            raise RefreshStoreError(f"refresh state was not saved: {error}") from error

    def _apply(self, puts: dict[str, ResourceState], deletes: set[str]) -> None:
        with self._pool.connection() as connection, connection.transaction():
            for resource_id in deletes:
                connection.execute(
                    "DELETE FROM compliance_refresh_state WHERE resource_id = %s",
                    (resource_id,),
                )
            for resource_id, state in puts.items():
                connection.execute(
                    """
                    INSERT INTO compliance_refresh_state (resource_id, payload)
                    VALUES (%s, %s)
                    ON CONFLICT (resource_id) DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (resource_id, Jsonb(_state_payload(state))),
                )


def _state_payload(state: ResourceState) -> dict[str, Any]:
    return {
        "resource_id": state.resource_id,
        "fingerprint": {
            "algorithm": state.fingerprint.algorithm,
            "value": state.fingerprint.value,
        },
        "version": state.version,
        "cheap_indicator": state.cheap_indicator,
        "snapshot_metadata": _thaw(state.snapshot_metadata),
        "dependencies": sorted(state.dependencies),
        "refreshed_at": state.refreshed_at.isoformat(),
    }


def _resource_state(payload: Mapping[str, Any]) -> ResourceState:
    return ResourceState(
        payload["resource_id"],
        ResourceFingerprint(**payload["fingerprint"]),
        version=payload.get("version"),
        cheap_indicator=payload.get("cheap_indicator"),
        snapshot_metadata=payload.get("snapshot_metadata") or {},
        dependencies=frozenset(payload.get("dependencies") or ()),
        refreshed_at=datetime.fromisoformat(payload["refreshed_at"]),
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple | AbstractSet):
        return [_thaw(item) for item in value]
    return value


class AuditLog:
    """The application's record of every audit and its compliance record state."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the audit table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS compliance_audits (
                    audit_id UUID PRIMARY KEY,
                    url TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    async def save(self, audit: Audit) -> None:
        """Insert or replace an audit."""
        await asyncio.to_thread(self._save, audit)

    async def get(self, audit_id: UUID) -> Audit | None:
        """Return one audit."""
        return await asyncio.to_thread(self._get, audit_id)

    async def update_record(
        self, audit_id: UUID, record: ComplianceRecord
    ) -> Audit | None:
        """Replace the compliance record state of an audit."""
        current = await self.get(audit_id)
        if current is None:
            return None
        updated = current.model_copy(update={"record": record})
        await self.save(updated)
        return updated

    def _save(self, audit: Audit) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO compliance_audits (audit_id, url, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (audit_id) DO UPDATE SET
                    payload = EXCLUDED.payload, updated_at = now()
                """,
                (audit.audit_id, audit.url, Jsonb(audit.model_dump(mode="json"))),
            )

    def _get(self, audit_id: UUID) -> Audit | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM compliance_audits WHERE audit_id = %s",
                (audit_id,),
            ).fetchone()
        return Audit.model_validate(row[0]) if row else None


class ComplianceRegister:
    """The compliance register that site owners and auditors rely on.

    Publishing to it is the auditor's consequential side effect; the policy
    manager calls `ainvoke` at most once per audit.
    """

    description = "Publish a grounded audit verdict to the compliance register."

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the register table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS compliance_register (
                    audit_id UUID PRIMARY KEY,
                    url TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    policy_ids JSONB NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    async def ainvoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Publish one audit and return its JSON-compatible receipt."""
        return await asyncio.to_thread(self._publish, dict(arguments))

    def _publish(self, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO compliance_register
                    (audit_id, url, verdict, summary, policy_ids)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (audit_id) DO NOTHING
                """,
                (
                    arguments["audit_id"],
                    arguments["url"],
                    arguments["verdict"],
                    arguments["summary"],
                    Jsonb(list(arguments["policy_ids"])),
                ),
            )
            row = connection.execute(
                """
                SELECT verdict, recorded_at FROM compliance_register
                WHERE audit_id = %s
                """,
                (arguments["audit_id"],),
            ).fetchone()
        if row is None:
            raise RuntimeError("the audit was not recorded")
        return {
            "audit_id": str(arguments["audit_id"]),
            "verdict": row[0],
            "recorded_at": row[1].isoformat(),
        }
