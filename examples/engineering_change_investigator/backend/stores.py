"""PostgreSQL persistence for the investigator.

Each class implements the persistence contract of the package that uses it.
The stores share one psycopg connection pool; asynchronous contracts run their
queries in worker threads, which keeps them independent of the event loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from datetime import datetime
from types import MappingProxyType
from typing import Any
from uuid import UUID

import psycopg
from behaviorweave import BehaviorState, StateStoreError
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
from langgraph_xai import Execution, ExecutionContext, ProvenanceLink, ProvenanceStore
from langgraph_xai.core.models import XAIEvent
from langgraph_xai.storage import StoreFilter
from mcp_capability_router import (
    Capability,
    CapabilityType,
    Prompt,
    PromptArgument,
    Resource,
    Tool,
)
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

from .models import Investigation, ReleaseDecision


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
        name="engineering-change-investigator",
    )
    pool.open(wait=True, timeout=30)
    return pool


TABLES = {
    "documents": "engineering_documents",
    "feedback": "engineering_feedback",
    "behavior_states": "engineering_behavior_state",
    "mcp_capabilities": "engineering_mcp_capabilities",
    "refresh_states": "engineering_refresh_state",
    "xai_records": "engineering_xai_records",
    "investigations": "engineering_investigations",
    "release_decisions": "engineering_release_register",
}


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
                CREATE TABLE IF NOT EXISTS engineering_feedback (
                    feedback_id UUID PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    payload JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS engineering_feedback_target
                    ON engineering_feedback (target_type, target_id, status);
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
                    INSERT INTO engineering_feedback (feedback_id, idempotency_key,
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
            "SELECT payload FROM engineering_feedback WHERE idempotency_key = %s",
            (key,),
        ).fetchone()
        return FeedbackEvent.model_validate(row[0]) if row else None

    def _get(self, feedback_id: UUID) -> FeedbackEvent | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM engineering_feedback WHERE feedback_id = %s",
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
                SELECT payload FROM engineering_feedback
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
                UPDATE engineering_feedback SET status = %s, payload = %s
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
                SELECT payload FROM engineering_feedback {where}
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


class PostgresBehaviorStateStore:
    """Durable BehaviorWeave pattern state with per-key serialized updates."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the behavior state table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_behavior_state (
                    state_key TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    version BIGINT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    def get(self, key: str) -> BehaviorState | None:
        """Return the state stored under `key`."""
        with self._pool.connection() as connection:
            row = connection.execute(
                "SELECT payload FROM engineering_behavior_state WHERE state_key = %s",
                (key,),
            ).fetchone()
        return BehaviorState.from_dict(row[0]) if row else None

    def put(self, key: str, state: BehaviorState) -> None:
        """Store `state` under `key`."""
        with self._pool.connection() as connection:
            self._write(connection, key, state)

    def update(
        self, key: str, fn: Callable[[BehaviorState | None], BehaviorState]
    ) -> BehaviorState:
        """Replace the state under `key` with `fn(current)`, serialized per key."""
        with self._pool.connection() as connection, connection.transaction():
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,)
            )
            row = connection.execute(
                "SELECT payload FROM engineering_behavior_state WHERE state_key = %s",
                (key,),
            ).fetchone()
            state = fn(BehaviorState.from_dict(row[0]) if row else None)
            if not isinstance(state, BehaviorState):
                raise StateStoreError("update function must return a BehaviorState")
            self._write(connection, key, state)
        return state

    def delete(self, key: str) -> None:
        """Remove the state stored under `key`."""
        with self._pool.connection() as connection:
            connection.execute(
                "DELETE FROM engineering_behavior_state WHERE state_key = %s", (key,)
            )

    @staticmethod
    def _write(
        connection: psycopg.Connection[Any], key: str, state: BehaviorState
    ) -> None:
        connection.execute(
            """
            INSERT INTO engineering_behavior_state (state_key, payload, version)
            VALUES (%s, %s, %s)
            ON CONFLICT (state_key) DO UPDATE SET
                payload = EXCLUDED.payload,
                version = EXCLUDED.version,
                updated_at = now()
            """,
            (key, Jsonb(state.to_dict()), state.version),
        )


_CAPABILITY_TYPES: dict[CapabilityType, type[Capability]] = {
    CapabilityType.TOOL: Tool,
    CapabilityType.RESOURCE: Resource,
    CapabilityType.PROMPT: Prompt,
}


class PostgresCapabilityRegistry:
    """Durable MCP capability metadata for `mcp_capability_router.MCPRuntime`."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the capability table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_mcp_capabilities (
                    capability_id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    capability_type TEXT NOT NULL,
                    payload JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS engineering_mcp_capabilities_server
                    ON engineering_mcp_capabilities (server_id, capability_type);
                """)

    async def upsert_many(self, capabilities: Iterable[Capability]) -> None:
        """Insert or replace capabilities by ID."""
        await asyncio.to_thread(self._upsert_many, list(capabilities))

    async def remove(self, capability_id: str) -> None:
        """Remove one capability."""
        await asyncio.to_thread(
            self._execute,
            "DELETE FROM engineering_mcp_capabilities WHERE capability_id = %s",
            (capability_id,),
        )

    async def remove_server(self, server_id: str) -> None:
        """Remove every capability of one server."""
        await asyncio.to_thread(
            self._execute,
            "DELETE FROM engineering_mcp_capabilities WHERE server_id = %s",
            (server_id,),
        )

    async def get(self, capability_id: str) -> Capability | None:
        """Return one capability by ID."""
        rows = await asyncio.to_thread(
            self._select,
            "WHERE capability_id = %s",
            (capability_id,),
        )
        return rows[0] if rows else None

    async def list(
        self,
        *,
        server_id: str | None = None,
        type: CapabilityType | None = None,  # noqa: A002 - registry contract name
    ) -> Sequence[Capability]:
        """Return the stored capabilities, optionally filtered."""
        clauses: list[str] = []
        values: list[str] = []
        if server_id is not None:
            clauses.append("server_id = %s")
            values.append(server_id)
        if type is not None:
            clauses.append("capability_type = %s")
            values.append(CapabilityType(type).value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return await asyncio.to_thread(self._select, where, tuple(values))

    async def close(self) -> None:
        """Keep the shared pool open; the application closes it."""

    def _upsert_many(self, capabilities: list[Capability]) -> None:
        with self._pool.connection() as connection, connection.transaction():
            for capability in capabilities:
                connection.execute(
                    """
                    INSERT INTO engineering_mcp_capabilities
                        (capability_id, server_id, capability_type, payload)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (capability_id) DO UPDATE SET
                        server_id = EXCLUDED.server_id,
                        capability_type = EXCLUDED.capability_type,
                        payload = EXCLUDED.payload
                    """,
                    (
                        capability.capability_id,
                        capability.server_id,
                        CapabilityType(capability.type).value,
                        Jsonb(_jsonable(capability)),
                    ),
                )

    def _execute(self, statement: str, values: tuple[Any, ...]) -> None:
        with self._pool.connection() as connection:
            connection.execute(statement, values)

    def _select(self, where: str, values: tuple[Any, ...]) -> list[Capability]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT payload FROM engineering_mcp_capabilities {where}
                ORDER BY capability_id
                """,
                values,
            ).fetchall()
        return [_capability(row[0]) for row in rows]


def _capability(payload: Mapping[str, Any]) -> Capability:
    values = dict(payload)
    kind = CapabilityType(values.pop("type"))
    values["tags"] = frozenset(values.get("tags", ()))
    values["depends_on"] = frozenset(values.get("depends_on", ()))
    if kind is CapabilityType.PROMPT:
        values["arguments"] = tuple(
            PromptArgument(**argument) for argument in values.get("arguments", ())
        )
    return _CAPABILITY_TYPES[kind](**values)


class PostgresRefreshStore(StateStoreABC):
    """Durable `RefreshEngine` fingerprints, so restarts refresh only changes."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the refresh state table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_refresh_state (
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
                "SELECT resource_id, payload FROM engineering_refresh_state"
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
                    "DELETE FROM engineering_refresh_state WHERE resource_id = %s",
                    (resource_id,),
                )
            for resource_id, state in puts.items():
                connection.execute(
                    """
                    INSERT INTO engineering_refresh_state (resource_id, payload)
                    VALUES (%s, %s)
                    ON CONFLICT (resource_id) DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    (resource_id, Jsonb(_jsonable(state))),
                )


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


def _event_types() -> dict[str, type[Any]]:
    found: dict[str, type[Any]] = {}
    pending: list[type[Any]] = [XAIEvent]
    while pending:
        current = pending.pop()
        found[current.__name__] = current
        pending.extend(current.__subclasses__())
    return found


_RECORD_TYPES: dict[str, type[Any]] = {
    "Execution": Execution,
    "ProvenanceLink": ProvenanceLink,
    **_event_types(),
}
type _Record = Execution | ProvenanceLink | XAIEvent


class PostgresProvenanceStore(ProvenanceStore):
    """Durable langgraph-xai executions, provenance links, and events.

    Records are upserted by type and ID (the latest write wins); queries are
    scoped to one application and tenant, and link lookups to one run.
    """

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the provenance table and its lookup indexes."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_xai_records (
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    application_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL,
                    sequence BIGINT NOT NULL,
                    source_id TEXT,
                    target_id TEXT,
                    payload JSONB NOT NULL,
                    PRIMARY KEY (record_type, record_id)
                );
                CREATE INDEX IF NOT EXISTS engineering_xai_records_id
                    ON engineering_xai_records (record_id);
                CREATE INDEX IF NOT EXISTS engineering_xai_records_scope
                    ON engineering_xai_records (application_id, tenant_id, run_id);
                CREATE INDEX IF NOT EXISTS engineering_xai_records_target
                    ON engineering_xai_records (run_id, target_id)
                    WHERE target_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS engineering_xai_records_source
                    ON engineering_xai_records (run_id, source_id)
                    WHERE source_id IS NOT NULL;
                """)

    async def write(self, item: _Record) -> None:
        """Insert or replace one record."""
        await asyncio.to_thread(self._write, item)

    async def get(self, entity_id: str) -> _Record | None:
        """Return the record with ID `entity_id`.

        Raises:
            LookupError: If records of different types share the ID.
        """
        rows = await asyncio.to_thread(
            self._rows,
            "SELECT record_type, payload FROM engineering_xai_records "
            "WHERE record_id = %s",
            (entity_id,),
        )
        if len(rows) > 1:
            raise LookupError("entity ID is ambiguous; use an isolation-aware query")
        return _load_record(rows[0]) if rows else None

    async def query(self, query: Any) -> AsyncIterator[_Record]:
        """Yield the records of one application and tenant in a stable order."""
        if not isinstance(query, StoreFilter):
            raise TypeError("PostgresProvenanceStore requires a StoreFilter")
        clauses = ["application_id = %s", "tenant_id = %s"]
        values: list[Any] = [query.application_id, query.tenant_id]
        if query.run_id is not None:
            clauses.append("run_id = %s")
            values.append(str(query.run_id))
        if query.item_type is not None:
            item_type = query.item_type
            clauses.append("record_type = ANY(%s)")
            values.append(
                [
                    name
                    for name, model in _RECORD_TYPES.items()
                    if issubclass(model, item_type)
                ]
            )
        values += [query.limit, query.offset]
        rows = await asyncio.to_thread(
            self._rows,
            f"""
            SELECT record_type, payload FROM engineering_xai_records
            WHERE {" AND ".join(clauses)}
            ORDER BY recorded_at, sequence, record_id
            LIMIT %s OFFSET %s
            """,
            tuple(values),
        )
        for row in rows:
            yield _load_record(row)

    async def parents(
        self, entity_id: str, *, context: ExecutionContext
    ) -> Sequence[ProvenanceLink]:
        """Return the links into `entity_id` within the context's run."""
        return await self._links("target_id", entity_id, context)

    async def children(
        self, entity_id: str, *, context: ExecutionContext
    ) -> Sequence[ProvenanceLink]:
        """Return the links out of `entity_id` within the context's run."""
        return await self._links("source_id", entity_id, context)

    async def close(self) -> None:
        """Keep the shared pool open; the application closes it."""

    async def _links(
        self, column: str, entity_id: str, context: ExecutionContext
    ) -> tuple[ProvenanceLink, ...]:
        rows = await asyncio.to_thread(
            self._rows,
            f"""
            SELECT record_type, payload FROM engineering_xai_records
            WHERE record_type = 'ProvenanceLink' AND {column} = %s
              AND application_id = %s AND tenant_id = %s AND run_id = %s
            ORDER BY recorded_at, sequence, record_id
            """,
            (entity_id, context.application_id, context.tenant_id, str(context.run_id)),
        )
        return tuple(ProvenanceLink.model_validate(row[1]) for row in rows)

    def _write(self, item: _Record) -> None:
        context = item.context
        link = isinstance(item, ProvenanceLink)
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO engineering_xai_records (record_type, record_id,
                    application_id, tenant_id, run_id, recorded_at, sequence,
                    source_id, target_id, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (record_type, record_id) DO UPDATE SET
                    application_id = EXCLUDED.application_id,
                    tenant_id = EXCLUDED.tenant_id,
                    run_id = EXCLUDED.run_id,
                    recorded_at = EXCLUDED.recorded_at,
                    sequence = EXCLUDED.sequence,
                    source_id = EXCLUDED.source_id,
                    target_id = EXCLUDED.target_id,
                    payload = EXCLUDED.payload
                """,
                (
                    type(item).__name__,
                    str(item.id),
                    context.application_id,
                    context.tenant_id,
                    str(context.run_id),
                    item.timestamp,
                    int(getattr(item, "sequence", -1)),
                    str(item.source_id) if link else None,
                    str(item.target_id) if link else None,
                    Jsonb(item.model_dump(mode="json")),
                ),
            )

    def _rows(self, statement: str, values: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        with self._pool.connection() as connection:
            return connection.execute(statement, values).fetchall()


def _load_record(row: tuple[Any, ...]) -> _Record:
    return _RECORD_TYPES[row[0]].model_validate(row[1])


class InvestigationLog:
    """The application's record of every investigation and its decision state."""

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the investigation table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_investigations (
                    investigation_id UUID PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    async def save(self, investigation: Investigation) -> None:
        """Insert or replace an investigation."""
        await asyncio.to_thread(self._save, investigation)

    async def get(self, investigation_id: UUID) -> Investigation | None:
        """Return one investigation."""
        return await asyncio.to_thread(self._get, investigation_id)

    async def update_decision(
        self, investigation_id: UUID, decision: ReleaseDecision
    ) -> Investigation | None:
        """Replace the release decision state of an investigation."""
        current = await self.get(investigation_id)
        if current is None:
            return None
        updated = current.model_copy(update={"release_decision": decision})
        await self.save(updated)
        return updated

    def _save(self, investigation: Investigation) -> None:
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO engineering_investigations (investigation_id, payload)
                VALUES (%s, %s)
                ON CONFLICT (investigation_id) DO UPDATE SET
                    payload = EXCLUDED.payload, updated_at = now()
                """,
                (
                    investigation.investigation_id,
                    Jsonb(investigation.model_dump(mode="json")),
                ),
            )

    def _get(self, investigation_id: UUID) -> Investigation | None:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                SELECT payload FROM engineering_investigations
                WHERE investigation_id = %s
                """,
                (investigation_id,),
            ).fetchone()
        return Investigation.model_validate(row[0]) if row else None


class ReleaseRegister:
    """The release register that deployment tooling reads.

    Publishing to it is the investigator's consequential side effect; the
    policy manager calls `ainvoke` at most once per investigation.
    """

    description = "Publish a validated go or no-go release decision."

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def initialize(self) -> None:
        """Create the register table."""
        with self._pool.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_release_register (
                    investigation_id UUID PRIMARY KEY,
                    decision TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    citations JSONB NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    async def ainvoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Publish one decision and return its JSON-compatible receipt."""
        return await asyncio.to_thread(self._publish, dict(arguments))

    def _publish(self, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO engineering_release_register
                    (investigation_id, decision, risk, summary, citations)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (investigation_id) DO NOTHING
                """,
                (
                    arguments["investigation_id"],
                    arguments["decision"],
                    arguments["risk"],
                    arguments["summary"],
                    Jsonb(list(arguments["citations"])),
                ),
            )
            row = connection.execute(
                """
                SELECT decision, recorded_at FROM engineering_release_register
                WHERE investigation_id = %s
                """,
                (arguments["investigation_id"],),
            ).fetchone()
        if row is None:
            raise RuntimeError("the release decision was not recorded")
        return {
            "investigation_id": str(arguments["investigation_id"]),
            "decision": row[0],
            "recorded_at": row[1].isoformat(),
        }


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, AbstractSet):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
