"""PostgreSQL persistence: pgvector retrieval, runs, the iteration timeline,
reviewer approvals, the published evidence ledger, and the run event log.

Every query is scoped to the tenant of the current principal.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import httpx
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .auth import current_principal
from .models import (
    Accuracy,
    Approval,
    Iteration,
    RunRequest,
    RunView,
    Source,
    Synthesis,
    empty_accuracy,
)

DATA = Path(__file__).parent.parent / "data" / "sources.json"
SCHEMA_REVISION = "0004"

Enqueue = Callable[[AsyncConnection[Any] | None, UUID], Awaitable[int]]

_REQUIRED_TABLES: dict[str, set[str]] = {
    "sources": {
        "id",
        "title",
        "uri",
        "revision",
        "body",
        "retrieved_at",
        "embedding",
        "tenant_id",
    },
    "loop_runs": {
        "id",
        "objective",
        "topics",
        "request",
        "status",
        "execution_status",
        "objective_state",
        "termination",
        "accuracy",
        "synthesis",
        "error",
        "approval",
        "idempotency_key",
        "job_id",
        "published_at",
        "created_at",
        "updated_at",
        "tenant_id",
        "user_id",
    },
    "claims": {"run_id", "claim_id", "topic", "text", "status", "tenant_id"},
    "evidence": {"run_id", "claim_id", "source_id", "quote", "valid", "tenant_id"},
    "loop_iterations": {"run_id", "iteration", "observation", "decision", "tenant_id"},
    "decisions": {
        "id",
        "run_id",
        "action_id",
        "decision",
        "note",
        "created_at",
        "tenant_id",
        "user_id",
    },
    "run_events": {
        "sequence",
        "tenant_id",
        "run_id",
        "event_type",
        "payload",
        "created_at",
    },
}


class Embeddings(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddings:
    """Embeds text with an Ollama embedding model (`all-minilm` by default)."""

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.url}/api/embed", json={"model": self.model, "input": texts}
            )
        response.raise_for_status()
        values = response.json().get("embeddings")
        if not isinstance(values, list) or len(values) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return values


@dataclass(frozen=True)
class RunState:
    """Execution bookkeeping of one run, not exposed by the API."""

    status: str
    execution_status: str
    job_id: int | None
    approval: Approval | None


class Repository(Protocol):
    """Persistence used by the API and the worker."""

    async def ping(self) -> None: ...
    async def admit_run(
        self, request: RunRequest, *, idempotency_key: str | None, enqueue: Enqueue
    ) -> tuple[UUID, bool]: ...
    async def get_run(self, run_id: UUID) -> RunView | None: ...
    async def get_request(self, run_id: UUID) -> RunRequest | None: ...
    async def run_state(self, run_id: UUID) -> RunState | None: ...
    async def search(self, query: str, limit: int = 8) -> list[Source]: ...
    async def upsert_source(self, source: Source) -> bool: ...
    async def delete_source(self, source_id: str) -> bool: ...
    async def mark_running(self, run_id: UUID) -> None: ...
    async def await_reconciliation(self, run_id: UUID, message: str) -> None: ...
    async def save_iteration(
        self, run_id: UUID, item: Iteration, decision: Mapping[str, Any]
    ) -> None: ...
    async def latest_iteration(self, run_id: UUID) -> Iteration | None: ...
    async def await_approval(
        self,
        run_id: UUID,
        synthesis: Synthesis,
        accuracy: Accuracy,
        approval: Approval,
    ) -> None: ...
    async def record_decision(
        self,
        run_id: UUID,
        decision: str,
        note: str | None,
        *,
        enqueue: Enqueue,
    ) -> RunView | None: ...
    async def publish(
        self, run_id: UUID, synthesis: Synthesis, accuracy: Accuracy
    ) -> datetime: ...
    async def requeue_paused(self, run_id: UUID, *, enqueue: Enqueue) -> bool: ...
    async def published_synthesis(self, run_id: UUID) -> Synthesis | None: ...
    async def complete_run(
        self,
        run_id: UUID,
        *,
        termination: str | None,
        synthesis: Synthesis | None,
        accuracy: Accuracy | None,
    ) -> None: ...
    async def reject_run(
        self, run_id: UUID, *, termination: str, error: str | None = None
    ) -> None: ...
    async def fail_run(
        self, run_id: UUID, error: str, *, termination: str = "execution_failed"
    ) -> None: ...
    async def cancel_run(self, run_id: UUID, reason: str) -> bool: ...
    async def set_job(self, run_id: UUID, job_id: int) -> None: ...
    async def append_event(
        self, run_id: UUID, event_type: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...
    async def read_events(
        self, run_id: UUID, after: int = 0
    ) -> list[dict[str, Any]]: ...


async def _register_vector(connection: AsyncConnection[Any]) -> None:
    try:
        await register_vector_async(connection)
    except psycopg.ProgrammingError:
        # The vector extension is created by the migrations; verify_schema
        # reports a database that has not been migrated yet.
        await connection.rollback()
    else:
        await connection.commit()


class EvidenceStore:
    """PostgreSQL/pgvector evidence retrieval and the complete audit trail."""

    def __init__(
        self,
        dsn: str,
        *,
        embeddings: Embeddings | None = None,
        max_connections: int = 10,
    ) -> None:
        self.dsn = dsn
        self.embeddings = embeddings or OllamaEmbeddings()
        self._max_connections = max_connections
        self._pool: AsyncConnectionPool[AsyncConnection[Any]] | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        pool: AsyncConnectionPool[AsyncConnection[Any]] = AsyncConnectionPool(
            self.dsn,
            min_size=1,
            max_size=self._max_connections,
            timeout=10,
            open=False,
            configure=_register_vector,
        )
        await pool.open(wait=False)
        self._pool = pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection[Any]]:
        """Borrow a pooled connection; the block commits unless it raises."""
        if self._pool is None:
            raise RuntimeError("the evidence store is not open")
        async with self._pool.connection() as connection:
            yield connection

    async def verify_schema(self) -> set[str]:
        """Return the tables, columns, or revision the database is missing."""
        async with await AsyncConnection.connect(self.dsn) as db:
            cursor = await db.execute(
                """
                SELECT table_name, column_name FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = ANY(%s)
                """,
                (list(_REQUIRED_TABLES),),
            )
            rows = await cursor.fetchall()
            try:
                revision = await (
                    await db.execute("SELECT version_num FROM alembic_version")
                ).fetchone()
            except psycopg.errors.UndefinedTable:
                revision = None
        present: dict[str, set[str]] = {}
        for table_name, column_name in rows:
            present.setdefault(table_name, set()).add(column_name)
        missing: set[str] = set()
        for table, columns in _REQUIRED_TABLES.items():
            have = present.get(table)
            if not have:
                missing.add(table)
            else:
                missing.update(f"{table}.{column}" for column in columns - have)
        if revision is None or revision[0] != SCHEMA_REVISION:
            missing.add(f"alembic_version.{SCHEMA_REVISION}")
        return missing

    async def ping(self) -> None:
        async with self.connection() as db:
            await db.execute("SELECT 1")

    async def admit_run(
        self, request: RunRequest, *, idempotency_key: str | None, enqueue: Enqueue
    ) -> tuple[UUID, bool]:
        """Insert the run and defer its job in one transaction.

        A repeated `Idempotency-Key` from the same user returns the existing run.
        """
        principal = current_principal()
        async with self.connection() as db, db.transaction():
            if idempotency_key:
                await db.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{principal.tenant_id}:{principal.subject}:{idempotency_key}",),
                )
                existing = await (
                    await db.execute(
                        """SELECT id FROM loop_runs WHERE tenant_id=%s AND user_id=%s
                           AND idempotency_key=%s""",
                        (principal.tenant_id, principal.subject, idempotency_key),
                    )
                ).fetchone()
                if existing is not None:
                    return existing[0], False
            run_id = uuid4()
            await db.execute(
                """INSERT INTO loop_runs
                   (id, objective, topics, request, status, execution_status,
                    objective_state, accuracy, idempotency_key, tenant_id, user_id)
                   VALUES (%s,%s,%s,%s,'running','queued','incomplete',%s,%s,%s,%s)""",
                (
                    run_id,
                    request.objective,
                    Jsonb(request.topics),
                    Jsonb(request.model_dump(mode="json")),
                    Jsonb(empty_accuracy(request.topics).model_dump()),
                    idempotency_key,
                    principal.tenant_id,
                    principal.subject,
                ),
            )
            job_id = await enqueue(db, run_id)
            await db.execute(
                "UPDATE loop_runs SET job_id=%s WHERE id=%s AND tenant_id=%s",
                (job_id, run_id, principal.tenant_id),
            )
        return run_id, True

    async def get_run(self, run_id: UUID) -> RunView | None:
        tenant = current_principal().tenant_id
        async with self.connection() as db:
            cursor = db.cursor(row_factory=dict_row)
            row = await (
                await cursor.execute(
                    "SELECT * FROM loop_runs WHERE id=%s AND tenant_id=%s",
                    (run_id, tenant),
                )
            ).fetchone()
            if row is None:
                return None
            iterations = await (
                await cursor.execute(
                    """SELECT observation FROM loop_iterations
                       WHERE run_id=%s AND tenant_id=%s ORDER BY iteration""",
                    (run_id, tenant),
                )
            ).fetchall()
            synthesis = (
                Synthesis.model_validate(row["synthesis"]) if row["synthesis"] else None
            )
            cited = sorted(
                {
                    citation.source_id
                    for claim in (synthesis.claims if synthesis else [])
                    for citation in claim.citations
                }
            )
            sources = await (
                await cursor.execute(
                    """SELECT id,title,uri,revision,body,retrieved_at,1 AS similarity
                       FROM sources WHERE tenant_id=%s AND id = ANY(%s) ORDER BY id""",
                    (tenant, cited),
                )
            ).fetchall()
        request = RunRequest.model_validate(row["request"])
        return RunView(
            id=row["id"],
            objective=row["objective"],
            topics=row["topics"],
            mode=request.mode,
            status=row["status"],
            execution_status=row["execution_status"],
            objective_state=row["objective_state"],
            termination=row["termination"],
            error=row["error"],
            synthesis=synthesis,
            accuracy=Accuracy.model_validate(row["accuracy"]),
            timeline=[
                Iteration.model_validate(item["observation"]) for item in iterations
            ],
            sources=[Source.model_validate(item) for item in sources],
            approval=Approval.model_validate(row["approval"])
            if row["approval"]
            else None,
            published_at=row["published_at"],
            created_at=row["created_at"],
        )

    async def get_request(self, run_id: UUID) -> RunRequest | None:
        async with self.connection() as db:
            row = await (
                await db.execute(
                    "SELECT request FROM loop_runs WHERE id=%s AND tenant_id=%s",
                    (run_id, current_principal().tenant_id),
                )
            ).fetchone()
        return RunRequest.model_validate(row[0]) if row else None

    async def run_state(self, run_id: UUID) -> RunState | None:
        async with self.connection() as db:
            row = await (
                await db.execute(
                    """SELECT status, execution_status, job_id, approval FROM loop_runs
                       WHERE id=%s AND tenant_id=%s""",
                    (run_id, current_principal().tenant_id),
                )
            ).fetchone()
        if row is None:
            return None
        return RunState(
            status=row[0],
            execution_status=row[1],
            job_id=row[2],
            approval=Approval.model_validate(row[3]) if row[3] else None,
        )

    async def _seed_sources(self) -> None:
        tenant = current_principal().tenant_id
        async with self.connection() as db:
            count = await (
                await db.execute(
                    "SELECT count(*) FROM sources WHERE tenant_id=%s", (tenant,)
                )
            ).fetchone()
        if count and count[0]:
            return
        items = [
            Source.model_validate(item)
            for item in json.loads(DATA.read_text(encoding="utf-8"))
        ]
        vectors = await self.embeddings.embed([item.body for item in items])
        async with self.connection() as db:
            for item, vector in zip(items, vectors, strict=True):
                await db.execute(
                    """INSERT INTO sources
                       (id,title,uri,revision,body,embedding,tenant_id)
                       VALUES(%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (tenant_id,id) DO NOTHING""",
                    (
                        item.id,
                        item.title,
                        item.uri,
                        item.revision,
                        item.body,
                        Vector(vector),
                        tenant,
                    ),
                )

    async def search(self, query: str, limit: int = 8) -> list[Source]:
        await self._seed_sources()
        tenant = current_principal().tenant_id
        vector = Vector((await self.embeddings.embed([query]))[0])
        async with self.connection() as db:
            rows = await (
                await db.cursor(row_factory=dict_row).execute(
                    """SELECT id,title,uri,revision,body,retrieved_at,
                       greatest(0, 1-(embedding <=> %s)) AS similarity
                       FROM sources WHERE tenant_id=%s
                       ORDER BY embedding <=> %s LIMIT %s""",
                    (vector, tenant, vector, limit),
                )
            ).fetchall()
        return [Source.model_validate(row) for row in rows]

    async def upsert_source(self, source: Source) -> bool:
        tenant = current_principal().tenant_id
        async with self.connection() as db:
            current = await (
                await db.execute(
                    "SELECT revision FROM sources WHERE id=%s AND tenant_id=%s",
                    (source.id, tenant),
                )
            ).fetchone()
        if current is not None and current[0] == source.revision:
            return False
        vector = (await self.embeddings.embed([source.body]))[0]
        async with self.connection() as db:
            await db.execute(
                """INSERT INTO sources
                   (id,title,uri,revision,body,retrieved_at,embedding,tenant_id)
                   VALUES(%s,%s,%s,%s,%s,now(),%s,%s)
                   ON CONFLICT(tenant_id,id) DO UPDATE SET title=excluded.title,
                   uri=excluded.uri, revision=excluded.revision, body=excluded.body,
                   retrieved_at=now(), embedding=excluded.embedding""",
                (
                    source.id,
                    source.title,
                    source.uri,
                    source.revision,
                    source.body,
                    Vector(vector),
                    tenant,
                ),
            )
        return True

    async def delete_source(self, source_id: str) -> bool:
        async with self.connection() as db:
            cursor = await db.execute(
                "DELETE FROM sources WHERE id=%s AND tenant_id=%s",
                (source_id, current_principal().tenant_id),
            )
            return cursor.rowcount > 0

    async def mark_running(self, run_id: UUID) -> None:
        async with self.connection() as db:
            await db.execute(
                """UPDATE loop_runs SET execution_status='running', error=NULL,
                   updated_at=now() WHERE id=%s AND tenant_id=%s
                   AND execution_status IN ('queued','running','paused')""",
                (run_id, current_principal().tenant_id),
            )

    async def await_reconciliation(self, run_id: UUID, message: str) -> None:
        """Hold a run whose publication outcome is unknown until its retry runs."""
        async with self.connection() as db:
            await db.execute(
                """UPDATE loop_runs SET execution_status='paused', error=%s,
                   updated_at=now() WHERE id=%s AND tenant_id=%s
                   AND execution_status='running'""",
                (message[:2000], run_id, current_principal().tenant_id),
            )

    async def save_iteration(
        self, run_id: UUID, item: Iteration, decision: Mapping[str, Any]
    ) -> None:
        tenant = current_principal().tenant_id
        async with self.connection() as db:
            await db.execute(
                """INSERT INTO loop_iterations
                   (run_id,iteration,observation,decision,tenant_id)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT(run_id,iteration)
                   DO UPDATE SET observation=excluded.observation,
                   decision=excluded.decision""",
                (
                    run_id,
                    item.iteration,
                    Jsonb(item.model_dump(mode="json")),
                    Jsonb(dict(decision)),
                    tenant,
                ),
            )

    async def latest_iteration(self, run_id: UUID) -> Iteration | None:
        async with self.connection() as db:
            row = await (
                await db.execute(
                    """SELECT observation FROM loop_iterations
                       WHERE run_id=%s AND tenant_id=%s
                       ORDER BY iteration DESC LIMIT 1""",
                    (run_id, current_principal().tenant_id),
                )
            ).fetchone()
        return Iteration.model_validate(row[0]) if row else None

    async def await_approval(
        self,
        run_id: UUID,
        synthesis: Synthesis,
        accuracy: Accuracy,
        approval: Approval,
    ) -> None:
        async with self.connection() as db:
            await db.execute(
                """UPDATE loop_runs SET status='awaiting_approval',
                   execution_status='paused', objective_state='awaiting_approval',
                   synthesis=%s, accuracy=%s, approval=%s, error=NULL,
                   updated_at=now()
                   WHERE id=%s AND tenant_id=%s
                   AND status IN ('running','awaiting_approval')""",
                (
                    Jsonb(synthesis.model_dump(mode="json")),
                    Jsonb(accuracy.model_dump(mode="json")),
                    Jsonb(approval.model_dump(mode="json")),
                    run_id,
                    current_principal().tenant_id,
                ),
            )

    async def record_decision(
        self,
        run_id: UUID,
        decision: str,
        note: str | None,
        *,
        enqueue: Enqueue,
    ) -> RunView | None:
        """Record the reviewer decision and defer the resume job atomically."""
        principal = current_principal()
        async with self.connection() as db, db.transaction():
            row = await (
                await db.execute(
                    """SELECT approval FROM loop_runs WHERE id=%s AND tenant_id=%s
                       AND status='awaiting_approval' FOR UPDATE""",
                    (run_id, principal.tenant_id),
                )
            ).fetchone()
            if row is None or row[0] is None:
                return None
            approval = Approval.model_validate(row[0])
            if approval.decision is not None:
                return None
            decided = approval.model_copy(
                update={
                    "decision": decision,
                    "decided_by": principal.subject,
                    "decided_at": datetime.now(UTC),
                    "note": note,
                }
            )
            await db.execute(
                """INSERT INTO decisions
                   (id,run_id,action_id,decision,note,created_at,tenant_id,user_id)
                   VALUES(%s,%s,%s,%s,%s,now(),%s,%s)""",
                (
                    uuid4(),
                    run_id,
                    approval.action_id,
                    decision,
                    note,
                    principal.tenant_id,
                    principal.subject,
                ),
            )
            rejected = decision == "reject"
            await db.execute(
                """UPDATE loop_runs SET approval=%s, execution_status='queued',
                   status = CASE WHEN %s THEN 'rejected' ELSE status END,
                   objective_state = CASE WHEN %s THEN 'failed'
                                          ELSE objective_state END,
                   updated_at=now() WHERE id=%s AND tenant_id=%s""",
                (
                    Jsonb(decided.model_dump(mode="json")),
                    rejected,
                    rejected,
                    run_id,
                    principal.tenant_id,
                ),
            )
            job_id = await enqueue(db, run_id)
            await db.execute(
                "UPDATE loop_runs SET job_id=%s WHERE id=%s AND tenant_id=%s",
                (job_id, run_id, principal.tenant_id),
            )
        return await self.get_run(run_id)

    async def requeue_paused(self, run_id: UUID, *, enqueue: Enqueue) -> bool:
        """Queue the resume job of a paused run, e.g. after its approval expired."""
        tenant = current_principal().tenant_id
        async with self.connection() as db, db.transaction():
            cursor = await db.execute(
                """UPDATE loop_runs SET execution_status='queued', updated_at=now()
                   WHERE id=%s AND tenant_id=%s AND execution_status='paused'""",
                (run_id, tenant),
            )
            if cursor.rowcount != 1:
                return False
            job_id = await enqueue(db, run_id)
            await db.execute(
                "UPDATE loop_runs SET job_id=%s WHERE id=%s AND tenant_id=%s",
                (job_id, run_id, tenant),
            )
        return True

    async def publish(
        self, run_id: UUID, synthesis: Synthesis, accuracy: Accuracy
    ) -> datetime:
        """Write the approved claims and evidence ledger exactly once per run."""
        tenant = current_principal().tenant_id
        async with self.connection() as db, db.transaction():
            row = await (
                await db.execute(
                    """SELECT published_at FROM loop_runs
                       WHERE id=%s AND tenant_id=%s FOR UPDATE""",
                    (run_id, tenant),
                )
            ).fetchone()
            if row is None:
                raise LookupError(f"run {run_id} was not found")
            if row[0] is not None:
                return row[0]
            cited = {
                citation.source_id
                for claim in synthesis.claims
                for citation in claim.citations
            }
            bodies = dict(
                await (
                    await db.execute(
                        """SELECT id, body FROM sources
                           WHERE tenant_id=%s AND id = ANY(%s)""",
                        (tenant, sorted(cited)),
                    )
                ).fetchall()
            )
            await db.execute(
                "DELETE FROM evidence WHERE run_id=%s AND tenant_id=%s",
                (run_id, tenant),
            )
            await db.execute(
                "DELETE FROM claims WHERE run_id=%s AND tenant_id=%s", (run_id, tenant)
            )
            for claim in synthesis.claims:
                await db.execute(
                    """INSERT INTO claims(run_id,claim_id,topic,text,status,tenant_id)
                       VALUES(%s,%s,%s,%s,%s,%s)""",
                    (run_id, claim.id, claim.topic, claim.text, claim.status, tenant),
                )
                for citation in claim.citations:
                    body = bodies.get(citation.source_id, "")
                    await db.execute(
                        """INSERT INTO evidence
                           (run_id,claim_id,source_id,quote,valid,tenant_id)
                           VALUES(%s,%s,%s,%s,%s,%s)""",
                        (
                            run_id,
                            claim.id,
                            citation.source_id,
                            citation.quote,
                            citation.quote.casefold() in body.casefold(),
                            tenant,
                        ),
                    )
            published = await (
                await db.execute(
                    """UPDATE loop_runs SET status='approved',
                       objective_state='validated', synthesis=%s, accuracy=%s,
                       published_at=now(), updated_at=now()
                       WHERE id=%s AND tenant_id=%s RETURNING published_at""",
                    (
                        Jsonb(synthesis.model_dump(mode="json")),
                        Jsonb(accuracy.model_dump(mode="json")),
                        run_id,
                        tenant,
                    ),
                )
            ).fetchone()
        if published is None:
            raise LookupError(f"run {run_id} was not found")
        return published[0]

    async def published_synthesis(self, run_id: UUID) -> Synthesis | None:
        async with self.connection() as db:
            row = await (
                await db.execute(
                    """SELECT synthesis FROM loop_runs WHERE id=%s AND tenant_id=%s
                       AND published_at IS NOT NULL""",
                    (run_id, current_principal().tenant_id),
                )
            ).fetchone()
        return Synthesis.model_validate(row[0]) if row and row[0] else None

    async def complete_run(
        self,
        run_id: UUID,
        *,
        termination: str | None,
        synthesis: Synthesis | None,
        accuracy: Accuracy | None,
    ) -> None:
        """Finish a run; runs without published evidence end as unvalidated."""
        async with self.connection() as db:
            await db.execute(
                """UPDATE loop_runs SET execution_status='completed', termination=%s,
                   error=NULL, updated_at=now(),
                   synthesis = CASE WHEN status='approved' THEN synthesis
                                    ELSE COALESCE(%s, synthesis) END,
                   accuracy = CASE WHEN status='approved' THEN accuracy
                                   ELSE COALESCE(%s, accuracy) END,
                   objective_state = CASE WHEN status='approved' THEN 'validated'
                                          WHEN status='rejected' THEN 'failed'
                                          ELSE 'incomplete' END,
                   status = CASE WHEN status IN ('approved','rejected','cancelled')
                                 THEN status ELSE 'unvalidated' END
                   WHERE id=%s AND tenant_id=%s
                   AND execution_status NOT IN ('cancelled','failed')""",
                (
                    termination,
                    Jsonb(synthesis.model_dump(mode="json")) if synthesis else None,
                    Jsonb(accuracy.model_dump(mode="json")) if accuracy else None,
                    run_id,
                    current_principal().tenant_id,
                ),
            )

    async def reject_run(
        self, run_id: UUID, *, termination: str, error: str | None = None
    ) -> None:
        """Finish a run whose publication was rejected or not decided in time."""
        async with self.connection() as db:
            await db.execute(
                """UPDATE loop_runs SET status='rejected', objective_state='failed',
                   execution_status='completed', termination=%s, error=%s,
                   updated_at=now()
                   WHERE id=%s AND tenant_id=%s
                   AND execution_status <> 'cancelled' AND status <> 'approved'""",
                (termination, error, run_id, current_principal().tenant_id),
            )

    async def fail_run(
        self, run_id: UUID, error: str, *, termination: str = "execution_failed"
    ) -> None:
        async with self.connection() as db:
            await db.execute(
                """UPDATE loop_runs SET execution_status='failed', termination=%s,
                   error=%s, updated_at=now(),
                   objective_state = CASE WHEN status='approved' THEN 'validated'
                                          ELSE 'failed' END,
                   status = CASE WHEN status IN ('approved','rejected','cancelled')
                                 THEN status ELSE 'failed' END
                   WHERE id=%s AND tenant_id=%s
                   AND execution_status <> 'cancelled'""",
                (termination, error[:2000], run_id, current_principal().tenant_id),
            )

    async def cancel_run(self, run_id: UUID, reason: str) -> bool:
        async with self.connection() as db:
            cursor = await db.execute(
                """UPDATE loop_runs SET status='cancelled',
                   execution_status='cancelled', objective_state='failed',
                   termination='cancelled', error=%s, updated_at=now()
                   WHERE id=%s AND tenant_id=%s
                   AND execution_status IN ('queued','running','paused')
                   AND status NOT IN ('approved','rejected')""",
                (reason[:2000], run_id, current_principal().tenant_id),
            )
            return cursor.rowcount == 1

    async def set_job(self, run_id: UUID, job_id: int) -> None:
        async with self.connection() as db:
            await db.execute(
                "UPDATE loop_runs SET job_id=%s WHERE id=%s AND tenant_id=%s",
                (job_id, run_id, current_principal().tenant_id),
            )

    async def append_event(
        self, run_id: UUID, event_type: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        async with self.connection() as db:
            row = await (
                await db.execute(
                    """INSERT INTO run_events (tenant_id, run_id, event_type, payload)
                       VALUES (%s,%s,%s,%s) RETURNING sequence, created_at""",
                    (
                        current_principal().tenant_id,
                        run_id,
                        event_type,
                        Jsonb(json.loads(json.dumps(dict(payload), default=str))),
                    ),
                )
            ).fetchone()
        if row is None:
            raise RuntimeError("the run event was not stored")
        return _event(row[0], event_type, payload, row[1])

    async def read_events(self, run_id: UUID, after: int = 0) -> list[dict[str, Any]]:
        async with self.connection() as db:
            rows = await (
                await db.execute(
                    """SELECT sequence, event_type, payload, created_at FROM run_events
                       WHERE tenant_id=%s AND run_id=%s AND sequence > %s
                       ORDER BY sequence""",
                    (current_principal().tenant_id, run_id, after),
                )
            ).fetchall()
        return [_event(row[0], row[1], row[2], row[3]) for row in rows]


def _event(
    sequence: int, event_type: str, payload: Mapping[str, Any], created_at: datetime
) -> dict[str, Any]:
    return {
        **json.loads(json.dumps(dict(payload), default=str)),
        "id": sequence,
        "type": event_type,
        "timestamp": created_at.isoformat(),
    }
