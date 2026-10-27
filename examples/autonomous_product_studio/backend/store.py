from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

import httpx
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from .auth import Principal, current_principal
from .models import (
    ACTIVE_RUN_STATUSES,
    WAITING_RUN_STATUSES,
    Approval,
    Artifact,
    Evidence,
    LoopRun,
    Project,
    ProjectCreate,
    RunProof,
    Task,
    TaskCreate,
)

EMBEDDING_DIMENSION = 384
SCHEMA_REVISION = "0005"
EVENT_PAGE_SIZE = 500

Enqueue = Callable[[UUID, Principal, Any], Awaitable[int]]
"""Defers a run job, optionally on the caller's transaction; returns the job id."""

_REQUIRED_TABLES: dict[str, set[str]] = {
    "studio_projects": {
        "id",
        "name",
        "objective",
        "acceptance_criteria",
        "embedding",
        "created_at",
        "tenant_id",
        "user_id",
    },
    "studio_tasks": {
        "id",
        "project_id",
        "title",
        "objective",
        "status",
        "created_at",
        "tenant_id",
        "user_id",
    },
    "studio_loop_runs": {
        "id",
        "task_id",
        "status",
        "proof",
        "result",
        "approvals",
        "job_id",
        "idempotency_key",
        "created_at",
        "finished_at",
        "tenant_id",
        "user_id",
    },
    "studio_evidence": {
        "id",
        "run_id",
        "iteration",
        "criterion_id",
        "source",
        "passed",
        "summary",
        "details",
        "created_at",
        "tenant_id",
    },
    "studio_artifacts": {
        "id",
        "project_id",
        "path",
        "kind",
        "sha256",
        "metadata",
        "embedding",
        "created_at",
        "tenant_id",
    },
    "studio_run_events": {
        "sequence",
        "run_id",
        "ownership",
        "event_type",
        "payload",
        "created_at",
    },
    "procrastinate_jobs": {"id", "status", "lock", "queueing_lock", "args"},
    "checkpoints": {"thread_id", "checkpoint_ns", "checkpoint_id", "checkpoint"},
    "checkpoint_blobs": {"thread_id", "channel", "version"},
    "checkpoint_writes": {"thread_id", "task_id", "idx"},
    "checkpoint_migrations": {"v"},
}
_ACTIVE = tuple(sorted(ACTIVE_RUN_STATUSES))
_WAITING = tuple(sorted(WAITING_RUN_STATUSES))


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddings:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return vectors


def _now() -> datetime:
    return datetime.now(UTC)


def _ownership(principal: Principal) -> dict[str, str]:
    return {"tenant_id": principal.tenant_id, "user_id": principal.subject}


def _event_message(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row["payload"],
        "id": row["sequence"],
        "type": row["event_type"],
        "timestamp": row["created_at"].isoformat(),
    }


def _run(row: dict[str, Any]) -> LoopRun:
    return LoopRun.model_validate(
        {
            key: row[key]
            for key in (
                "id",
                "task_id",
                "status",
                "proof",
                "result",
                "approvals",
                "created_at",
                "finished_at",
            )
        }
    )


class StudioStore:
    """PostgreSQL persistence; every query is scoped to the current tenant."""

    def __init__(
        self,
        dsn: str,
        embedder: Embedder | None = None,
        *,
        async_pool: AsyncConnectionPool | None = None,
        max_connections: int = 10,
    ) -> None:
        self.dsn = dsn
        self.embedder = embedder or OllamaEmbeddings()
        self.async_pool = async_pool
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=max_connections,
            open=False,
            configure=register_vector,
            kwargs={"row_factory": dict_row},
        )

    def open(self) -> None:
        self._pool.open(wait=True, timeout=30)

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        with self._pool.connection() as db:
            yield db

    def verify_schema(self) -> set[str]:
        """Return the tables or columns this code needs but the database lacks."""
        with psycopg.connect(self.dsn, connect_timeout=5) as db:
            rows = db.execute(
                """
                SELECT table_name, column_name FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = ANY(%s)
                """,
                (list(_REQUIRED_TABLES),),
            ).fetchall()
            versioned = db.execute(
                "SELECT to_regclass('alembic_version') IS NOT NULL"
            ).fetchone()
            revision = (
                db.execute("SELECT version_num FROM alembic_version").fetchone()
                if versioned and versioned[0]
                else None
            )
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

    def ping(self) -> None:
        with psycopg.connect(self.dsn, connect_timeout=2) as db:
            db.execute("SELECT 1")

    def create_project(self, request: ProjectCreate) -> Project:
        project = Project(
            id=uuid4(),
            name=request.name,
            objective=request.objective,
            acceptance_criteria=request.acceptance_criteria,
            created_at=_now(),
        )
        vector = Vector(self.embedder.embed([request.objective])[0])
        principal = current_principal()
        with self.connect() as db:
            db.execute(
                """INSERT INTO studio_projects
                   (id,name,objective,acceptance_criteria,embedding,created_at,
                    tenant_id,user_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    project.id,
                    project.name,
                    project.objective,
                    Jsonb([item.model_dump() for item in project.acceptance_criteria]),
                    vector,
                    project.created_at,
                    principal.tenant_id,
                    principal.subject,
                ),
            )
        return project

    def list_projects(self) -> list[Project]:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            rows = db.execute(
                """SELECT id,name,objective,acceptance_criteria,created_at
                   FROM studio_projects WHERE tenant_id=%s ORDER BY created_at""",
                (tenant,),
            ).fetchall()
        return [Project.model_validate(row) for row in rows]

    def get_project(self, project_id: UUID) -> Project | None:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                """SELECT id,name,objective,acceptance_criteria,created_at
                   FROM studio_projects WHERE id=%s AND tenant_id=%s""",
                (project_id, tenant),
            ).fetchone()
        return Project.model_validate(row) if row else None

    def create_task(self, project_id: UUID, request: TaskCreate) -> Task:
        task = Task(
            id=uuid4(),
            project_id=project_id,
            title=request.title,
            objective=request.objective,
            status="queued",
            created_at=_now(),
        )
        principal = current_principal()
        with self.connect() as db:
            inserted = db.execute(
                """INSERT INTO studio_tasks
                   (id,project_id,title,objective,status,created_at,tenant_id,user_id)
                   SELECT %s,%s,%s,%s,%s,%s,%s,%s
                   WHERE EXISTS (SELECT 1 FROM studio_projects
                                 WHERE id=%s AND tenant_id=%s)""",
                (
                    task.id,
                    task.project_id,
                    task.title,
                    task.objective,
                    task.status,
                    task.created_at,
                    principal.tenant_id,
                    principal.subject,
                    project_id,
                    principal.tenant_id,
                ),
            )
            if inserted.rowcount != 1:
                raise LookupError("project not found")
        return task

    def get_task(self, task_id: UUID) -> Task | None:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                """SELECT id,project_id,title,objective,status,created_at
                   FROM studio_tasks WHERE id=%s AND tenant_id=%s""",
                (task_id, tenant),
            ).fetchone()
        return Task.model_validate(row) if row else None

    def get_run(self, run_id: UUID) -> LoopRun | None:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM studio_loop_runs WHERE id=%s AND tenant_id=%s",
                (run_id, tenant),
            ).fetchone()
        return _run(row) if row else None

    def run_job_id(self, run_id: UUID) -> int | None:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                "SELECT job_id FROM studio_loop_runs WHERE id=%s AND tenant_id=%s",
                (run_id, tenant),
            ).fetchone()
        return row["job_id"] if row else None

    def run_for_idempotency_key(self, task_id: UUID, key: str) -> LoopRun | None:
        principal = current_principal()
        with self.connect() as db:
            row = db.execute(
                """SELECT * FROM studio_loop_runs
                   WHERE tenant_id=%s AND user_id=%s AND task_id=%s
                     AND idempotency_key=%s""",
                (principal.tenant_id, principal.subject, task_id, key),
            ).fetchone()
        return _run(row) if row else None

    async def schedule_run(
        self, task_id: UUID, *, idempotency_key: str | None, enqueue: Enqueue
    ) -> tuple[LoopRun, bool]:
        """Create a queued run and defer its job in one transaction.

        Returns the run and whether it was created. A repeated idempotency key
        returns the existing run. Raises `ValueError` when the task already has
        an active run and `LookupError` when the task does not exist.
        """
        if self.async_pool is None:
            raise RuntimeError("scheduling requires the asynchronous connection pool")
        principal = current_principal()
        run = LoopRun(
            id=uuid4(),
            task_id=task_id,
            status="queued",
            proof=RunProof(),
            created_at=_now(),
        )
        async with self.async_pool.connection() as conn, conn.transaction():
            cursor = conn.cursor(row_factory=dict_row)
            task = await (
                await cursor.execute(
                    """SELECT id FROM studio_tasks WHERE id=%s AND tenant_id=%s
                       FOR UPDATE""",
                    (task_id, principal.tenant_id),
                )
            ).fetchone()
            if task is None:
                raise LookupError("task not found")
            if idempotency_key:
                existing = await (
                    await cursor.execute(
                        """SELECT * FROM studio_loop_runs
                           WHERE tenant_id=%s AND user_id=%s AND task_id=%s
                             AND idempotency_key=%s""",
                        (
                            principal.tenant_id,
                            principal.subject,
                            task_id,
                            idempotency_key,
                        ),
                    )
                ).fetchone()
                if existing is not None:
                    return _run(existing), False
            active = await (
                await cursor.execute(
                    """SELECT 1 FROM studio_loop_runs
                       WHERE tenant_id=%s AND task_id=%s AND status = ANY(%s)""",
                    (principal.tenant_id, task_id, list(_ACTIVE)),
                )
            ).fetchone()
            if active is not None:
                raise ValueError("task already has an active run")
            await cursor.execute(
                """INSERT INTO studio_loop_runs
                   (id,task_id,status,proof,result,approvals,idempotency_key,
                    created_at,tenant_id,user_id)
                   VALUES (%s,%s,%s,%s,'{}','[]',%s,%s,%s,%s)""",
                (
                    run.id,
                    task_id,
                    run.status,
                    Jsonb(run.proof.model_dump()),
                    idempotency_key,
                    run.created_at,
                    principal.tenant_id,
                    principal.subject,
                ),
            )
            await cursor.execute(
                "UPDATE studio_tasks SET status='queued' WHERE id=%s AND tenant_id=%s",
                (task_id, principal.tenant_id),
            )
            job_id = await enqueue(run.id, principal, conn)
            await cursor.execute(
                "UPDATE studio_loop_runs SET job_id=%s WHERE id=%s",
                (job_id, run.id),
            )
        return run, True

    async def requeue_run(self, run_id: UUID, *, enqueue: Enqueue) -> LoopRun:
        """Queue a paused run again (after its approvals or reconciliations)."""
        if self.async_pool is None:
            raise RuntimeError("scheduling requires the asynchronous connection pool")
        principal = current_principal()
        async with self.async_pool.connection() as conn, conn.transaction():
            cursor = conn.cursor(row_factory=dict_row)
            row = await (
                await cursor.execute(
                    """UPDATE studio_loop_runs SET status='queued'
                       WHERE id=%s AND tenant_id=%s AND status = ANY(%s)
                       RETURNING *""",
                    (run_id, principal.tenant_id, list(_WAITING)),
                )
            ).fetchone()
            if row is None:
                raise ValueError("run is not waiting for a person")
            await cursor.execute(
                "UPDATE studio_tasks SET status='queued' WHERE id=%s AND tenant_id=%s",
                (row["task_id"], principal.tenant_id),
            )
            owner = Principal(row["user_id"], principal.tenant_id)
            job_id = await enqueue(run_id, owner, conn)
            await cursor.execute(
                "UPDATE studio_loop_runs SET job_id=%s WHERE id=%s", (job_id, run_id)
            )
        return _run({**row, "status": "queued"})

    def mark_running(self, run_id: UUID) -> LoopRun | None:
        """Move a queued run to running; `None` when it is no longer runnable."""
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                """UPDATE studio_loop_runs SET status='running'
                   WHERE id=%s AND tenant_id=%s AND status IN ('queued','running')
                   RETURNING *""",
                (run_id, tenant),
            ).fetchone()
            if row is not None:
                db.execute(
                    "UPDATE studio_tasks SET status='running' WHERE id=%s AND tenant_id=%s",
                    (row["task_id"], tenant),
                )
        return _run(row) if row else None

    def record_pause(
        self, run_id: UUID, status: str, items: list[Approval], proof: RunProof
    ) -> LoopRun | None:
        """Park a running run until people decide `items` (`status` is a waiting state)."""
        if status not in WAITING_RUN_STATUSES:
            raise ValueError(f"{status!r} is not a waiting state")
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                """UPDATE studio_loop_runs
                   SET status=%s, approvals=%s, proof=%s
                   WHERE id=%s AND tenant_id=%s AND status IN ('queued','running')
                   RETURNING *""",
                (
                    status,
                    Jsonb([item.model_dump(mode="json") for item in items]),
                    Jsonb(proof.model_dump()),
                    run_id,
                    tenant,
                ),
            ).fetchone()
            if row is not None:
                db.execute(
                    "UPDATE studio_tasks SET status=%s WHERE id=%s AND tenant_id=%s",
                    (status, row["task_id"], tenant),
                )
        return _run(row) if row else None

    def finish_run(
        self, run_id: UUID, status: str, proof: RunProof, result: dict[str, Any]
    ) -> LoopRun | None:
        """Record a terminal outcome once; later calls leave the run unchanged."""
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                """UPDATE studio_loop_runs
                   SET status=%s, proof=%s, result=%s, finished_at=now()
                   WHERE id=%s AND tenant_id=%s AND status = ANY(%s)
                   RETURNING *""",
                (
                    status,
                    Jsonb(proof.model_dump()),
                    Jsonb(result),
                    run_id,
                    tenant,
                    list(_ACTIVE),
                ),
            ).fetchone()
            if row is not None:
                db.execute(
                    "UPDATE studio_tasks SET status=%s WHERE id=%s AND tenant_id=%s",
                    (status, row["task_id"], tenant),
                )
        return _run(row) if row else None

    @contextmanager
    def publication_lock(self, project_id: UUID) -> Iterator[None]:
        """Serialize publications of one project across workers."""
        with self.connect() as db, db.transaction():
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"studio-publication:{project_id}",),
            )
            yield

    def add_evidence(
        self,
        run_id: UUID,
        iteration: int,
        criterion_id: str,
        source: str,
        passed: bool,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> Evidence:
        evidence = Evidence(
            id=uuid4(),
            run_id=run_id,
            iteration=iteration,
            criterion_id=criterion_id,
            source=source,
            passed=passed,
            summary=summary,
            details=details or {},
            created_at=_now(),
        )
        tenant = current_principal().tenant_id
        with self.connect() as db:
            row = db.execute(
                """INSERT INTO studio_evidence
                   (id,run_id,iteration,criterion_id,source,passed,summary,details,
                    created_at,tenant_id)
                   SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                   WHERE EXISTS (SELECT 1 FROM studio_loop_runs
                                 WHERE id=%s AND tenant_id=%s)
                   ON CONFLICT (tenant_id, run_id, iteration, criterion_id)
                   DO UPDATE SET source=EXCLUDED.source, passed=EXCLUDED.passed,
                       summary=EXCLUDED.summary, details=EXCLUDED.details,
                       created_at=EXCLUDED.created_at
                   RETURNING *""",
                (
                    evidence.id,
                    evidence.run_id,
                    evidence.iteration,
                    evidence.criterion_id,
                    evidence.source,
                    evidence.passed,
                    evidence.summary,
                    Jsonb(evidence.details),
                    evidence.created_at,
                    tenant,
                    run_id,
                    tenant,
                ),
            ).fetchone()
            if row is None:
                raise LookupError("run not found")
        return Evidence.model_validate(row)

    def evidence_for_run(self, run_id: UUID) -> list[Evidence]:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM studio_evidence WHERE run_id=%s AND tenant_id=%s
                   ORDER BY iteration,criterion_id""",
                (run_id, tenant),
            ).fetchall()
        return [Evidence.model_validate(row) for row in rows]

    def index_artifact(
        self, project_id: UUID, path: str, kind: str, content: str
    ) -> Artifact:
        artifact = Artifact(
            id=uuid4(),
            project_id=project_id,
            path=path,
            kind=kind,
            sha256=hashlib.sha256(content.encode()).hexdigest(),
            created_at=_now(),
        )
        vector = Vector(self.embedder.embed([content[:10_000]])[0])
        tenant = current_principal().tenant_id
        with self.connect() as db:
            inserted = db.execute(
                """INSERT INTO studio_artifacts
                   (id,project_id,path,kind,sha256,metadata,embedding,created_at,
                    tenant_id)
                   SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s
                   WHERE EXISTS (SELECT 1 FROM studio_projects
                                 WHERE id=%s AND tenant_id=%s)""",
                (
                    artifact.id,
                    artifact.project_id,
                    artifact.path,
                    artifact.kind,
                    artifact.sha256,
                    Jsonb(artifact.metadata),
                    vector,
                    artifact.created_at,
                    tenant,
                    project_id,
                    tenant,
                ),
            )
            if inserted.rowcount != 1:
                raise LookupError("project not found")
        return artifact

    def search_artifacts(
        self, project_id: UUID, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        vector = Vector(self.embedder.embed([query])[0])
        tenant = current_principal().tenant_id
        with self.connect() as db:
            rows = db.execute(
                """SELECT path,kind,sha256,1-(embedding <=> %s) AS score
                   FROM studio_artifacts WHERE project_id=%s AND tenant_id=%s
                   ORDER BY embedding <=> %s LIMIT %s""",
                (vector, project_id, tenant, vector, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def append_event(
        self, run_id: UUID, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        principal = current_principal()
        with self.connect() as db:
            row = db.execute(
                """INSERT INTO studio_run_events (run_id, ownership, event_type, payload)
                   VALUES (%s, %s, %s, %s)
                   RETURNING sequence, event_type, payload, created_at""",
                (run_id, Jsonb(_ownership(principal)), event_type, Jsonb(payload)),
            ).fetchone()
        assert row is not None
        return _event_message(row)

    def read_events(self, run_id: UUID, *, after: int = 0) -> list[dict[str, Any]]:
        tenant = current_principal().tenant_id
        with self.connect() as db:
            rows = db.execute(
                """SELECT sequence, event_type, payload, created_at
                   FROM studio_run_events
                   WHERE run_id=%s AND ownership->>'tenant_id'=%s AND sequence > %s
                   ORDER BY sequence LIMIT %s""",
                (run_id, tenant, after, EVENT_PAGE_SIZE),
            ).fetchall()
        return [_event_message(row) for row in rows]

    def counts(self) -> dict[str, int]:
        tenant = current_principal().tenant_id
        names = ("projects", "tasks", "loop_runs", "artifacts", "evidence")
        with self.connect() as db:
            return {
                name: db.execute(
                    f"SELECT count(*) AS n FROM studio_{name} WHERE tenant_id=%s",
                    (tenant,),
                ).fetchone()["n"]
                for name in names
            }


class MemoryStudioStore:
    """In-process store with the same tenant rules, for tests and local tooling."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder
        self.projects: dict[UUID, Project] = {}
        self.tasks: dict[UUID, Task] = {}
        self.runs: dict[UUID, LoopRun] = {}
        self.run_jobs: dict[UUID, int] = {}
        self.run_keys: dict[tuple[str, str, UUID, str], UUID] = {}
        self.run_users: dict[UUID, str] = {}
        self.evidence: dict[tuple[UUID, int, str], Evidence] = {}
        self.artifacts: list[tuple[Artifact, list[float], str]] = []
        self.events: list[tuple[UUID, str, dict[str, Any]]] = []
        self.owners: dict[UUID, str] = {}
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._publication_locks: dict[UUID, threading.Lock] = {}

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def verify_schema(self) -> set[str]:
        return set()

    def ping(self) -> None:
        return None

    def _owned(self, key: UUID) -> bool:
        return self.owners.get(key) == current_principal().tenant_id

    def create_project(self, request: ProjectCreate) -> Project:
        value = Project(id=uuid4(), created_at=_now(), **request.model_dump())
        with self._lock:
            self.projects[value.id] = value
            self.owners[value.id] = current_principal().tenant_id
        return value

    def list_projects(self) -> list[Project]:
        return [item for key, item in self.projects.items() if self._owned(key)]

    def get_project(self, project_id: UUID) -> Project | None:
        return self.projects.get(project_id) if self._owned(project_id) else None

    def create_task(self, project_id: UUID, request: TaskCreate) -> Task:
        if self.get_project(project_id) is None:
            raise LookupError("project not found")
        value = Task(
            id=uuid4(),
            project_id=project_id,
            status="queued",
            created_at=_now(),
            **request.model_dump(),
        )
        with self._lock:
            self.tasks[value.id] = value
            self.owners[value.id] = current_principal().tenant_id
        return value

    def get_task(self, task_id: UUID) -> Task | None:
        return self.tasks.get(task_id) if self._owned(task_id) else None

    def _set_task_status(self, task_id: UUID, status: str) -> None:
        task = self.tasks.get(task_id)
        if task is not None:
            self.tasks[task_id] = task.model_copy(update={"status": status})

    def get_run(self, run_id: UUID) -> LoopRun | None:
        return self.runs.get(run_id) if self._owned(run_id) else None

    def run_job_id(self, run_id: UUID) -> int | None:
        return self.run_jobs.get(run_id) if self._owned(run_id) else None

    def run_for_idempotency_key(self, task_id: UUID, key: str) -> LoopRun | None:
        principal = current_principal()
        run_id = self.run_keys.get(
            (principal.tenant_id, principal.subject, task_id, key)
        )
        return self.get_run(run_id) if run_id else None

    async def schedule_run(
        self, task_id: UUID, *, idempotency_key: str | None, enqueue: Enqueue
    ) -> tuple[LoopRun, bool]:
        principal = current_principal()
        async with self._async_lock:
            if self.get_task(task_id) is None:
                raise LookupError("task not found")
            if idempotency_key:
                existing = self.run_for_idempotency_key(task_id, idempotency_key)
                if existing is not None:
                    return existing, False
            if any(
                run.task_id == task_id and run.status in ACTIVE_RUN_STATUSES
                for key, run in self.runs.items()
                if self._owned(key)
            ):
                raise ValueError("task already has an active run")
            run = LoopRun(
                id=uuid4(),
                task_id=task_id,
                status="queued",
                proof=RunProof(),
                created_at=_now(),
            )
            with self._lock:
                self.runs[run.id] = run
                self.owners[run.id] = principal.tenant_id
                self.run_users[run.id] = principal.subject
                if idempotency_key:
                    self.run_keys[
                        (
                            principal.tenant_id,
                            principal.subject,
                            task_id,
                            idempotency_key,
                        )
                    ] = run.id
                self._set_task_status(task_id, "queued")
            self.run_jobs[run.id] = await enqueue(run.id, principal, None)
        return run, True

    async def requeue_run(self, run_id: UUID, *, enqueue: Enqueue) -> LoopRun:
        principal = current_principal()
        async with self._async_lock:
            run = self.get_run(run_id)
            if run is None or run.status not in WAITING_RUN_STATUSES:
                raise ValueError("run is not waiting for a person")
            queued = run.model_copy(update={"status": "queued"})
            with self._lock:
                self.runs[run_id] = queued
                self._set_task_status(run.task_id, "queued")
            owner = Principal(self.run_users[run_id], principal.tenant_id)
            self.run_jobs[run_id] = await enqueue(run_id, owner, None)
        return queued

    def _transition(
        self, run_id: UUID, allowed: set[str], **changes: Any
    ) -> LoopRun | None:
        with self._lock:
            run = self.get_run(run_id)
            if run is None or run.status not in allowed:
                return None
            updated = run.model_copy(update=changes)
            self.runs[run_id] = updated
            self._set_task_status(run.task_id, updated.status)
            return updated

    def mark_running(self, run_id: UUID) -> LoopRun | None:
        return self._transition(run_id, {"queued", "running"}, status="running")

    def record_pause(
        self, run_id: UUID, status: str, items: list[Approval], proof: RunProof
    ) -> LoopRun | None:
        if status not in WAITING_RUN_STATUSES:
            raise ValueError(f"{status!r} is not a waiting state")
        return self._transition(
            run_id,
            {"queued", "running"},
            status=status,
            approvals=list(items),
            proof=proof,
        )

    def finish_run(
        self, run_id: UUID, status: str, proof: RunProof, result: dict[str, Any]
    ) -> LoopRun | None:
        return self._transition(
            run_id,
            set(ACTIVE_RUN_STATUSES),
            status=status,
            proof=proof,
            result=result,
            finished_at=_now(),
        )

    @contextmanager
    def publication_lock(self, project_id: UUID) -> Iterator[None]:
        with self._lock:
            lock = self._publication_locks.setdefault(project_id, threading.Lock())
        with lock:
            yield

    def add_evidence(
        self,
        run_id: UUID,
        iteration: int,
        criterion_id: str,
        source: str,
        passed: bool,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> Evidence:
        if self.get_run(run_id) is None:
            raise LookupError("run not found")
        value = Evidence(
            id=uuid4(),
            run_id=run_id,
            iteration=iteration,
            criterion_id=criterion_id,
            source=source,
            passed=passed,
            summary=summary,
            details=details or {},
            created_at=_now(),
        )
        with self._lock:
            self.evidence[(run_id, iteration, criterion_id)] = value
        return value

    def evidence_for_run(self, run_id: UUID) -> list[Evidence]:
        if self.get_run(run_id) is None:
            return []
        return sorted(
            (item for item in self.evidence.values() if item.run_id == run_id),
            key=lambda item: (item.iteration, item.criterion_id),
        )

    def index_artifact(
        self, project_id: UUID, path: str, kind: str, content: str
    ) -> Artifact:
        if self.get_project(project_id) is None:
            raise LookupError("project not found")
        value = Artifact(
            id=uuid4(),
            project_id=project_id,
            path=path,
            kind=kind,
            sha256=hashlib.sha256(content.encode()).hexdigest(),
            created_at=_now(),
        )
        vector = self.embedder.embed([content])[0] if self.embedder else [0.0]
        with self._lock:
            self.artifacts.append((value, vector, content))
            self.owners[value.id] = current_principal().tenant_id
        return value

    def search_artifacts(
        self, project_id: UUID, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed([query])[0] if self.embedder else [0.0]

        def cosine(vector: list[float]) -> float:
            denominator = math.sqrt(sum(x * x for x in vector)) * math.sqrt(
                sum(x * x for x in query_vector)
            )
            return (
                sum(a * b for a, b in zip(vector, query_vector, strict=True))
                / denominator
                if denominator
                else 0.0
            )

        candidates = [
            (cosine(vector), artifact)
            for artifact, vector, _ in self.artifacts
            if artifact.project_id == project_id and self._owned(artifact.id)
        ]
        return [
            {
                "path": item.path,
                "kind": item.kind,
                "sha256": item.sha256,
                "score": score,
            }
            for score, item in sorted(
                candidates, key=lambda pair: pair[0], reverse=True
            )[:limit]
        ]

    def append_event(
        self, run_id: UUID, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        tenant = current_principal().tenant_id
        with self._lock:
            row = {
                "sequence": len(self.events) + 1,
                "event_type": event_type,
                "payload": dict(payload),
                "created_at": _now(),
            }
            self.events.append((run_id, tenant, row))
        return _event_message(row)

    def read_events(self, run_id: UUID, *, after: int = 0) -> list[dict[str, Any]]:
        tenant = current_principal().tenant_id
        with self._lock:
            rows = [
                row
                for event_run, event_tenant, row in self.events
                if event_run == run_id
                and event_tenant == tenant
                and row["sequence"] > after
            ]
        return [_event_message(row) for row in rows[:EVENT_PAGE_SIZE]]

    def counts(self) -> dict[str, int]:
        tenant = current_principal().tenant_id
        return {
            "projects": sum(self.owners[key] == tenant for key in self.projects),
            "tasks": sum(self.owners[key] == tenant for key in self.tasks),
            "loop_runs": sum(self.owners[key] == tenant for key in self.runs),
            "artifacts": sum(
                self.owners[item.id] == tenant for item, _, _ in self.artifacts
            ),
            "evidence": sum(
                self.owners.get(item.run_id) == tenant
                for item in self.evidence.values()
            ),
        }
