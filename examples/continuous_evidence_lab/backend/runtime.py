"""Process resources: database pools, the checkpointer, agentbarrier, MCP, and jobs."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from agentbarrier.runtime import PostgresRuntimeStore, RuntimeStore
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from mcp_capability_router import MCPRuntime
from procrastinate import App, PsycopgConnector
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .agent import WorkspaceRefreshSource, chat_model
from .config import Settings
from .jobs import RUNNER_CONTEXT, create_job_app
from .mcp import mcp_servers, register_servers
from .observability import Telemetry, process_telemetry
from .policy import ApprovalDesk, load_policy
from .runner import EvidenceRunner, RunServices
from .store import EvidenceStore, Repository

CHECKPOINT_TABLES = frozenset(
    {"checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"}
)


def run_async[T](main: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on an event loop psycopg's async driver supports."""
    if sys.platform == "win32":
        return asyncio.run(main, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(main)


def job_connector(dsn: str, *, max_size: int = 5) -> PsycopgConnector:
    return PsycopgConnector(conninfo=dsn, min_size=1, max_size=max_size)


async def missing_platform_schema(dsn: str) -> set[str]:
    """Return the job-queue and checkpoint schema objects that are missing."""
    async with await psycopg.AsyncConnection.connect(dsn) as db:
        rows = await (
            await db.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = current_schema() AND table_name = ANY(%s)""",
                ([*CHECKPOINT_TABLES, "procrastinate_jobs"],),
            )
        ).fetchall()
        present = {row[0] for row in rows}
        missing = {
            table
            for table in (*CHECKPOINT_TABLES, "procrastinate_jobs")
            if table not in present
        }
        if "checkpoint_migrations" in present:
            version = await (
                await db.execute("SELECT max(v) FROM checkpoint_migrations")
            ).fetchone()
            expected = len(AsyncPostgresSaver.MIGRATIONS) - 1
            if version is None or version[0] is None or version[0] < expected:
                missing.add(f"checkpoint_migrations.v{expected}")
    return missing


@dataclass
class ApiResources:
    """What the API process needs: runs, the job queue, and reviewer decisions."""

    store: Repository
    jobs: App
    approvals: ApprovalDesk


@asynccontextmanager
async def open_api_resources(settings: Settings) -> AsyncIterator[ApiResources]:
    """Open the store, the job queue, and the approval desk; verify every schema."""
    dsn = settings.require_database()
    async with AsyncExitStack() as stack:
        store = EvidenceStore(dsn)
        missing = await store.verify_schema()
        missing |= await missing_platform_schema(dsn)
        if missing:
            raise RuntimeError(
                "Continuous Evidence Lab schema is missing: "
                + ", ".join(sorted(missing))
                + ". Run `python -m backend.setup_database`."
            )
        await store.open()
        stack.push_async_callback(store.close)
        jobs = create_job_app(job_connector(dsn), settings)
        await stack.enter_async_context(jobs.open_async())
        barrier = await asyncio.to_thread(
            PostgresRuntimeStore, dsn, schema=settings.barrier_schema
        )
        stack.callback(barrier.close)
        yield ApiResources(
            store=store,
            jobs=jobs,
            approvals=ApprovalDesk(
                barrier, require_separate_approver=settings.require_separate_approver
            ),
        )


@dataclass
class WorkerRuntime:
    """A worker's runner and the job app it serves."""

    jobs: App
    runner: EvidenceRunner
    telemetry: Telemetry
    settings: Settings

    async def run(self, **options: Any) -> None:
        """Process evidence jobs until the worker is stopped."""
        await self.jobs.run_worker_async(
            queues=[self.settings.queue],
            concurrency=self.settings.worker_concurrency,
            additional_context={RUNNER_CONTEXT: self.runner},
            **options,
        )


@asynccontextmanager
async def open_worker_runtime(
    settings: Settings,
    *,
    store: Repository | None = None,
    jobs: App | None = None,
    barrier: RuntimeStore | None = None,
    model_factory: Callable[[], BaseChatModel] | None = None,
    servers: Mapping[str, Any] | None = None,
) -> AsyncIterator[WorkerRuntime]:
    """Open everything a worker needs to execute and resume runs.

    `servers` maps MCP server IDs to `MCPRuntime.register_mcp` targets; by
    default the deployment's stdio servers from `mcp_servers()`. Each server's
    capabilities are discovered when it is registered (`register_servers`).
    """
    dsn = settings.require_database()
    async with AsyncExitStack() as stack:
        if store is None:
            evidence = EvidenceStore(dsn)
            missing = await evidence.verify_schema() | await missing_platform_schema(
                dsn
            )
            if missing:
                raise RuntimeError(
                    "Continuous Evidence Lab schema is missing: "
                    + ", ".join(sorted(missing))
                    + ". Run `python -m backend.setup_database`."
                )
            await evidence.open()
            stack.push_async_callback(evidence.close)
            store = evidence
        if jobs is None:
            jobs = create_job_app(
                job_connector(dsn, max_size=settings.worker_concurrency + 3), settings
            )
            await stack.enter_async_context(jobs.open_async())
        if barrier is None:
            barrier = await asyncio.to_thread(
                PostgresRuntimeStore, dsn, schema=settings.barrier_schema
            )
            stack.callback(barrier.close)
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            dsn,
            min_size=1,
            max_size=settings.worker_concurrency + 2,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        await pool.open()
        stack.push_async_callback(pool.close)
        mcp = await stack.enter_async_context(
            MCPRuntime(max_concurrency=4, operation_timeout=45.0)
        )
        await register_servers(mcp, mcp_servers() if servers is None else servers)
        telemetry = process_telemetry("worker")
        runner = EvidenceRunner(
            RunServices(
                store=store,
                checkpointer=AsyncPostgresSaver(pool),
                barrier=barrier,
                policy=load_policy(settings.policy_file),
                telemetry=telemetry,
                model_factory=model_factory or chat_model,
                mcp=mcp,
                refresh_source=WorkspaceRefreshSource(),
            )
        )
        yield WorkerRuntime(
            jobs=jobs, runner=runner, telemetry=telemetry, settings=settings
        )
