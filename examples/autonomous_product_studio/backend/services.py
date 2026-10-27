"""Process-wide services, opened by the API lifespan and by the worker."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agentbarrier.errors import RuntimeStoreError
from agentbarrier.runtime import PostgresRuntimeStore, RuntimePolicy, RuntimeStore
from agloom import Observer
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from mcp_capability_router import MCPRuntime
from procrastinate import PsycopgConnector
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .governance import load_policy
from .jobs import JobQueue, job_app
from .mcp import open_mcp_runtime, playwright_server
from .observability import build_observers, shutdown_observers
from .safety import ExecutionBoundary, RemoteExecutionBoundary
from .settings import StudioSettings
from .store import StudioStore
from .streaming import RunEventBroker


@dataclass
class StudioServices:
    """Everything a request handler or a run job needs, owned by one process."""

    settings: StudioSettings
    store: Any
    events: RunEventBroker
    jobs: JobQueue
    policy: RuntimePolicy
    policy_store: RuntimeStore
    observers: tuple[Observer, ...] = ()
    checkpointer: BaseCheckpointSaver[Any] | None = None
    mcp: MCPRuntime | None = None
    model_factory: Callable[[], BaseChatModel] | None = None
    execution_boundary: ExecutionBoundary | None = None
    background: set[asyncio.Task[Any]] = field(default_factory=set)


def chat_model(settings: StudioSettings) -> BaseChatModel:
    """The Experiential Labs chat model; the key is read from the environment only."""
    key = os.getenv("EXPLABS_API_KEY")
    if not key:
        raise RuntimeError("EXPLABS_API_KEY is required")
    return ChatOpenAI(
        model=settings.model,
        api_key=key,
        base_url=settings.model_base_url,
        timeout=120,
        max_retries=2,
        use_responses_api=False,
    )


@asynccontextmanager
async def _checkpointer(
    database_url: str, *, enabled: bool
) -> AsyncIterator[BaseCheckpointSaver[Any] | None]:
    """LangGraph's saver on its own pool: it requires autocommit connections."""
    if not enabled:
        yield None
        return
    async with AsyncConnectionPool(
        database_url,
        max_size=10,
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        yield AsyncPostgresSaver(pool)


@asynccontextmanager
async def open_services(
    settings: StudioSettings, *, run_agents: bool
) -> AsyncIterator[StudioServices]:
    """Open the database pools, queue, policy store, and (for workers) the agent runtime.

    Fails fast with migration guidance when any schema is missing; this
    process never creates or alters schema.
    """
    # The queue pool is shared by procrastinate and run admission, so a run and
    # its job commit in one transaction.
    async with AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        open=False,
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        store = StudioStore(settings.database_url, async_pool=pool)
        missing = store.verify_schema()
        if missing:
            raise RuntimeError(
                "Product Studio schema is missing: "
                + ", ".join(sorted(missing))
                + ". Run `python -m backend.migrate`."
            )
        store.open()
        try:
            policy_store = PostgresRuntimeStore(
                settings.database_url, schema=settings.policy_schema
            )
        except RuntimeStoreError as error:
            store.close()
            raise RuntimeError(
                f"the agentbarrier schema {settings.policy_schema!r} is unavailable; "
                "run `python -m backend.migrate`"
            ) from error
        observers: tuple[Observer, ...] = ()
        try:
            # A fresh connector per lifetime: procrastinate keeps an external pool
            # referenced after closing, so a reused connector would hold a closed pool.
            with job_app.replace_connector(PsycopgConnector()):
                async with job_app.open_async(pool=pool):
                    observers = build_observers(database_url=settings.database_url)
                    server = (
                        playwright_server(
                            output_dir=settings.projects_root / ".playwright-mcp",
                            executable_path=settings.playwright_executable,
                            browser=settings.playwright_browser,
                        )
                        if run_agents and settings.browser_validation
                        else None
                    )
                    async with (
                        _checkpointer(
                            settings.database_url, enabled=run_agents
                        ) as saver,
                        open_mcp_runtime(server) as mcp,
                    ):
                        yield StudioServices(
                            settings=settings,
                            store=store,
                            events=RunEventBroker(store),
                            jobs=JobQueue(job_app),
                            policy=load_policy(
                                settings.policy_file, settings.preview_url
                            ),
                            policy_store=policy_store,
                            observers=observers,
                            checkpointer=saver,
                            mcp=mcp,
                            model_factory=(lambda: chat_model(settings))
                            if run_agents
                            else None,
                            execution_boundary=(
                                RemoteExecutionBoundary(
                                    settings.validation_executor_url,
                                    settings.projects_root,
                                )
                                if settings.validation_executor_url
                                else None
                            ),
                        )
        finally:
            shutdown_observers(observers)
            policy_store.close()
            store.close()
