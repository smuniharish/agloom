"""The run worker: `python -m backend.worker`.

Runs procrastinate's worker with the studio services, so jobs build agents
with the PostgreSQL checkpointer, the agentbarrier policy store, and the
process-wide Playwright MCP runtime.
"""

from __future__ import annotations

import asyncio
import logging
import os

from .jobs import RUN_QUEUE, job_app
from .services import StudioServices, open_services
from .settings import StudioSettings


async def run_worker(
    services: StudioServices, *, install_signal_handlers: bool = True, wait: bool = True
) -> None:
    """Process run jobs until stopped (or until the queue is empty with `wait=False`)."""
    settings = services.settings
    await job_app.run_worker_async(
        queues=[RUN_QUEUE],
        name="studio-worker",
        concurrency=settings.worker_concurrency,
        wait=wait,
        fetch_job_polling_interval=settings.worker_poll_interval,
        abort_job_polling_interval=settings.worker_poll_interval,
        shutdown_graceful_timeout=settings.shutdown_grace_seconds,
        additional_context={"services": services},
        install_signal_handlers=install_signal_handlers,
    )


def event_loop_factory() -> type[asyncio.AbstractEventLoop] | None:
    """psycopg's asynchronous driver needs a selector event loop on Windows."""
    return asyncio.SelectorEventLoop if os.name == "nt" else None


async def serve() -> None:
    settings = StudioSettings.from_env()
    if settings.validation_executor_url is None:
        raise RuntimeError(
            "VALIDATION_EXECUTOR_URL must configure an isolated validator"
        )
    async with open_services(settings, run_agents=True) as services:
        await run_worker(services)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(serve(), loop_factory=event_loop_factory())


if __name__ == "__main__":
    main()
