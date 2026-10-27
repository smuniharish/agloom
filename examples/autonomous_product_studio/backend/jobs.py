"""Background execution with procrastinate.

One job executes (or continues) one run. Its lock is the run, so a run never
executes twice at the same time; its queueing lock stops duplicate queued
jobs. Aborting the job stops the agent cooperatively at the next model or tool
boundary instead of interrupting a tool mid-flight.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import psycopg
from procrastinate import App, JobContext, PsycopgConnector, RetryStrategy
from procrastinate.exceptions import JobAborted
from procrastinate.job_context import AbortReason
from procrastinate.jobs import Status

from .auth import Principal
from .execution import RunExecution, discard_run_state

if TYPE_CHECKING:
    from .services import StudioServices

_LOGGER = logging.getLogger(__name__)
RUN_QUEUE = "studio-runs"
STALLED_WORKER_SECONDS = 30.0


class RunInterrupted(Exception):
    """The worker stopped before the run finished; the run resumes on retry."""


RUN_RETRY = RetryStrategy(
    max_attempts=5,
    exponential_wait=2,
    retry_exceptions=[
        RunInterrupted,
        psycopg.OperationalError,
        ConnectionError,
        TimeoutError,
    ],
)

job_app = App(connector=PsycopgConnector())


def _services(context: JobContext) -> StudioServices:
    return context.additional_context["services"]


@job_app.task(
    name="studio.execute_run", queue=RUN_QUEUE, pass_context=True, retry=RUN_RETRY
)
async def execute_run(
    context: JobContext, run_id: str, tenant_id: str, user_id: str
) -> str:
    """Execute a run, or continue it from its checkpoint after a crash or approval."""
    execution = RunExecution(
        _services(context), UUID(run_id), Principal(user_id, tenant_id)
    )
    try:
        return await _until_done_or_aborted(context, execution)
    except JobAborted:
        raise
    except Exception as error:
        retried = (
            RUN_RETRY.get_retry_decision(exception=error, job=context.job) is not None
        )
        await execution.record_failure(
            error, retrying=retried, attempt=context.job.attempts + 1
        )
        raise


async def _until_done_or_aborted(context: JobContext, execution: RunExecution) -> str:
    work = asyncio.ensure_future(execution.run())
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        if work.done():
            raise
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
    cancelled = context.abort_reason() is AbortReason.USER_REQUEST
    execution.stop("cancelled" if cancelled else "interrupted")
    outcome = await work
    if outcome == "cancelled":
        raise JobAborted("run cancelled")
    if outcome == "interrupted":
        raise RunInterrupted("the worker stopped during the run")
    return outcome


@job_app.task(name="studio.discard_run", queue=RUN_QUEUE, pass_context=True)
async def discard_run(
    context: JobContext, run_id: str, tenant_id: str, user_id: str
) -> None:
    """Delete the checkpoint thread and private workspace of a finished run."""
    await discard_run_state(
        _services(context), UUID(run_id), Principal(user_id, tenant_id)
    )


@job_app.periodic(cron="* * * * *", periodic_id="stalled-runs")
@job_app.task(
    name="studio.recover_stalled_runs",
    queue=RUN_QUEUE,
    queueing_lock="studio.recover_stalled_runs",
    pass_context=True,
)
async def recover_stalled_runs(context: JobContext, timestamp: int) -> int:
    """Requeue run jobs whose worker stopped sending heartbeats."""
    del timestamp
    manager = context.app.job_manager
    stalled = await manager.get_stalled_jobs(
        task_name=execute_run.name, seconds_since_heartbeat=STALLED_WORKER_SECONDS
    )
    for job in stalled:
        _LOGGER.warning("requeueing stalled run job %s", job.id)
        await manager.retry_job(job)
    return len(stalled)


class JobQueue:
    """The application's view of the procrastinate queue."""

    def __init__(self, app: App) -> None:
        self.app = app

    async def enqueue_run(
        self, run_id: UUID, principal: Principal, connection: Any = None
    ) -> int:
        lock = f"run:{run_id}"
        return await execute_run.configure(
            lock=lock, queueing_lock=lock, connection=connection
        ).defer_async(
            run_id=str(run_id),
            tenant_id=principal.tenant_id,
            user_id=principal.subject,
        )

    async def enqueue_discard(self, run_id: UUID, principal: Principal) -> int:
        return await discard_run.configure(lock=f"run:{run_id}").defer_async(
            run_id=str(run_id),
            tenant_id=principal.tenant_id,
            user_id=principal.subject,
        )

    async def cancel(self, job_id: int) -> Literal["cancelled", "aborting", "finished"]:
        """Cancel a queued job, or ask the worker running it to stop the agent."""
        manager = self.app.job_manager
        if not await manager.cancel_job_by_id_async(job_id, abort=True):
            return "finished"
        status = await manager.get_job_status_async(job_id)
        return "cancelled" if status is Status.CANCELLED else "aborting"

    async def ping(self) -> bool:
        return await self.app.check_connection_async()
