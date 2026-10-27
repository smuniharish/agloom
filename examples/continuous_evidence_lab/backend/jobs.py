"""Background execution with procrastinate.

* One `evidence.run` job advances one run: start, resume after a reviewer
  decision, or continue from the checkpoint after a crash.
* `RetryStrategy` retries transient failures with exponential backoff; the run
  is marked failed only when the last attempt fails.
* Aborting a running job (`cancel_job_by_id_async(..., abort=True)`) cancels
  the job's asyncio task; the task then stops its agent cooperatively with
  `Agent.stop()` and the job ends as aborted.
* Jobs of workers that died without a graceful shutdown are retried by a
  periodic task, and the retried job continues from the run's checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

from procrastinate import App, JobContext, RetryStrategy
from procrastinate.connector import BaseConnector
from procrastinate.exceptions import AlreadyEnqueued, JobAborted
from procrastinate.job_context import AbortReason
from psycopg import AsyncConnection

from .auth import Principal, principal_scope
from .config import Settings

if TYPE_CHECKING:
    from .runner import EvidenceRunner

RUN_TASK = "evidence.run"
RECOVER_TASK = "evidence.recover_stalled_jobs"
RUNNER_CONTEXT = "evidence_runner"
STOP_GRACE_SECONDS = 15.0


def run_lock(run_id: UUID) -> str:
    """Serializes the jobs of one run: at most one queued and one running."""
    return f"evidence-run:{run_id}"


def create_job_app(connector: BaseConnector, settings: Settings) -> App:
    """Create the procrastinate app with the evidence tasks."""
    app = App(connector=connector)
    retry = RetryStrategy(
        max_attempts=settings.max_attempts,
        exponential_wait=settings.retry_backoff_seconds,
    )

    @app.task(name=RUN_TASK, queue=settings.queue, pass_context=True, retry=retry)
    async def run_evidence(
        context: JobContext, run_id: str, tenant_id: str, user_id: str
    ) -> str:
        runner: EvidenceRunner = context.additional_context[RUNNER_CONTEXT]
        with principal_scope(Principal(subject=user_id, tenant_id=tenant_id)):
            return await execute_job(runner, context, UUID(run_id), retry)

    @app.periodic(cron="* * * * *", periodic_id="recover-stalled-evidence-jobs")
    @app.task(
        name=RECOVER_TASK,
        queue=settings.queue,
        queueing_lock=RECOVER_TASK,
        pass_context=True,
    )
    async def recover_stalled_jobs(context: JobContext, timestamp: int) -> int:
        del timestamp
        manager = context.app.job_manager
        stalled = list(
            await manager.get_stalled_jobs(
                queue=settings.queue,
                task_name=RUN_TASK,
                seconds_since_heartbeat=settings.stalled_after_seconds,
            )
        )
        for job in stalled:
            await manager.retry_job(job)
        return len(stalled)

    return app


async def execute_job(
    runner: EvidenceRunner, context: JobContext, run_id: UUID, retry: RetryStrategy
) -> str:
    """Run one job step; translate aborts and failures into run states."""
    execution = asyncio.ensure_future(
        runner.execute(run_id, attempt=context.job.attempts)
    )
    try:
        return await asyncio.shield(execution)
    except asyncio.CancelledError:
        runner.stop(run_id)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(execution, timeout=STOP_GRACE_SECONDS)
        if context.abort_reason() is AbortReason.USER_REQUEST:
            await runner.record_cancellation(run_id)
            raise JobAborted("the run was cancelled") from None
        # Shutdown aborts are retried; the next attempt resumes the checkpoint.
        stopped = JobAborted("the worker stopped before the run finished")
        if retry.get_retry_decision(exception=stopped, job=context.job) is None:
            await runner.record_transient_failure(
                run_id, stopped, final=True, attempt=context.job.attempts
            )
        raise stopped from None
    except Exception as error:
        final = retry.get_retry_decision(exception=error, job=context.job) is None
        await runner.record_transient_failure(
            run_id, error, final=final, attempt=context.job.attempts
        )
        raise


async def enqueue_run(
    app: App,
    run_id: UUID,
    *,
    tenant_id: str,
    user_id: str,
    connection: AsyncConnection[Any] | None = None,
) -> int:
    """Defer the job advancing a run, optionally inside the caller's transaction."""
    lock = run_lock(run_id)
    options: dict[str, Any] = {"lock": lock, "queueing_lock": lock}
    if connection is not None:
        options["connection"] = connection
    deferrer = app.configure_task(RUN_TASK, **options)
    arguments = {"run_id": str(run_id), "tenant_id": tenant_id, "user_id": user_id}
    if connection is None:
        return await deferrer.defer_async(**arguments)
    async with connection.transaction():
        return await deferrer.defer_async(**arguments)


async def queued_job(app: App, run_id: UUID) -> int | None:
    """Return the ID of the job already waiting for a run, if any."""
    jobs = await app.job_manager.list_jobs_async(
        task=RUN_TASK, queueing_lock=run_lock(run_id), status="todo"
    )
    for job in jobs:
        return job.id
    return None


async def enqueue_or_reuse(
    app: App,
    run_id: UUID,
    *,
    tenant_id: str,
    user_id: str,
    connection: AsyncConnection[Any] | None = None,
) -> int:
    """Defer the run's job, or return the job already queued for the run."""
    try:
        return await enqueue_run(
            app, run_id, tenant_id=tenant_id, user_id=user_id, connection=connection
        )
    except AlreadyEnqueued:
        queued = await queued_job(app, run_id)
        if queued is None:
            raise
        return queued
