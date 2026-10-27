from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from agentbarrier.runtime import RuntimeStatus, SQLiteRuntimeStore
from backend import jobs as job_module
from backend.agent import WorkspaceRefreshSource
from backend.config import POLICY_FILE, Settings
from backend.jobs import (
    RECOVER_TASK,
    RUN_TASK,
    RUNNER_CONTEXT,
    enqueue_run,
    execute_job,
)
from backend.mcp import register_servers
from backend.models import RunRequest
from backend.observability import Telemetry
from backend.policy import ApprovalDesk, load_policy
from backend.runner import EvidenceRunner, PublicationOutcomeUnknown, RunServices
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from mcp_capability_router import MCPRuntime
from procrastinate import App, JobContext, RetryStrategy
from procrastinate.exceptions import JobAborted
from procrastinate.job_context import AbortReason
from procrastinate.jobs import Job
from pydantic import ConfigDict

from tests.support import (
    SOURCES,
    TEST_PRINCIPAL,
    EvidenceScript,
    MemoryStore,
    ScriptedChatModel,
    fake_mcp_servers,
    synthesis,
)

TOPICS = ["encryption", "incident response"]


class BlockingModel(ScriptedChatModel):
    """Blocks inside one model call to simulate a crash or a long provider call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    block_at: int = 0
    entered: Any = None
    release: Any = None

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.block_at and len(self.calls) + 1 == self.block_at:
            self.block_at = 0
            self.entered.set()
            await self.release.wait()
        return self._respond(messages, **kwargs)


def blocking(script: EvidenceScript, block_at: int) -> BlockingModel:
    return BlockingModel(
        responder=script,
        block_at=block_at,
        entered=asyncio.Event(),
        release=asyncio.Event(),
    )


def runner_for(
    store: MemoryStore,
    model: ScriptedChatModel,
    barrier: SQLiteRuntimeStore,
    *,
    saver: InMemorySaver | None = None,
    mcp: MCPRuntime | None = None,
    refresh_source: WorkspaceRefreshSource | None = None,
) -> EvidenceRunner:
    return EvidenceRunner(
        RunServices(
            store=store,
            checkpointer=saver or InMemorySaver(),
            barrier=barrier,
            policy=load_policy(POLICY_FILE),
            telemetry=Telemetry(),
            model_factory=lambda: model,
            mcp=mcp,
            refresh_source=refresh_source,
        )
    )


def enqueuer(app: App):
    async def enqueue(connection: Any, run_id: UUID) -> int:
        return await enqueue_run(
            app,
            run_id,
            tenant_id=TEST_PRINCIPAL.tenant_id,
            user_id=TEST_PRINCIPAL.subject,
            connection=connection,
        )

    return enqueue


async def admit(
    store: MemoryStore,
    app: App,
    *,
    mode: str = "loop",
    target_url: str | None = None,
) -> UUID:
    request = RunRequest(
        objective="Verify Acme security and incident response evidence.",
        topics=TOPICS,
        mode=mode,
        target_url=target_url,
    )
    run_id, created = await store.admit_run(
        request, idempotency_key=None, enqueue=enqueuer(app)
    )
    assert created
    return run_id


async def work(app: App, runner: EvidenceRunner) -> None:
    await app.run_worker_async(
        queues=["evidence"],
        wait=False,
        additional_context={RUNNER_CONTEXT: runner},
        install_signal_handlers=False,
    )


async def decide(
    store: MemoryStore,
    barrier: SQLiteRuntimeStore,
    app: App,
    run_id: UUID,
    *,
    approve: bool = True,
    reviewer: str = "reviewer-1",
) -> None:
    view = await store.get_run(run_id)
    assert view is not None
    assert view.approval is not None
    await ApprovalDesk(barrier).decide(
        view.approval.action_id,
        approve=approve,
        reviewer=reviewer,
        tenant_id=TEST_PRINCIPAL.tenant_id,
    )
    decided = await store.record_decision(
        run_id, "approve" if approve else "reject", None, enqueue=enqueuer(app)
    )
    assert decided is not None


def job_context(
    app: App, runner: EvidenceRunner, run_id: UUID, *, attempts: int = 0, reason=None
) -> JobContext:
    job = Job(
        id=99,
        queue="evidence",
        lock=job_module.run_lock(run_id),
        queueing_lock=job_module.run_lock(run_id),
        task_name=RUN_TASK,
        task_kwargs={"run_id": str(run_id)},
        attempts=attempts,
    )
    return JobContext(
        app=app,
        job=job,
        start_timestamp=time.time(),
        abort_reason=lambda: reason,
        additional_context={RUNNER_CONTEXT: runner},
    )


@pytest.fixture
async def mcp_runtime() -> AsyncIterator[MCPRuntime]:
    async with MCPRuntime(max_concurrency=4, operation_timeout=10.0) as runtime:
        await register_servers(runtime, fake_mcp_servers())
        yield runtime


async def test_registration_discovers_every_server_capability(mcp_runtime):
    discovered = {
        capability.capability_id for capability in await mcp_runtime.registry.list()
    }
    assert {
        "evidence-filesystem:tool:list_allowed_directories",
        "evidence-everything:tool:echo",
        "evidence-browser:tool:browser_navigate",
    } <= discovered


async def test_run_pauses_for_review_and_publishes_exactly_once(memory_jobs, barrier):
    store = MemoryStore()
    script = EvidenceScript(drafts=[synthesis()])
    model = ScriptedChatModel(responder=script)
    saver = InMemorySaver()
    runner = runner_for(store, model, barrier, saver=saver)
    run_id = await admit(store, memory_jobs)

    await work(memory_jobs, runner)

    paused = await store.get_run(run_id)
    assert paused is not None
    assert paused.status == "awaiting_approval"
    assert paused.execution_status == "paused"
    assert paused.approval is not None
    assert paused.approval.tool == "publish_evidence"
    assert paused.approval.policy_rule == "reviewer approves every evidence publication"
    assert paused.approval.expires_at is not None
    assert paused.accuracy.quality == 1
    assert {source.id for source in paused.sources} == {
        "VENDOR-SEC-001",
        "VENDOR-IR-002",
    }
    action = barrier.get_action(paused.approval.action_id)
    assert action.status is RuntimeStatus.PENDING
    assert action.organization_id == TEST_PRINCIPAL.tenant_id
    assert action.requested_by == TEST_PRINCIPAL.subject
    assert store.publications == []
    assert len(model.calls) == 2
    first_call = model.calls[0][0]
    objective = next(m for m in first_call if isinstance(m, HumanMessage))
    assert str(objective.content).startswith("Objective:")
    assert '"covered_topics"' in str(objective.content)
    assert all(
        '"covered_topics"' not in str(m.content)
        for m in first_call
        if isinstance(m, SystemMessage)
    )

    await decide(store, barrier, memory_jobs, run_id)
    await work(memory_jobs, runner_for(store, model, barrier, saver=saver))

    done = await store.get_run(run_id)
    assert done is not None
    assert done.status == "approved"
    assert done.objective_state == "validated"
    assert done.execution_status == "completed"
    assert done.termination == "validated"
    assert done.published_at is not None
    assert store.publications == [run_id]
    assert (
        barrier.get_action(paused.approval.action_id).status is RuntimeStatus.SUCCEEDED
    )
    assert [item.action for item in done.timeline] == ["complete"]
    assert done.timeline[0].objective_complete is True
    events = store.event_types(run_id)
    for expected in (
        "started",
        "iteration",
        "awaiting_approval",
        "resumed",
        "published",
    ):
        assert expected in events
    assert events[-1] == "completed"
    assert len(model.calls) == 3

    replay = await runner_for(store, model, barrier, saver=saver).execute(run_id)
    assert replay == "completed"
    assert store.publications == [run_id]


async def test_invalid_drafts_never_reach_the_reviewer(memory_jobs, barrier):
    store = MemoryStore()
    model = ScriptedChatModel(
        responder=EvidenceScript(drafts=[synthesis(invalid_quote=True), synthesis()])
    )
    runner = runner_for(store, model, barrier)
    run_id = await admit(store, memory_jobs)

    await work(memory_jobs, runner)

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "awaiting_approval"
    assert len(barrier.list_actions()) == 1
    assert [item.action for item in view.timeline] == ["replan"]
    assert view.timeline[0].violations
    gate_reply = next(
        message
        for messages, _ in model.calls
        for message in messages
        if getattr(message, "name", None) == "publish_evidence"
    )
    assert json.loads(str(gate_reply.content))["status"] == "not_submitted"


async def test_reviewer_rejection_ends_the_run_unpublished(memory_jobs, barrier):
    store = MemoryStore()
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    saver = InMemorySaver()
    run_id = await admit(store, memory_jobs)
    await work(memory_jobs, runner_for(store, model, barrier, saver=saver))

    await decide(store, barrier, memory_jobs, run_id, approve=False)
    rejected = await store.get_run(run_id)
    assert rejected is not None
    assert rejected.status == "rejected"
    await work(memory_jobs, runner_for(store, model, barrier, saver=saver))

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "rejected"
    assert view.objective_state == "failed"
    assert view.execution_status == "completed"
    assert view.termination == "policy_violation"
    assert view.synthesis is not None
    assert store.publications == []
    assert store.event_types(run_id)[-1] == "completed"


async def test_one_shot_baseline_ends_unvalidated(memory_jobs, barrier):
    store = MemoryStore()
    model = ScriptedChatModel(
        responder=EvidenceScript(drafts=[synthesis(invalid_quote=True)])
    )
    run_id = await admit(store, memory_jobs, mode="one_shot")

    await work(memory_jobs, runner_for(store, model, barrier))

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "unvalidated"
    assert view.objective_state == "incomplete"
    assert view.termination == "one_shot_complete"
    assert view.accuracy.violations
    assert barrier.list_actions() == ()


async def test_changed_revisions_are_refreshed_before_publication(
    memory_jobs, barrier, tmp_path: Path
):
    revised = [source.model_dump(mode="json") for source in SOURCES]
    revised[0]["revision"] = "2026-10-01"
    revised[0]["body"] += " Keys rotate every 90 days."
    data = tmp_path / "sources.json"
    data.write_text(json.dumps(revised), encoding="utf-8")
    store = MemoryStore()
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    runner = runner_for(
        store, model, barrier, refresh_source=WorkspaceRefreshSource(data)
    )
    run_id = await admit(store, memory_jobs)

    await work(memory_jobs, runner)

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "awaiting_approval"
    assert store.sources[(TEST_PRINCIPAL.tenant_id, "VENDOR-SEC-001")].revision == (
        "2026-10-01"
    )
    assert [item.action for item in view.timeline] == ["refresh"]
    runtime_events = [
        event.get("name")
        for _, event_run, event in store.events
        if event_run == run_id and event["type"] == "runtime"
    ]
    assert "LoopRefreshed" in runtime_events


async def test_crashed_run_continues_from_its_checkpoint(memory_jobs, barrier):
    store = MemoryStore()
    model = blocking(
        EvidenceScript(drafts=[synthesis(invalid_quote=True), synthesis()]), block_at=4
    )
    saver = InMemorySaver()
    run_id = await admit(store, memory_jobs)
    crashed = asyncio.create_task(
        runner_for(store, model, barrier, saver=saver).execute(run_id)
    )
    await asyncio.wait_for(model.entered.wait(), 30)
    crashed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await crashed
    assert len(model.calls) == 3
    model.release.set()

    outcome = await runner_for(store, model, barrier, saver=saver).execute(
        run_id, attempt=1
    )

    assert outcome == "awaiting_approval"
    assert len(model.calls) == 5
    assert "resumed" in store.event_types(run_id)
    view = await store.get_run(run_id)
    assert view is not None
    assert [item.action for item in view.timeline] == ["replan"]


async def test_aborted_job_stops_the_agent_and_cancels_the_run(
    memory_jobs, barrier, settings: Settings
):
    store = MemoryStore()
    model = blocking(EvidenceScript(drafts=[synthesis()]), block_at=1)
    runner = runner_for(store, model, barrier)
    run_id = await admit(store, memory_jobs)
    retry = RetryStrategy(max_attempts=settings.max_attempts)
    context = job_context(memory_jobs, runner, run_id, reason=AbortReason.USER_REQUEST)
    job = asyncio.create_task(execute_job(runner, context, run_id, retry))
    await asyncio.wait_for(model.entered.wait(), 30)

    job.cancel()
    await asyncio.sleep(0.2)
    model.release.set()

    with pytest.raises(JobAborted):
        await job
    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "cancelled"
    assert view.execution_status == "cancelled"
    assert store.event_types(run_id)[-1] == "cancelled"
    assert "ExecutionCancelled" in [
        event.get("name") for _, _, event in store.events if event["type"] == "runtime"
    ]


async def test_shutdown_aborts_are_retried_not_cancelled(
    memory_jobs, barrier, settings
):
    store = MemoryStore()
    model = blocking(EvidenceScript(drafts=[synthesis()]), block_at=1)
    runner = runner_for(store, model, barrier)
    run_id = await admit(store, memory_jobs)
    context = job_context(memory_jobs, runner, run_id, reason=AbortReason.SHUTDOWN)
    job = asyncio.create_task(
        execute_job(runner, context, run_id, RetryStrategy(max_attempts=2))
    )
    await asyncio.wait_for(model.entered.wait(), 30)
    job.cancel()
    await asyncio.sleep(0.2)
    model.release.set()
    with pytest.raises(JobAborted):
        await job
    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "running"


async def test_transient_failures_retry_then_fail_the_run(
    memory_jobs, barrier, settings: Settings, monkeypatch
):
    store = MemoryStore()
    runner = runner_for(
        store,
        ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()])),
        barrier,
    )
    run_id = await admit(store, memory_jobs)

    async def unavailable(*args: Any, **kwargs: Any) -> list[Any]:
        raise ConnectionError("database restarting")

    monkeypatch.setattr(store, "search", unavailable)
    retry = RetryStrategy(max_attempts=settings.max_attempts)
    with pytest.raises(ConnectionError):
        await execute_job(
            runner, job_context(memory_jobs, runner, run_id), run_id, retry
        )
    assert store.event_types(run_id)[-1] == "retrying"
    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "running"

    final = job_context(memory_jobs, runner, run_id, attempts=settings.max_attempts)
    with pytest.raises(ConnectionError):
        await execute_job(runner, final, run_id, retry)
    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "failed"
    assert view.error == "ConnectionError: database restarting"
    assert store.event_types(run_id)[-1] == "failed"


class FlakyPublicationStore(MemoryStore):
    """Fails the first publication, before or after its transaction committed."""

    def __init__(self, *, committed: bool) -> None:
        super().__init__()
        self.committed = committed
        self.publish_calls = 0

    async def publish(self, run_id, synthesis, accuracy):
        self.publish_calls += 1
        if self.publish_calls == 1:
            if self.committed:
                await super().publish(run_id, synthesis, accuracy)
            raise ConnectionError("connection lost during publication")
        return await super().publish(run_id, synthesis, accuracy)


@dataclass
class FlakyPublication:
    store: FlakyPublicationStore
    model: ScriptedChatModel
    saver: InMemorySaver
    run_id: UUID
    action_id: str


def make_retries_due(app: App) -> None:
    for job in app.connector.jobs.values():
        if job["status"] == "todo":
            job["scheduled_at"] = None


async def publication_fails_once(
    memory_jobs, barrier, *, committed: bool
) -> FlakyPublication:
    """Approve a run whose publication raises, then let the worker retry the job."""
    store = FlakyPublicationStore(committed=committed)
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    saver = InMemorySaver()
    run_id = await admit(store, memory_jobs)
    await work(memory_jobs, runner_for(store, model, barrier, saver=saver))
    await decide(store, barrier, memory_jobs, run_id)

    await work(memory_jobs, runner_for(store, model, barrier, saver=saver))
    view = await store.get_run(run_id)
    assert view is not None
    assert view.approval is not None
    action_id = view.approval.action_id
    assert barrier.get_action(action_id).status is RuntimeStatus.UNKNOWN
    assert view.execution_status == "paused"
    assert view.error is not None
    assert "publication outcome is unknown" in view.error
    assert store.event_types(run_id)[-2:] == ["reconciling", "retrying"]
    retrying = [e for e in await store.read_events(run_id) if e["type"] == "retrying"]
    assert retrying[-1]["message"].startswith(PublicationOutcomeUnknown.__name__)

    make_retries_due(memory_jobs)
    await work(memory_jobs, runner_for(store, model, barrier, saver=saver))
    return FlakyPublication(store, model, saver, run_id, action_id)


async def test_failed_publication_returns_to_the_reviewer_and_runs_exactly_once(
    memory_jobs, barrier
):
    flaky = await publication_fails_once(memory_jobs, barrier, committed=False)
    store, run_id = flaky.store, flaky.run_id

    # Nothing was published, so agentbarrier asks for a new approval.
    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "awaiting_approval"
    assert view.approval is not None
    assert view.approval.action_id == flaky.action_id
    assert view.approval.decision is None
    assert barrier.get_action(flaky.action_id).status is RuntimeStatus.PENDING
    assert store.publications == []

    await decide(store, barrier, memory_jobs, run_id)
    await work(memory_jobs, runner_for(store, flaky.model, barrier, saver=flaky.saver))

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "approved"
    assert store.publish_calls == 2
    assert store.publications == [run_id]
    assert barrier.get_action(flaky.action_id).status is RuntimeStatus.SUCCEEDED
    assert store.event_types(run_id).count("awaiting_approval") == 2


async def test_committed_publication_is_replayed_after_reconciliation(
    memory_jobs, barrier
):
    flaky = await publication_fails_once(memory_jobs, barrier, committed=True)
    store, run_id = flaky.store, flaky.run_id

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "approved"
    assert view.execution_status == "completed"
    assert store.publish_calls == 1
    assert store.publications == [run_id]
    action = barrier.get_action(flaky.action_id)
    assert action.status is RuntimeStatus.SUCCEEDED
    assert action.result["published"] is True


async def test_stalled_jobs_are_returned_to_the_queue(memory_jobs, barrier):
    store = MemoryStore()
    run_id = await admit(store, memory_jobs)
    connector = memory_jobs.connector
    job_id = next(iter(connector.jobs))
    connector.jobs[job_id]["status"] = "doing"
    connector.jobs[job_id]["worker_id"] = 404
    context = job_context(
        memory_jobs,
        runner_for(
            store,
            ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()])),
            barrier,
        ),
        run_id,
    )

    recovered = await memory_jobs.tasks[RECOVER_TASK](context, timestamp=0)

    assert recovered == 1
    assert connector.jobs[job_id]["status"] == "todo"


async def test_mcp_observations_and_tool_visibility_follow_the_policy(
    memory_jobs, barrier, mcp_runtime
):
    store = MemoryStore()
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    run_id = await admit(store, memory_jobs, target_url="http://target-site")

    await work(memory_jobs, runner_for(store, model, barrier, mcp=mcp_runtime))

    first_messages, offered = model.calls[0]
    prompt = "\n".join(str(message.content) for message in first_messages)
    assert "Echo: continuous-evidence-ok" in prompt
    assert "Allowed directories" in prompt
    assert "Acme trust page" in prompt
    assert "tool_call_id" not in prompt
    assert {
        "get_validation_feedback",
        "publish_evidence",
        "evidence-filesystem__read_text_file",
        "evidence-browser__browser_navigate",
        "evidence-everything__echo",
    } <= set(offered)
    for hidden in (
        "evidence-filesystem__write_file",
        "evidence-everything__get-env",
        "evidence-browser__browser_click",
    ):
        assert hidden not in offered
    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "awaiting_approval"


async def test_denied_tool_calls_stop_the_run(memory_jobs, barrier, mcp_runtime):
    store = MemoryStore()
    model = ScriptedChatModel(
        responder=EvidenceScript(drafts=[synthesis()], browse="file:///etc/passwd")
    )
    run_id = await admit(store, memory_jobs)

    await work(memory_jobs, runner_for(store, model, barrier, mcp=mcp_runtime))

    view = await store.get_run(run_id)
    assert view is not None
    assert view.status == "failed"
    assert view.termination == "policy_violation"
    assert "browser interactions are denied" in (view.error or "")
    assert barrier.list_actions() == ()


def _messages(model: ScriptedChatModel) -> list[BaseMessage]:
    return [message for messages, _ in model.calls for message in messages]
