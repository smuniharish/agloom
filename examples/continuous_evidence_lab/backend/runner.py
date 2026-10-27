"""Executes and resumes evidence runs; one procrastinate job drives one step.

A run starts fresh, resumes after a reviewer decision, or continues from its
LangGraph checkpoint after a crash. Pauses for approval end the job with the
run awaiting approval; the reviewer's decision enqueues the resume job.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from agentbarrier.errors import ApprovalExpired, ApprovalRejected, RuntimeBarrierError
from agentbarrier.runtime import (
    RuntimePolicy,
    RuntimeReconciliation,
    RuntimeStatus,
    RuntimeStore,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from mcp_capability_router import MCPRuntime
from xstructured import XStructuredError

from agloom import (
    Agent,
    AgentEvent,
    AgloomError,
    CancellationError,
    CapabilityResolutionError,
    EventName,
    ExecutionPaused,
    PolicyViolationError,
)

from .agent import WorkspaceRefreshSource, make_agent, run_config
from .auth import current_principal
from .mcp import BROWSER_SERVER, EVERYTHING_SERVER, FILESYSTEM_SERVER, capability
from .models import (
    TERMINAL_EXECUTION,
    Approval,
    RunRequest,
    Source,
    Synthesis,
)
from .observability import QUALITY, RUNS, Telemetry
from .policy import (
    NAMESPACE,
    PUBLISH_TOOL,
    approval_from_request,
    approval_request,
    publication_key,
    reconciliation_request,
    run_policy,
)
from .store import Repository
from .validation import EvidenceLoopObserver, extract_synthesis, score_synthesis

_LOGGER = logging.getLogger(__name__)
_LOOP_END = frozenset({EventName.LOOP_COMPLETED, EventName.LOOP_TERMINATED})
LEDGER_RECONCILER = f"{NAMESPACE}:evidence-ledger"
_RECONCILED = {
    "committed": "The evidence ledger records the publication; resuming with it.",
    "not committed": "The publication did not commit; it needs approval again.",
    "already reconciled": "The publication was reconciled; resuming.",
}


class PublicationOutcomeUnknown(RuntimeError):
    """The publication raised; the retried job reconciles it against the ledger."""


class RunRecorder:
    """Persists one run's events and timeline in order without blocking the agent.

    Agloom delivers observer events synchronously, possibly from worker
    threads; the recorder hands them to one writer task on the job's loop.
    """

    def __init__(self, store: Repository, run_id: UUID) -> None:
        self.store = store
        self.run_id = run_id
        self.observer: EvidenceLoopObserver | None = None
        self.termination: str | None = None
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        self._writer = asyncio.create_task(self._drain())

    def publish(self, event_type: str, payload: Mapping[str, Any]) -> None:
        item = (event_type, dict(payload))
        if threading.get_ident() == self._loop_thread:
            self._queue.put_nowait(item)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def on_agent_event(self, event: AgentEvent) -> None:
        payload = dict(event.payload)
        if event.name == EventName.LOOP_ITERATION_COMPLETED:
            self.publish("timeline", payload)
        elif event.name in _LOOP_END and payload.get("termination_reason"):
            self.termination = str(payload["termination_reason"])
        self.publish(
            "runtime", {"name": str(event.name), "topology": event.topology, **payload}
        )

    async def flush(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        self._queue.put_nowait(None)
        await self._writer

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                event_type, payload = item
                if event_type == "timeline":
                    await self._save_iteration(payload)
                else:
                    await self.store.append_event(self.run_id, event_type, payload)
            except Exception:
                _LOGGER.exception("failed to record a run event")
            finally:
                self._queue.task_done()

    async def _save_iteration(self, payload: Mapping[str, Any]) -> None:
        if self.observer is None:
            return
        entry = self.observer.timeline_entry(
            int(payload["iteration"]),
            str(payload.get("action", "")),
            str(payload.get("reason", "")),
        )
        await self.store.save_iteration(
            self.run_id,
            entry,
            {
                key: payload.get(key)
                for key in ("action", "reason", "progress_status", "objective_status")
            },
        )


@dataclass
class RunServices:
    """Long-lived dependencies shared by every run of one worker process."""

    store: Repository
    checkpointer: BaseCheckpointSaver[Any]
    barrier: RuntimeStore
    policy: RuntimePolicy
    telemetry: Telemetry
    model_factory: Callable[[], BaseChatModel]
    mcp: MCPRuntime | None = None
    refresh_source: WorkspaceRefreshSource | None = None


class EvidenceRunner:
    """Executes one step of a run: start, resume, or recover from a crash."""

    def __init__(self, services: RunServices) -> None:
        self.services = services
        self._agents: dict[UUID, Agent] = {}

    def stop(self, run_id: UUID) -> bool:
        """Cooperatively cancel the agent executing a run in this process."""
        agent = self._agents.get(run_id)
        if agent is None:
            return False
        agent.stop()
        return True

    async def execute(self, run_id: UUID, *, attempt: int = 0) -> str:
        store = self.services.store
        state = await store.run_state(run_id)
        if state is None:
            return "missing"
        if state.execution_status in TERMINAL_EXECUTION:
            return state.execution_status
        request = await store.get_request(run_id)
        if request is None:
            return "missing"
        recorder = RunRecorder(store, run_id)
        try:
            return await self._execute(run_id, request, recorder, attempt)
        finally:
            self._agents.pop(run_id, None)
            await recorder.aclose()

    async def _execute(
        self, run_id: UUID, request: RunRequest, recorder: RunRecorder, attempt: int
    ) -> str:
        services = self.services
        store = services.store
        principal = current_principal()
        sources = await store.search(f"{request.objective} {' '.join(request.topics)}")
        observer = EvidenceLoopObserver(
            sources,
            request.topics,
            revision_provider=services.refresh_source or WorkspaceRefreshSource(),
            events=recorder.publish,
        )
        recorder.observer = observer
        observer.restore(
            await store.latest_iteration(run_id),
            published=await store.published_synthesis(run_id),
        )
        agent = make_agent(
            model=services.model_factory(),
            observer=observer,
            policy=run_policy(
                policy=services.policy,
                store=services.barrier,
                tenant_id=principal.tenant_id,
                run_id=run_id,
                requested_by=principal.subject,
            ),
            store=store,
            run_id=run_id,
            tenant_id=principal.tenant_id,
            publish=self._publisher(run_id, observer, recorder),
            mode=request.mode,
            mcp=services.mcp,
            checkpointer=services.checkpointer,
            refresh_source=services.refresh_source,
            observers=(*services.telemetry.observers, recorder.on_agent_event),
        )
        self._agents[run_id] = agent
        config = run_config(run_id, mode=request.mode)
        try:
            answer = await self._advance(
                agent, config, run_id, request, observer, recorder, attempt
            )
        except ExecutionPaused as paused:
            reconciliation = reconciliation_request(paused.interrupts)
            if reconciliation is not None:
                error = reconciliation.get("error") or "an error"
                await self.services.store.await_reconciliation(
                    run_id,
                    f"The publication outcome is unknown after {error}; the "
                    "retried job reconciles it against the evidence ledger.",
                )
                recorder.publish(
                    "reconciling",
                    {
                        "message": "The publication failed; the retried job "
                        "reconciles its outcome against the evidence ledger.",
                        "action_id": str(reconciliation["action_id"]),
                        "error": reconciliation.get("error"),
                    },
                )
                raise PublicationOutcomeUnknown(
                    f"publication raised {error}"
                ) from paused
            await self._await_approval(run_id, paused.interrupts, observer, recorder)
            return "awaiting_approval"
        except CancellationError:
            raise
        except ApprovalRejected:
            return await self._rejected(
                run_id, recorder, "Reviewer rejected publication."
            )
        except ApprovalExpired:
            return await self._rejected(
                run_id,
                recorder,
                "The approval window closed before a reviewer decided.",
                termination="approval_expired",
            )
        except PolicyViolationError as denied:
            return await self._failed(
                run_id, recorder, str(denied), termination="policy_violation"
            )
        except (AgloomError, RuntimeBarrierError) as error:
            return await self._failed(
                run_id, recorder, f"{type(error).__name__}: {error}"
            )
        finally:
            agent.flush()
        if answer is None:
            return "awaiting_approval"
        return await self._complete(run_id, answer, observer, recorder)

    async def _advance(
        self,
        agent: Agent,
        config: dict[str, Any],
        run_id: UUID,
        request: RunRequest,
        observer: EvidenceLoopObserver,
        recorder: RunRecorder,
        attempt: int,
    ) -> Any:
        store = self.services.store
        snapshot = await agent.aget_state(config)
        if snapshot.interrupts:
            reconciliation = reconciliation_request(snapshot.interrupts)
            if reconciliation is not None:
                outcome = await self._reconcile_publication(run_id, reconciliation)
                observer.approval_decided = True
                await store.mark_running(run_id)
                recorder.publish(
                    "resumed",
                    {"message": _RECONCILED[outcome], "attempt": attempt},
                )
                return await agent.aresume(config)
            pending = approval_request(snapshot.interrupts)
            if pending is None:
                raise RuntimeError("the run paused at an interrupt it cannot resume")
            action = await asyncio.to_thread(
                self.services.barrier.get_action, str(pending["action_id"])
            )
            if action.status is RuntimeStatus.PENDING:
                await self._await_approval(
                    run_id, snapshot.interrupts, observer, recorder
                )
                return None
            observer.approval_decided = True
            await store.mark_running(run_id)
            recorder.publish(
                "resumed",
                {
                    "message": "Resuming after the reviewer decision.",
                    "attempt": attempt,
                },
            )
            return await agent.aresume(config)
        if snapshot.next:
            await store.mark_running(run_id)
            recorder.publish(
                "resumed",
                {"message": "Continuing from the last checkpoint.", "attempt": attempt},
            )
            return await agent.aresume(config)
        if snapshot.values:
            messages = snapshot.values.get("messages") or []
            return next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, AIMessage)
                ),
                None,
            ) or AIMessage(content="")
        await store.mark_running(run_id)
        recorder.publish(
            "started", {"message": "Evidence retrieval finished; the loop is starting."}
        )
        observations = await collect_observations(agent, request)
        structured = agent.with_structured_output(Synthesis)
        try:
            result = await structured.ainvoke(
                build_prompt(request, observer.sources, observations), config
            )
        except XStructuredError:
            return AIMessage(content="")
        return result.structured

    def _publisher(
        self, run_id: UUID, observer: EvidenceLoopObserver, recorder: RunRecorder
    ) -> Callable[[Synthesis], Any]:
        store = self.services.store

        async def publish(synthesis: Synthesis) -> dict[str, Any]:
            accuracy = score_synthesis(synthesis, observer.sources, observer.topics)
            published_at = await store.publish(run_id, synthesis, accuracy)
            observer.mark_published(synthesis)
            recorder.publish(
                "published",
                {
                    "message": "Reviewer-approved evidence was published.",
                    "quality": accuracy.quality,
                    "published_at": published_at.isoformat(),
                },
            )
            return {
                "published": True,
                "run_id": str(run_id),
                "published_at": published_at.isoformat(),
                "quality": accuracy.quality,
            }

        return publish

    async def _reconcile_publication(
        self, run_id: UUID, request: Mapping[str, Any]
    ) -> str:
        """Settle an unknown publication outcome from the evidence ledger.

        Publication is one transaction in the application database, so the
        ledger shows whether it committed. A committed publication is replayed
        to the model; one that did not commit returns to the reviewer, whose new
        approval lets it run exactly once. An action an operator already
        reconciled is left as it is.
        """
        barrier = self.services.barrier
        action = await asyncio.to_thread(barrier.get_action, str(request["action_id"]))
        tenant_id = current_principal().tenant_id
        if (
            action.tool_name != PUBLISH_TOOL
            or action.namespace != NAMESPACE
            or action.organization_id != tenant_id
            or action.idempotency_key != publication_key(tenant_id, run_id)
        ):
            raise RuntimeError("the paused action does not belong to this run")
        if action.status is not RuntimeStatus.UNKNOWN:
            return "already reconciled"
        view = await self.services.store.get_run(run_id)
        if view is not None and view.published_at is not None:
            await asyncio.to_thread(
                barrier.reconcile,
                action.action_id,
                RuntimeReconciliation.COMMITTED,
                resolved_by=LEDGER_RECONCILER,
                reason="the evidence ledger records this publication",
                result={
                    "published": True,
                    "run_id": str(run_id),
                    "published_at": view.published_at.isoformat(),
                    "quality": view.accuracy.quality,
                },
            )
            return "committed"
        await asyncio.to_thread(
            barrier.reconcile,
            action.action_id,
            RuntimeReconciliation.NOT_COMMITTED,
            resolved_by=LEDGER_RECONCILER,
            reason="the evidence ledger has no publication for this run",
        )
        return "not committed"

    async def _await_approval(
        self,
        run_id: UUID,
        interrupts: Any,
        observer: EvidenceLoopObserver,
        recorder: RunRecorder,
    ) -> None:
        pending = approval_request(interrupts)
        if pending is None:
            raise RuntimeError("the run paused at an interrupt it cannot resume")
        synthesis = Synthesis.model_validate(pending.get("arguments") or {})
        accuracy = score_synthesis(synthesis, observer.sources, observer.topics)
        approval: Approval = approval_from_request(pending)
        store = self.services.store
        await store.await_approval(run_id, synthesis, accuracy, approval)
        await recorder.flush()
        view = await store.get_run(run_id)
        RUNS.labels("awaiting_approval").inc()
        QUALITY.set(accuracy.quality)
        recorder.publish(
            "awaiting_approval",
            {
                "message": "Validated evidence is waiting for a reviewer decision.",
                "run": view.model_dump(mode="json") if view else None,
            },
        )

    async def _complete(
        self,
        run_id: UUID,
        answer: Any,
        observer: EvidenceLoopObserver,
        recorder: RunRecorder,
    ) -> str:
        store = self.services.store
        synthesis = (
            observer.published or extract_synthesis(answer) or observer.latest_synthesis
        )
        accuracy = (
            score_synthesis(synthesis, observer.sources, observer.topics)
            if synthesis is not None
            else observer.latest_accuracy
        )
        await recorder.flush()
        await store.complete_run(
            run_id,
            termination=recorder.termination,
            synthesis=synthesis,
            accuracy=accuracy,
        )
        view = await store.get_run(run_id)
        status = view.status if view else "unvalidated"
        RUNS.labels(status).inc()
        recorder.publish(
            "completed",
            {
                "message": "Reviewer-approved evidence was published."
                if status == "approved"
                else "The loop ended without approval-ready evidence.",
                "run": view.model_dump(mode="json") if view else None,
            },
        )
        return status

    async def _rejected(
        self,
        run_id: UUID,
        recorder: RunRecorder,
        message: str,
        *,
        termination: str | None = None,
    ) -> str:
        await recorder.flush()
        store = self.services.store
        await store.reject_run(
            run_id,
            termination=termination or recorder.termination or "policy_violation",
            error=None if termination is None else message,
        )
        view = await store.get_run(run_id)
        RUNS.labels("rejected").inc()
        recorder.publish(
            "completed",
            {"message": message, "run": view.model_dump(mode="json") if view else None},
        )
        return "rejected"

    async def _failed(
        self,
        run_id: UUID,
        recorder: RunRecorder,
        detail: str,
        *,
        termination: str = "execution_failed",
    ) -> str:
        await recorder.flush()
        store = self.services.store
        await store.fail_run(run_id, detail, termination=termination)
        view = await store.get_run(run_id)
        RUNS.labels("failed").inc()
        recorder.publish(
            "failed",
            {"message": detail, "run": view.model_dump(mode="json") if view else None},
        )
        return "failed"

    async def record_transient_failure(
        self, run_id: UUID, error: BaseException, *, final: bool, attempt: int
    ) -> None:
        """Record a job failure that procrastinate retries unless it was the last."""
        store = self.services.store
        detail = f"{type(error).__name__}: {error}"
        if final:
            await store.fail_run(run_id, detail)
            view = await store.get_run(run_id)
            RUNS.labels("failed").inc()
            await store.append_event(
                run_id,
                "failed",
                {
                    "message": detail,
                    "run": view.model_dump(mode="json") if view else None,
                },
            )
            return
        await store.append_event(
            run_id, "retrying", {"message": detail, "attempt": attempt + 1}
        )

    async def record_cancellation(self, run_id: UUID) -> None:
        store = self.services.store
        await store.cancel_run(run_id, "cancelled by the user")
        view = await store.get_run(run_id)
        RUNS.labels("cancelled").inc()
        await store.append_event(
            run_id,
            "cancelled",
            {
                "message": "The run was cancelled.",
                "run": view.model_dump(mode="json") if view else None,
            },
        )


async def collect_observations(agent: Agent, request: RunRequest) -> dict[str, str]:
    """Collect MCP observations under the run's policy before the loop starts."""
    output: dict[str, str] = {}
    for surface, name, arguments in (
        ("filesystem", capability(FILESYSTEM_SERVER, "list_allowed_directories"), {}),
        (
            "everything",
            capability(EVERYTHING_SERVER, "echo"),
            {"message": "continuous-evidence-ok"},
        ),
    ):
        with contextlib.suppress(CapabilityResolutionError):
            result = await agent.aexecute_capability(name, arguments)
            output[surface] = tool_text(result)[:800]
    if request.target_url is not None:
        try:
            navigation = await agent.aexecute_capability(
                capability(BROWSER_SERVER, "browser_navigate"),
                {"url": str(request.target_url)},
            )
            page = await agent.aexecute_capability(
                capability(BROWSER_SERVER, "browser_snapshot"), {}
            )
        except CapabilityResolutionError as error:
            raise CapabilityResolutionError(
                "the isolated browser is not configured for live targets"
            ) from error
        output["playwright"] = f"{tool_text(navigation)}\n{tool_text(page)}"[:6000]
    return output


def tool_text(result: Any) -> str:
    """Return the text of an MCP tool result, a LangChain `ToolMessage`."""
    content = result.content if isinstance(result, BaseMessage) else result
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) if isinstance(block, Mapping) else str(block)
            for block in content
        )
    return str(content)


def build_prompt(
    request: RunRequest, sources: list[Source], observations: Mapping[str, str]
) -> str:
    serialized = json.dumps([source.model_dump(mode="json") for source in sources])
    return (
        f"Objective: {request.objective}\nRequired topics: {request.topics}\n"
        f"Retrieved sources: {serialized}\n"
        f"MCP observations: {json.dumps(dict(observations))}\n"
        "Before answering, call get_validation_feedback. On later iterations its "
        "gaps, violations, and source revisions are authoritative."
    )
