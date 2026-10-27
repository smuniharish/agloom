"""One attempt at executing a run inside a worker job."""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from agloom import Agent, AgentEvent, CancellationError, LoopResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from .agent import RunScope, build_agent
from .auth import Principal, principal_scope
from .governance import (
    POLICY_NAMESPACE,
    approvals_from_interrupts,
    reconciliations_from_interrupts,
    unsupported_pauses,
)
from .models import TERMINAL_RUN_STATUSES, Evidence, LoopRun, RunProof
from .observability import APPLICATION_ID
from .safety import RunWorkspace
from .validation import ObjectiveValidator, release_is_current

if TYPE_CHECKING:
    from .services import StudioServices

_LOGGER = logging.getLogger(__name__)
_RESERVED_EVENT_KEYS = frozenset({"id", "type", "timestamp", "name", "topology"})
StopReason = Literal["cancelled", "interrupted"]


class UsageCounter(BaseCallbackHandler):
    """Counts the model calls, tool calls, and execution time of one job attempt.

    The run proof adds them up over attempts, so a run that paused for approval
    reports the work done before and after the pause. (A loop budget resumed
    after a pause counts only the part of the iteration that ran after it.)
    """

    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self.model_calls = 0
        self.tool_calls = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.model_calls += 1

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.tool_calls += 1

    def usage(self, previous: RunProof) -> dict[str, Any]:
        """The run's usage including this attempt."""
        return {
            "model_calls": previous.model_calls + self.model_calls,
            "tool_calls": previous.tool_calls + self.tool_calls,
            "elapsed_seconds": previous.elapsed_seconds
            + time.monotonic()
            - self.started,
        }


class RunExecution:
    """Executes a run for its owner, or continues it from its checkpoint.

    A run without a checkpoint starts with its objective. A run with one (a
    crashed attempt, an interrupted worker, or a decided approval) continues
    from the last checkpoint: completed loop iterations and tool calls are not
    repeated, and an approved release executes exactly once.
    """

    def __init__(
        self, services: StudioServices, run_id: UUID, principal: Principal
    ) -> None:
        self.services = services
        self.run_id = run_id
        self.principal = principal
        self.agent: Agent | None = None
        self._stop: StopReason | None = None

    def stop(self, reason: StopReason) -> None:
        """Ask the agent to stop at its next model or tool boundary."""
        self._stop = reason
        if self.agent is not None:
            self.agent.stop()

    async def run(self) -> str:
        with principal_scope(self.principal):
            return await self._run()

    async def _run(self) -> str:
        services = self.services
        store = services.store
        run = store.mark_running(self.run_id)
        if run is None:
            return "skipped"
        task = store.get_task(run.task_id)
        project = store.get_project(task.project_id) if task is not None else None
        if task is None or project is None:
            await self._finish(run, "failed", RunProof(), {"error": "task not found"})
            return "failed"
        if services.checkpointer is None or services.model_factory is None:
            raise RuntimeError("this process is not configured to execute runs")
        scope = RunScope(run, task, project, self.principal)
        run_workspace = RunWorkspace.open(
            services.settings.projects_root,
            project.id,
            run.id,
            services.execution_boundary,
        )
        counter = UsageCounter()
        agent = build_agent(
            scope=scope,
            run_workspace=run_workspace,
            model=services.model_factory(),
            store=store,
            publish_event=self._publish,
            policy=services.policy,
            policy_store=services.policy_store,
            checkpointer=services.checkpointer,
            mcp=services.mcp,
            preview_url=services.settings.preview_url,
            observers=(*services.observers, self._runtime_event),
            on_iteration=lambda payload: self._publish(run.id, "iteration", **payload),
        )
        self.agent = agent
        if self._stop is not None:
            agent.stop()
        config: dict[str, Any] = {
            "configurable": {"thread_id": str(run.id)},
            "callbacks": [counter],
            "metadata": {
                "application_id": APPLICATION_ID,
                "tenant_id": self.principal.tenant_id,
                "run_id": str(run.id),
                "policy_namespace": POLICY_NAMESPACE,
            },
        }
        snapshot = await agent.aget_state(config)
        resuming = snapshot.created_at is not None
        self._publish(
            run.id,
            "resumed" if resuming else "running",
            message=(
                "Continuing from the latest checkpoint."
                if resuming
                else "The bounded build started."
            ),
        )
        try:
            outcome = await agent.ainvoke_loop(
                None if resuming else scope.objective, config
            )
        except CancellationError:
            return await self._stopped(run, run_workspace, counter)
        return await self._settle(scope, run_workspace, outcome, counter)

    async def _settle(
        self,
        scope: RunScope,
        run_workspace: RunWorkspace,
        outcome: LoopResult,
        counter: UsageCounter,
    ) -> str:
        run = scope.run
        store = self.services.store
        state = outcome.state
        criteria = scope.project.acceptance_criteria
        total = len(criteria)
        if outcome.paused:
            # Evidence is recorded when an iteration ends; the release gate has
            # just validated the paused iteration, so its check results are cached.
            current = await asyncio.to_thread(
                ObjectiveValidator(run_workspace.workspace).validate, criteria
            )
            passed = sum(result.passed for result in current)
        else:
            latest = _latest(store.evidence_for_run(run.id))
            passed = sum(item.passed for item in latest)
        proof = RunProof(
            iterations=state.budget.iterations,
            acceptance_criteria=total,
            passed_criteria=passed,
            validation_pass_rate=passed / total if total else 0,
            termination_reason=(
                state.termination_reason.value if state.termination_reason else None
            ),
            **counter.usage(run.proof),
        )
        if outcome.paused:
            return await self._pause(run, run_workspace, outcome, proof)
        workspace = run_workspace.workspace
        released = release_is_current(workspace)
        status = (
            "completed"
            if total and passed == total and released and outcome.error is None
            else "failed"
        )
        result: dict[str, Any] = {
            "agent_result": _text(outcome.result)[:20_000],
            "loop_id": state.loop_id,
            "release": workspace.read_release_marker() if released else None,
        }
        if outcome.error is not None:
            result["error"] = f"{type(outcome.error).__name__}: {outcome.error}"
        await self._finish(run, status, proof, result, run_workspace)
        return status

    async def _pause(
        self,
        run: LoopRun,
        run_workspace: RunWorkspace,
        outcome: LoopResult,
        proof: RunProof,
    ) -> str:
        """Park the run until people approve or reconcile its consequential calls.

        A call whose tool raised has an unknown outcome; it must be reconciled
        before anything else, so reconciliation takes precedence over approvals.
        """
        approvals = approvals_from_interrupts(outcome.interrupts)
        reconciliations = reconciliations_from_interrupts(outcome.interrupts)
        unsupported = unsupported_pauses(outcome.interrupts)
        if unsupported or not (approvals or reconciliations):
            await self._finish(
                run,
                "failed",
                proof.model_copy(update={"termination_reason": "unsupported_pause"}),
                {"error": f"the run paused for an unsupported reason: {unsupported}"},
                run_workspace,
            )
            return "failed"
        status = "awaiting_reconciliation" if reconciliations else "awaiting_approval"
        items = [*reconciliations, *approvals]
        paused = self.services.store.record_pause(run.id, status, items, proof)
        message = (
            "A release failed after it started, so its outcome is unknown; an "
            "operator must reconcile it."
            if reconciliations
            else "The release is waiting for a reviewer."
        )
        self._publish(
            run.id,
            status,
            message=message,
            approvals=[item.model_dump(mode="json") for item in items],
            run=paused.model_dump(mode="json") if paused else None,
        )
        return status

    async def _stopped(
        self, run: LoopRun, run_workspace: RunWorkspace, counter: UsageCounter
    ) -> str:
        if self._stop == "cancelled":
            proof = run.proof.model_copy(
                update={
                    **counter.usage(run.proof),
                    "termination_reason": "cancelled",
                }
            )
            await self._finish(
                run, "cancelled", proof, {"error": "run cancelled"}, run_workspace
            )
            return "cancelled"
        self._publish(
            run.id,
            "interrupted",
            message="The worker stopped; the run continues from its checkpoint.",
        )
        return "interrupted"

    async def _finish(
        self,
        run: LoopRun,
        status: str,
        proof: RunProof,
        result: dict[str, Any],
        run_workspace: RunWorkspace | None = None,
    ) -> None:
        finished = self.services.store.finish_run(run.id, status, proof, result)
        self._publish(
            run.id,
            status,
            message=_terminal_message(status, result),
            run=finished.model_dump(mode="json") if finished else None,
        )
        await self._discard(run.id, run_workspace)

    async def record_failure(
        self, error: BaseException, *, retrying: bool, attempt: int
    ) -> None:
        """Report an attempt that failed outside the loop (for example a lost database)."""
        detail = f"{type(error).__name__}: {error}"
        with principal_scope(self.principal):
            if retrying:
                self._publish(
                    self.run_id, "retrying", attempt=attempt + 1, message=detail
                )
                return
            run = self.services.store.get_run(self.run_id)
            if run is not None and run.status not in TERMINAL_RUN_STATUSES:
                await self._finish(
                    run,
                    "failed",
                    run.proof.model_copy(
                        update={"termination_reason": "execution_failed"}
                    ),
                    {"error": detail},
                )

    async def _discard(
        self, run_id: UUID, run_workspace: RunWorkspace | None = None
    ) -> None:
        if self.services.checkpointer is not None:
            await self.services.checkpointer.adelete_thread(str(run_id))
        if run_workspace is not None:
            run_workspace.discard()

    def _publish(self, run_id: UUID, event_type: str, **payload: Any) -> None:
        with principal_scope(self.principal):
            self.services.events.publish(run_id, event_type, **payload)

    def _runtime_event(self, event: AgentEvent) -> None:
        details = {
            key: value
            for key, value in event.payload.items()
            if key not in _RESERVED_EVENT_KEYS
        }
        self._publish(
            self.run_id,
            "runtime",
            name=event.name,
            topology=event.topology,
            **details,
        )


async def discard_run_state(
    services: StudioServices, run_id: UUID, principal: Principal
) -> None:
    """Delete the checkpoint and private workspace of a terminal run."""
    with principal_scope(principal):
        run = services.store.get_run(run_id)
        if run is None or run.status not in TERMINAL_RUN_STATUSES:
            return
        task = services.store.get_task(run.task_id)
    if services.checkpointer is not None:
        await services.checkpointer.adelete_thread(str(run_id))
    if task is not None:
        shutil.rmtree(
            services.settings.projects_root.resolve()
            / ".runs"
            / str(task.project_id)
            / str(run_id),
            ignore_errors=True,
        )


def _latest(evidence: list[Evidence]) -> list[Evidence]:
    if not evidence:
        return []
    iteration = max(item.iteration for item in evidence)
    return [item for item in evidence if item.iteration == iteration]


def _text(result: Any) -> str:
    if isinstance(result, BaseMessage):
        return result.text
    return "" if result is None else str(result)


def _terminal_message(status: str, result: dict[str, Any]) -> str:
    if status == "completed":
        return "Objective validated and the release is published."
    if status == "cancelled":
        return "Run cancelled."
    if status == "rejected":
        return "The reviewer rejected the release."
    return str(result.get("error") or "The loop budget ended before the objective.")
