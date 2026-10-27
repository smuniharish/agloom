"""Pure, deterministic Loop Engineering transitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agloom.errors import LoopDecisionError, LoopStateError
from agloom.loop.contracts import LoopController, LoopObserver
from agloom.loop.models import (
    LoopAction,
    LoopBudget,
    LoopDecision,
    LoopEvaluation,
    LoopHistoryEntry,
    LoopObservation,
    LoopPhase,
    LoopPolicy,
    LoopState,
)
from agloom.models import TopologyName

_TERMINAL_PHASES = {
    LoopAction.COMPLETE: LoopPhase.COMPLETED,
    LoopAction.STOP: LoopPhase.TERMINATED,
    LoopAction.FAIL: LoopPhase.FAILED,
    LoopAction.WAIT: LoopPhase.WAITING,
    LoopAction.HUMAN_INTERVENTION: LoopPhase.WAITING,
}
PAUSE_ACTIONS = frozenset({LoopAction.WAIT, LoopAction.HUMAN_INTERVENTION})


class IterationFailure(Exception):
    """Describes an iteration failure whose original exception is unavailable.

    Attributes:
        error_type: Name of the original exception type.
    """

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class LoopMachine:
    """Observe, decide, and record transitions for one loop policy."""

    def __init__(
        self, policy: LoopPolicy, observer: LoopObserver, controller: LoopController
    ) -> None:
        self.policy = policy
        self.observer = observer
        self.controller = controller

    def start(self, objective: str, loop_id: str) -> LoopState:
        """Return the initial state of a loop."""
        return LoopState(
            loop_id=loop_id,
            objective=objective,
            budget=LoopBudget.from_policy(self.policy),
        )

    def begin(self, state: LoopState, action: LoopAction) -> LoopState:
        """Account for one new iteration.

        Raises:
            LoopStateError: If the iteration would exceed the iteration budget.
        """
        budget = state.budget.model_copy(
            update={
                "iterations": state.budget.iterations + 1,
                "retries": state.budget.retries + int(action is LoopAction.RETRY),
                "replans": state.budget.replans + int(action is LoopAction.REPLAN),
            }
        )
        if budget.iterations > budget.max_iterations:
            raise LoopStateError(
                "the controller requested an iteration beyond the budget"
            )
        return state.model_copy(
            update={
                "phase": LoopPhase.EXECUTING,
                "iteration": budget.iterations,
                "budget": budget,
            }
        )

    def observe(
        self, state: LoopState, outcome: Mapping[str, Any], error: BaseException | None
    ) -> LoopObservation:
        """Return the observation of the current iteration."""
        if error is None and outcome.get("error_type"):
            error = IterationFailure(
                outcome["error_type"], outcome.get("error_message") or ""
            )
        topology = outcome.get("topology")
        return self.observer.observe(
            iteration=state.iteration,
            result=outcome.get("result"),
            error=error,
            topology=TopologyName(topology) if topology else None,
            duration=float(outcome.get("duration", 0.0)),
            metadata={
                "tool_calls": int(outcome.get("tool_calls", 0)),
                "behavior_signals": list(outcome.get("behavior_signals") or ()),
            },
        )

    def decide(
        self,
        state: LoopState,
        observation: LoopObservation,
        evaluation: LoopEvaluation,
        *,
        evaluation_duration: float = 0.0,
    ) -> tuple[LoopState, LoopDecision]:
        """Ask the controller for a decision and record the iteration.

        Raises:
            LoopDecisionError: If the controller fails or returns no decision.
        """
        budget = state.budget.model_copy(
            update={
                "elapsed": state.budget.elapsed
                + observation.duration
                + evaluation_duration,
                "tool_calls": state.budget.tool_calls + observation.tool_calls,
            }
        )
        try:
            decision = self.controller.decide(
                state, observation, evaluation, self.policy, budget
            )
        except Exception as error:
            raise LoopDecisionError(
                f"the loop controller failed at iteration {state.iteration}"
            ) from error
        if not isinstance(decision, LoopDecision):
            raise LoopDecisionError("the loop controller returned no LoopDecision")
        entry = LoopHistoryEntry(
            observation=observation.model_copy(update={"result": None}),
            evaluation=evaluation,
            decision=decision,
        )
        recorded = state.model_copy(
            update={
                "phase": LoopPhase.EVALUATING,
                "budget": budget,
                "history": (*state.history, entry)[-self.policy.history_limit :],
                "evaluation": evaluation,
                "decision": decision,
                "objective_status": evaluation.objective_status,
                "progress_status": evaluation.progress_status,
                "progress": (
                    evaluation.objective_progress
                    if evaluation.objective_progress is not None
                    else state.progress
                ),
                "active_topology": observation.topology or state.active_topology,
            }
        )
        return recorded, decision

    @staticmethod
    def terminate(state: LoopState, decision: LoopDecision) -> LoopState:
        """Return the terminal state for a terminal decision."""
        return state.model_copy(
            update={
                "phase": _TERMINAL_PHASES.get(decision.action, LoopPhase.TERMINATED),
                "termination_reason": decision.termination_reason,
            }
        )


def is_terminal(decision: LoopDecision) -> bool:
    """Return whether a decision ends iteration (pauses included)."""
    return decision.action in _TERMINAL_PHASES
