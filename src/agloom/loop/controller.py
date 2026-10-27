"""Budget-first, policy-bounded loop decisions."""

from __future__ import annotations

from agloom.loop.models import (
    LoopAction,
    LoopBudget,
    LoopDecision,
    LoopEvaluation,
    LoopObservation,
    LoopPolicy,
    LoopState,
    LoopTerminationReason,
)

_AWAITING_APPROVAL = frozenset({"ApprovalRequired"})
_POLICY_STOPS = frozenset(
    {
        "PolicyDenied",
        "PolicyViolationError",
        "ApprovalRejected",
        "ApprovalExpired",
        "EmergencyPauseActive",
        "ActionLimitExceeded",
        "ActionLimitValueError",
    }
)
_UNRECOVERABLE = frozenset(
    {
        "ActionOutcomeUnknown",
        "ActionInProgress",
        "ActionBindingError",
        "RuntimeStoreError",
        "FrameworkControlSignalError",
        "PolicyConfigurationError",
        "ConfigurationError",
    }
)
_BUDGET_ERRORS = frozenset({"LoopBudgetExceededError"})
_BEHAVIOR_PRIORITY = ("stop", "pause", "human_review", "escalate", "redirect", "retry")


class DefaultLoopController:
    """Decides the next action: completion first, then budgets, signals, and progress.

    An achieved objective (or convergence) completes the loop. Otherwise the
    duration and tool-call budgets are checked before any decision that would
    continue iterating. Actions that change the safety posture (human
    intervention, refresh, and topology change) stop the loop when the policy
    does not permit them. Quality hints (retry and replan) that the policy does
    not permit fall back to continuing within the remaining budget.
    """

    def decide(
        self,
        state: LoopState,
        observation: LoopObservation,
        evaluation: LoopEvaluation,
        policy: LoopPolicy,
        budget: LoopBudget,
    ) -> LoopDecision:
        """Return the decision for one evaluated iteration."""
        if not observation.success:
            return _failure(observation, policy, budget)
        if evaluation.objective_complete:
            return LoopDecision(
                action=LoopAction.COMPLETE,
                reason=evaluation.reason,
                termination_reason=(
                    LoopTerminationReason.VALIDATED
                    if observation.validation_passed
                    else LoopTerminationReason.OBJECTIVE_SATISFIED
                ),
            )
        if evaluation.convergence and policy.complete_on_convergence:
            return LoopDecision(
                action=LoopAction.COMPLETE,
                reason=evaluation.reason,
                termination_reason=LoopTerminationReason.CONVERGED,
            )
        if budget.elapsed >= budget.max_duration:
            return _stop(
                "the duration budget is exhausted", LoopTerminationReason.TIMEOUT
            )
        if budget.tool_calls > budget.max_tool_calls:
            return _stop(
                "the tool-call budget is exhausted",
                LoopTerminationReason.TOOL_CALL_LIMIT,
            )
        behavior = _behavior(observation, policy, budget)
        if behavior is not None:
            return behavior
        recommended = _recommended(observation, evaluation, policy, budget)
        if recommended is not None:
            return recommended
        if policy.max_iterations == 1:
            return LoopDecision(
                action=LoopAction.COMPLETE,
                reason="the one-shot execution succeeded",
                termination_reason=LoopTerminationReason.ONE_SHOT_COMPLETE,
            )
        if evaluation.stagnation:
            return _stagnation(policy, budget)
        if _oscillates(state, policy):
            return _stop(
                "alternating actions or topologies made no progress",
                LoopTerminationReason.OSCILLATION,
            )
        if budget.iterations >= budget.max_iterations:
            return _stop(
                "the iteration budget is exhausted",
                LoopTerminationReason.ITERATION_LIMIT,
            )
        return LoopDecision(action=LoopAction.CONTINUE, reason=evaluation.reason)


def _failure(
    observation: LoopObservation, policy: LoopPolicy, budget: LoopBudget
) -> LoopDecision:
    error = observation.error_type or "Exception"
    if error in _BUDGET_ERRORS:
        return _stop(
            "the tool-call budget is exhausted", LoopTerminationReason.TOOL_CALL_LIMIT
        )
    if error in _AWAITING_APPROVAL:
        return LoopDecision(
            action=LoopAction.HUMAN_INTERVENTION,
            reason="a tool call is waiting for policy approval",
            termination_reason=LoopTerminationReason.HUMAN_INTERVENTION,
        )
    if error in _POLICY_STOPS:
        return _stop(
            f"the policy blocked a tool call ({error})",
            LoopTerminationReason.POLICY_VIOLATION,
        )
    if error in _UNRECOVERABLE:
        return _fail(
            f"the iteration failed with {error}", LoopTerminationReason.EXECUTION_FAILED
        )
    if budget.elapsed >= budget.max_duration:
        return _fail("the duration budget is exhausted", LoopTerminationReason.TIMEOUT)
    if policy.allow_retry and budget.retries < budget.max_retries:
        if budget.iterations >= budget.max_iterations:
            return _fail(
                "the iteration budget is exhausted",
                LoopTerminationReason.ITERATION_LIMIT,
            )
        return LoopDecision(action=LoopAction.RETRY, reason=f"retrying after {error}")
    if policy.allow_retry:
        return _fail("the retry budget is exhausted", LoopTerminationReason.RETRY_LIMIT)
    return _fail(
        f"the iteration failed with {error}", LoopTerminationReason.EXECUTION_FAILED
    )


def _behavior(
    observation: LoopObservation, policy: LoopPolicy, budget: LoopBudget
) -> LoopDecision | None:
    signals = set(observation.behavior_signals)
    signal = next((kind for kind in _BEHAVIOR_PRIORITY if kind in signals), None)
    if signal is None:
        return None
    reason = f"behavior intervention: {signal}"
    if signal == "stop":
        return _stop(reason, LoopTerminationReason.BEHAVIOR_INTERVENTION)
    if signal in {"pause", "human_review", "escalate"}:
        if not policy.allow_human_intervention:
            return _stop(reason, LoopTerminationReason.BEHAVIOR_INTERVENTION)
        return LoopDecision(
            action=LoopAction.WAIT
            if signal == "pause"
            else LoopAction.HUMAN_INTERVENTION,
            reason=reason,
            termination_reason=LoopTerminationReason.HUMAN_INTERVENTION,
        )
    action = LoopAction.REPLAN if signal == "redirect" else LoopAction.RETRY
    return _adapt(action, reason, policy, budget)


def _recommended(
    observation: LoopObservation,
    evaluation: LoopEvaluation,
    policy: LoopPolicy,
    budget: LoopBudget,
) -> LoopDecision | None:
    action = evaluation.recommended_action
    reason = evaluation.reason
    if action is None or action is LoopAction.CONTINUE:
        return None
    if action is LoopAction.COMPLETE:
        return LoopDecision(
            action=action,
            reason=reason,
            termination_reason=LoopTerminationReason.OBJECTIVE_SATISFIED,
        )
    if action is LoopAction.STOP:
        return _stop(reason, LoopTerminationReason.EXPLICIT_STOP)
    if action is LoopAction.FAIL:
        return _fail(reason, LoopTerminationReason.EXECUTION_FAILED)
    if action in {LoopAction.WAIT, LoopAction.HUMAN_INTERVENTION}:
        if not policy.allow_human_intervention:
            return _stop(
                "human intervention is not permitted by the loop policy",
                LoopTerminationReason.POLICY_VIOLATION,
            )
        return LoopDecision(
            action=action,
            reason=reason,
            termination_reason=LoopTerminationReason.HUMAN_INTERVENTION,
        )
    if action is LoopAction.REFRESH:
        if not policy.allow_refresh:
            return _stop(
                "refresh is not permitted by the loop policy",
                LoopTerminationReason.POLICY_VIOLATION,
            )
        return _continuing(LoopAction.REFRESH, reason, budget)
    if action is LoopAction.CHANGE_TOPOLOGY:
        if not policy.allow_topology_change or observation.requested_topology is None:
            return _stop(
                "topology change is not permitted by the loop policy",
                LoopTerminationReason.POLICY_VIOLATION,
            )
        if budget.iterations >= budget.max_iterations:
            return _stop(
                "the iteration budget is exhausted",
                LoopTerminationReason.ITERATION_LIMIT,
            )
        return LoopDecision(
            action=action, reason=reason, next_topology=observation.requested_topology
        )
    return _adapt(action, reason, policy, budget)


def _adapt(
    action: LoopAction, reason: str, policy: LoopPolicy, budget: LoopBudget
) -> LoopDecision | None:
    if action is LoopAction.REPLAN:
        if not policy.allow_replan:
            return None
        if budget.replans >= budget.max_replans:
            return _stop(
                "the replan budget is exhausted", LoopTerminationReason.REPLAN_LIMIT
            )
        return _continuing(action, reason, budget)
    if not policy.allow_retry:
        return None
    if budget.retries >= budget.max_retries:
        return _stop("the retry budget is exhausted", LoopTerminationReason.RETRY_LIMIT)
    return _continuing(action, reason, budget)


def _continuing(action: LoopAction, reason: str, budget: LoopBudget) -> LoopDecision:
    if budget.iterations >= budget.max_iterations:
        return _stop(
            "the iteration budget is exhausted", LoopTerminationReason.ITERATION_LIMIT
        )
    return LoopDecision(action=action, reason=reason)


def _stagnation(policy: LoopPolicy, budget: LoopBudget) -> LoopDecision:
    if policy.allow_replan and budget.replans < budget.max_replans:
        return _continuing(
            LoopAction.REPLAN, "stagnation requires a new approach", budget
        )
    if policy.allow_refresh:
        return _continuing(
            LoopAction.REFRESH, "stagnation requires refreshed sources", budget
        )
    if policy.allow_human_intervention:
        return LoopDecision(
            action=LoopAction.HUMAN_INTERVENTION,
            reason="stagnation requires human intervention",
            termination_reason=LoopTerminationReason.HUMAN_INTERVENTION,
        )
    return _stop("objective progress stagnated", LoopTerminationReason.STAGNATION)


def _oscillates(state: LoopState, policy: LoopPolicy) -> bool:
    entries = state.history[-policy.oscillation_window :]
    if len(entries) < policy.oscillation_window:
        return False
    actions = tuple(entry.decision.action for entry in entries)
    topologies = tuple(entry.observation.topology for entry in entries)
    return (_alternates(actions) or _alternates(topologies)) and not any(
        entry.evaluation.progress for entry in entries
    )


def _alternates(values: tuple[object, ...]) -> bool:
    return (
        None not in values
        and len(set(values)) == 2
        and all(values[index] == values[index % 2] for index in range(len(values)))
    )


def _stop(reason: str, termination: LoopTerminationReason) -> LoopDecision:
    return LoopDecision(
        action=LoopAction.STOP, reason=reason, termination_reason=termination
    )


def _fail(reason: str, termination: LoopTerminationReason) -> LoopDecision:
    return LoopDecision(
        action=LoopAction.FAIL, reason=reason, termination_reason=termination
    )
