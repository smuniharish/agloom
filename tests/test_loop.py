"""Loop Engineering: models, observation, evaluation, control, and execution."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener
from langgraph.types import Command

from agloom import (
    LOOP_INTERRUPT,
    ConfigurationError,
    DefaultLoopController,
    DefaultLoopEvaluator,
    DefaultLoopObserver,
    EventName,
    ExecutionPaused,
    IterationFailure,
    LoopAction,
    LoopAssessment,
    LoopBudgetExceededError,
    LoopDecision,
    LoopDecisionError,
    LoopEvaluation,
    LoopEvaluationError,
    LoopObservation,
    LoopPhase,
    LoopPolicy,
    LoopResult,
    LoopState,
    LoopStateError,
    LoopTerminationReason,
    ModelLoopEvaluator,
    ObjectiveStatus,
    PipelineStage,
    ProgressStatus,
    TopologyName,
    behavior_signals,
    create_agent,
)
from agloom.loop.machine import LoopMachine, is_terminal
from agloom.loop.observer import fingerprint
from agloom.loop.runner import _standing, _with_notes, iteration_context
from agloom.loop.signals import collect_behavior_signals
from helpers import (
    Call,
    EventLog,
    ScriptedChatModel,
    allow_threads,
    analysis,
    tool_call,
)

Reason = LoopTerminationReason


def signals(text: str, **loop: Any) -> AIMessage:
    """An answer carrying explicit loop signals."""
    return AIMessage(content=text, additional_kwargs={"loop": loop})


def assessment(complete: bool, progress: float, **fields: Any) -> AIMessage:
    return tool_call(
        "LoopAssessment",
        {
            "objective_complete": complete,
            "progress": progress,
            "quality": fields.pop("quality", 0.9),
            "next_step": fields.pop(
                "next_step", "complete" if complete else "continue"
            ),
            **fields,
        },
    )


def direct(*answers: Any, judge: Callable[[Call], Any] | None = None) -> Any:
    """A responder for DIRECT runs that returns `answers` in order."""
    queue = list(answers)

    def respond(call: Call) -> Any:
        if call.wants("TaskAnalysis"):
            return analysis()
        if call.wants("LoopAssessment"):
            assert judge is not None
            return judge(call)
        return queue.pop(0)

    return respond


@pytest.fixture
def agloom_serde_events() -> Iterator[list[dict[str, Any]]]:
    """Record checkpoint deserialization of unregistered Agloom types."""
    recorded: list[dict[str, Any]] = []

    def listen(event: Any) -> None:
        if str(event.get("module", "")).startswith("agloom"):
            recorded.append(dict(event))

    unregister = register_serde_event_listener(listen)
    yield recorded
    unregister()


class TestModels:
    def test_policy_defaults_are_one_shot(self) -> None:
        policy = LoopPolicy()
        assert policy.max_iterations == 1
        assert not any(
            (
                policy.allow_retry,
                policy.allow_replan,
                policy.allow_refresh,
                policy.allow_topology_change,
                policy.allow_human_intervention,
            )
        )

    def test_policy_rejects_odd_oscillation_window(self) -> None:
        with pytest.raises(ValueError, match="must be even"):
            LoopPolicy(oscillation_window=5)

    def test_observation_and_decision_validation(self) -> None:
        with pytest.raises(ValueError, match="cannot contain an error"):
            LoopObservation(
                iteration=1, success=True, result_fingerprint="x", error_type="E"
            )
        with pytest.raises(ValueError, match="require error_type"):
            LoopObservation(iteration=1, success=False, result_fingerprint="x")
        with pytest.raises(ValueError, match="next_topology"):
            LoopDecision(action=LoopAction.CHANGE_TOPOLOGY, reason="r")
        with pytest.raises(ValueError, match="next_topology"):
            LoopDecision(
                action=LoopAction.CONTINUE,
                reason="r",
                next_topology=TopologyName.REACT,
            )

    def test_state_and_result_properties(self) -> None:
        machine = LoopMachine(
            LoopPolicy(), DefaultLoopObserver(), DefaultLoopController()
        )
        state = machine.start("goal", "loop-1")
        assert state.phase is LoopPhase.CREATED
        assert state.last_observation is None
        assert not state.terminal
        result = LoopResult(state=state, interrupts=("pending",))
        assert result.paused
        assert not LoopResult(state=state).paused


class TestObserver:
    observer = DefaultLoopObserver()

    def test_reads_signals_from_messages_and_mappings(self) -> None:
        message = signals(
            "x",
            objective_progress=0.5,
            objective_complete=False,
            quality=1,
            requested_topology=" Planner ",
            feedback="tighten",
            tool_calls=3,
        )
        observation = self.observer.observe(iteration=1, result=message)
        assert observation.objective_progress == 0.5
        assert observation.objective_complete is False
        assert observation.quality == 1.0
        assert observation.requested_topology is TopologyName.PLANNER
        assert observation.feedback == ("tighten",)
        assert observation.tool_calls == 3
        nested = self.observer.observe(
            iteration=1, result={"loop": {"retryable": True}, "other": 1}
        )
        assert nested.retryable
        flat = self.observer.observe(
            iteration=1, result={"replan_required": True, "feedback": ["a", 2]}
        )
        assert flat.replan_required
        assert flat.feedback == ("a", "2")

    def test_counts_tool_calls_and_metadata(self) -> None:
        message = AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        assert self.observer.observe(iteration=1, result=message).tool_calls == 1
        assert self.observer.observe(iteration=1, result="text").tool_calls == 0
        observed = self.observer.observe(
            iteration=1,
            result="text",
            metadata={"tool_calls": 4, "behavior_signals": ["stop"], "retryable": True},
        )
        assert observed.tool_calls == 4
        assert observed.behavior_signals == ("stop",)
        assert observed.retryable
        assert observed.metadata == {"tool_calls": 4, "retryable": True}

    @pytest.mark.parametrize(
        "loop",
        [
            {"objective_progress": "high"},
            {"objective_progress": True},
            {"objective_complete": "yes"},
            {"feedback": 3},
        ],
    )
    def test_rejects_malformed_signals(self, loop: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="loop"):
            self.observer.observe(iteration=1, result={"loop": loop})

    def test_failures_and_fingerprints(self) -> None:
        failure = IterationFailure("TimeoutError", "slow")
        observed = self.observer.observe(iteration=2, result="x", error=failure)
        assert not observed.success
        assert observed.error_type == "TimeoutError"
        assert observed.error_message == "slow"
        assert fingerprint(AIMessage(content="a"), None) == fingerprint(
            AIMessage(content="a", id="other"), None
        )
        assert fingerprint("a", None) != fingerprint("b", None)
        assert fingerprint({"b": 1, "a": 2}, None) == fingerprint(
            {"a": 2, "b": 1}, None
        )
        assert fingerprint({"m": AIMessage(content="x")}, None)
        assert fingerprint({1, 2}, None)
        assert fingerprint(None, ValueError("x")) != fingerprint(None, KeyError("x"))


def observation(**fields: Any) -> LoopObservation:
    fields.setdefault("iteration", 1)
    fields.setdefault("success", "error_type" not in fields)
    fields.setdefault("result_fingerprint", "f")
    return LoopObservation(**fields)


def state_after(
    policy: LoopPolicy, *entries: tuple[LoopObservation, LoopEvaluation]
) -> LoopState:
    machine = LoopMachine(policy, DefaultLoopObserver(), DefaultLoopController())
    state = machine.start("goal", "loop")
    for item, evaluation in entries:
        state = machine.begin(state, LoopAction.CONTINUE)
        state, _ = machine.decide(state, item, evaluation)
    return state


class TestDefaultEvaluator:
    evaluator = DefaultLoopEvaluator()
    policy = LoopPolicy(max_iterations=10, stagnation_threshold=2)

    def evaluate(self, state: LoopState, item: LoopObservation) -> LoopEvaluation:
        return self.evaluator.evaluate(state, item, self.policy)

    def test_progress_regression_and_completion(self) -> None:
        start = state_after(self.policy)
        first = self.evaluate(start, observation(objective_progress=0.3))
        assert first.progress
        assert first.progress_status is ProgressStatus.PROGRESS
        assert first.objective_status is ObjectiveStatus.IN_PROGRESS
        after = state_after(self.policy, (observation(objective_progress=0.3), first))
        regressed = self.evaluate(
            after, observation(iteration=2, objective_progress=0.1)
        )
        assert regressed.regression
        assert regressed.objective_status is ObjectiveStatus.AT_RISK
        flat = self.evaluate(after, observation(iteration=2, objective_progress=0.3))
        assert flat.progress_status is ProgressStatus.NO_PROGRESS
        done = self.evaluate(after, observation(iteration=2, objective_progress=1.0))
        assert done.objective_complete
        assert done.recommended_action is LoopAction.COMPLETE
        assert self.evaluate(
            start, observation(validation_passed=True)
        ).objective_complete

    @pytest.mark.parametrize(
        ("signals", "complete"),
        [
            ({"objective_progress": 1.0}, True),
            ({"objective_progress": 1.0, "objective_complete": False}, False),
            ({"objective_progress": 1.0, "validation_passed": False}, False),
            ({"objective_complete": False, "validation_passed": True}, False),
            ({"objective_complete": True, "validation_passed": False}, True),
            ({"objective_progress": 0.5, "validation_passed": True}, True),
            ({"objective_progress": 0.5}, False),
        ],
    )
    def test_explicit_completion_signals_take_precedence(
        self, signals: dict[str, Any], complete: bool
    ) -> None:
        evaluated = self.evaluate(state_after(self.policy), observation(**signals))
        assert evaluated.objective_complete is complete

    def test_explicitly_incomplete_results_follow_their_recommendation(self) -> None:
        policy = LoopPolicy(max_iterations=5, allow_replan=True, max_replans=2)
        state = state_after(policy)
        item = observation(
            objective_progress=1.0,
            objective_complete=False,
            validation_passed=False,
            replan_required=True,
            feedback=("C2: invalid citation",),
        )
        evaluation = DefaultLoopEvaluator().evaluate(state, item, policy)
        assert not evaluation.objective_complete
        assert evaluation.recommended_action is LoopAction.REPLAN
        assert decide(policy, item, evaluation).action is LoopAction.REPLAN

    def test_repeated_results_stagnate_or_converge(self) -> None:
        item = observation()
        first = self.evaluate(state_after(self.policy), item)
        assert first.progress_status is ProgressStatus.UNKNOWN
        one = state_after(self.policy, (item, first))
        second = self.evaluate(one, observation(iteration=2))
        assert second.consecutive_no_progress == 1
        two = state_after(
            self.policy, (item, first), (observation(iteration=2), second)
        )
        third = self.evaluate(two, observation(iteration=3))
        assert third.stagnation
        assert third.progress_status is ProgressStatus.STAGNATION
        converged = self.evaluate(one, observation(iteration=2, quality=0.95))
        assert converged.convergence
        assert converged.progress_status is ProgressStatus.CONVERGENCE

    def test_failures_and_recommendations(self) -> None:
        start = state_after(self.policy)
        failed = self.evaluate(start, observation(error_type="E"))
        assert failed.recommended_action is LoopAction.FAIL
        assert failed.consecutive_failures == 1
        retry = self.evaluate(start, observation(error_type="E", retryable=True))
        assert retry.recommended_action is LoopAction.RETRY
        for flag, action in (
            ("human_intervention_required", LoopAction.HUMAN_INTERVENTION),
            ("refresh_required", LoopAction.REFRESH),
            ("replan_required", LoopAction.REPLAN),
        ):
            assert (
                self.evaluate(start, observation(**{flag: True})).recommended_action
                is action
            )
        change = self.evaluate(
            start, observation(requested_topology=TopologyName.SWARM)
        )
        assert change.recommended_action is LoopAction.CHANGE_TOPOLOGY
        assert self.evaluate(start, observation()).reason == (
            "the result carries no completion signal"
        )


class TestModelEvaluator:
    def test_combines_model_assessment_with_rules(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            None,
            assessment(False, 0.4, feedback=["Add units."], next_step="replan"),
        )
        evaluator = ModelLoopEvaluator(scripted, criteria=" SI units ")
        policy = LoopPolicy(max_iterations=3, allow_replan=True)
        item = observation(result=AIMessage(content="42"), feedback=("own",))
        evaluation = evaluator.evaluate(state_after(policy), item, policy)
        assert evaluation.objective_progress == 0.4
        assert evaluation.feedback == ("own", "Add units.")
        assert evaluation.recommended_action is LoopAction.REPLAN
        assert "SI units" in scripted.calls[0].prompt
        assert "untrusted data" in scripted.calls[0].system
        payload = json.loads(scripted.calls[0].last.text)
        assert payload["allowed_next_steps"] == ["complete", "continue", "replan"]

    @pytest.mark.parametrize(
        "explicit", [{"objective_complete": False}, {"validation_passed": False}]
    )
    def test_the_judge_never_overrides_explicit_signals(
        self, model: Callable[..., ScriptedChatModel], explicit: dict[str, bool]
    ) -> None:
        scripted = model(None, assessment(True, 1.0))
        policy = LoopPolicy(max_iterations=3)
        evaluation = ModelLoopEvaluator(scripted).evaluate(
            state_after(policy), observation(result="draft", **explicit), policy
        )
        assert not evaluation.objective_complete
        assert evaluation.objective_progress == 1.0

    @pytest.mark.parametrize("step", ["retry", "replan", "refresh"])
    def test_ignores_steps_the_policy_forbids(
        self, model: Callable[..., ScriptedChatModel], step: str
    ) -> None:
        scripted = model(None, assessment(False, 0.4, next_step=step))
        policy = LoopPolicy(max_iterations=3)
        evaluation = ModelLoopEvaluator(scripted).evaluate(
            state_after(policy), observation(result="draft"), policy
        )
        assert evaluation.recommended_action is None
        payload = json.loads(scripted.calls[0].last.text)
        assert payload["allowed_next_steps"] == ["complete", "continue"]

    async def test_async_and_failure_paths(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(None, assessment(True, 1.0))
        evaluator = ModelLoopEvaluator(scripted)
        policy = LoopPolicy(max_iterations=3)
        done = await evaluator.aevaluate(
            state_after(policy), observation(result={"value": 1}), policy
        )
        assert done.objective_complete
        failed = await evaluator.aevaluate(
            state_after(policy), observation(error_type="E"), policy
        )
        assert failed.recommended_action is LoopAction.FAIL
        assert evaluator.evaluate(
            state_after(policy), observation(error_type="E"), policy
        )
        assert len(scripted.calls) == 1

    def test_validation(self, model: Callable[..., ScriptedChatModel]) -> None:
        with pytest.raises(ConfigurationError, match="positive"):
            ModelLoopEvaluator(model(), max_result_chars=0)
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        with pytest.raises(ConfigurationError, match="structured output"):
            ModelLoopEvaluator(FakeListChatModel(responses=["x"]))
        evaluator = ModelLoopEvaluator(model(None, AIMessage(content="no tool call")))
        with pytest.raises(LoopEvaluationError, match="no assessment"):
            evaluator.evaluate(
                state_after(LoopPolicy()), observation(result="x"), LoopPolicy()
            )

    def test_assessment_schema(self) -> None:
        parsed = LoopAssessment(
            objective_complete=False, progress=0.5, quality=0.5, next_step="retry"
        )
        assert parsed.feedback == []


def decide(
    policy: LoopPolicy,
    item: LoopObservation,
    evaluation: LoopEvaluation | None = None,
    *,
    state: LoopState | None = None,
    **budget: Any,
) -> LoopDecision:
    current = state or state_after(policy)
    spent = current.budget.model_copy(update={"iterations": 1, **budget})
    return DefaultLoopController().decide(
        current,
        item,
        evaluation or LoopEvaluation(reason="r"),
        policy,
        spent,
    )


class TestController:
    def test_failures(self) -> None:
        policy = LoopPolicy(max_iterations=3, allow_retry=True, max_retries=1)
        cases = {
            "LoopBudgetExceededError": (LoopAction.STOP, Reason.TOOL_CALL_LIMIT),
            "ApprovalRequired": (
                LoopAction.HUMAN_INTERVENTION,
                Reason.HUMAN_INTERVENTION,
            ),
            "PolicyViolationError": (LoopAction.STOP, Reason.POLICY_VIOLATION),
            "ActionOutcomeUnknown": (LoopAction.FAIL, Reason.EXECUTION_FAILED),
        }
        for error, (action, reason) in cases.items():
            decision = decide(policy, observation(error_type=error))
            assert (decision.action, decision.termination_reason) == (action, reason)
        assert decide(policy, observation(error_type="E")).action is LoopAction.RETRY
        timeout = decide(policy, observation(error_type="E"), elapsed=10_000)
        assert timeout.termination_reason is Reason.TIMEOUT
        exhausted = decide(policy, observation(error_type="E"), retries=1)
        assert exhausted.termination_reason is Reason.RETRY_LIMIT
        last = decide(policy, observation(error_type="E"), iterations=3)
        assert last.termination_reason is Reason.ITERATION_LIMIT
        no_retry = decide(LoopPolicy(), observation(error_type="E"))
        assert no_retry.termination_reason is Reason.EXECUTION_FAILED

    def test_completion_precedes_budgets(self) -> None:
        policy = LoopPolicy(max_iterations=3)
        complete = LoopEvaluation(reason="done", objective_complete=True)
        decision = decide(policy, observation(), complete, elapsed=10_000)
        assert decision.termination_reason is Reason.OBJECTIVE_SATISFIED
        validated = decide(policy, observation(validation_passed=True), complete)
        assert validated.termination_reason is Reason.VALIDATED
        converged = decide(
            policy, observation(), LoopEvaluation(reason="c", convergence=True)
        )
        assert converged.termination_reason is Reason.CONVERGED
        assert decide(policy, observation(), elapsed=10_000).termination_reason is (
            Reason.TIMEOUT
        )
        assert decide(policy, observation(), tool_calls=10_000).termination_reason is (
            Reason.TOOL_CALL_LIMIT
        )

    def test_behavior_signals(self) -> None:
        strict = LoopPolicy(max_iterations=3)
        open_policy = LoopPolicy(
            max_iterations=3,
            allow_human_intervention=True,
            allow_retry=True,
            max_retries=1,
            allow_replan=True,
            max_replans=1,
        )

        def behaved(policy: LoopPolicy, *kinds: str) -> LoopDecision:
            return decide(policy, observation(behavior_signals=kinds))

        assert behaved(strict, "retry", "stop").termination_reason is (
            Reason.BEHAVIOR_INTERVENTION
        )
        assert (
            behaved(strict, "pause").termination_reason is Reason.BEHAVIOR_INTERVENTION
        )
        assert behaved(open_policy, "pause").action is LoopAction.WAIT
        assert behaved(open_policy, "escalate").action is LoopAction.HUMAN_INTERVENTION
        assert behaved(open_policy, "redirect").action is LoopAction.REPLAN
        assert behaved(open_policy, "retry").action is LoopAction.RETRY
        assert behaved(strict, "retry").action is LoopAction.CONTINUE
        assert behaved(strict, "unknown").action is LoopAction.CONTINUE

    def test_recommendations(self) -> None:
        strict = LoopPolicy(max_iterations=3)
        permissive = LoopPolicy(
            max_iterations=3,
            allow_human_intervention=True,
            allow_refresh=True,
            allow_topology_change=True,
            allow_replan=True,
            max_replans=1,
            allow_retry=True,
            max_retries=1,
        )

        def recommended(
            policy: LoopPolicy,
            action: LoopAction,
            item: LoopObservation | None = None,
            **budget: Any,
        ) -> LoopDecision:
            evaluation = LoopEvaluation(reason="hint", recommended_action=action)
            return decide(policy, item or observation(), evaluation, **budget)

        assert recommended(strict, LoopAction.COMPLETE).action is LoopAction.COMPLETE
        assert (
            recommended(strict, LoopAction.STOP).termination_reason
            is Reason.EXPLICIT_STOP
        )
        assert recommended(strict, LoopAction.FAIL).action is LoopAction.FAIL
        for action in (LoopAction.WAIT, LoopAction.REFRESH, LoopAction.CHANGE_TOPOLOGY):
            assert (
                recommended(strict, action).termination_reason
                is Reason.POLICY_VIOLATION
            )
        assert recommended(permissive, LoopAction.WAIT).action is LoopAction.WAIT
        assert recommended(permissive, LoopAction.REFRESH).action is LoopAction.REFRESH
        change = recommended(
            permissive,
            LoopAction.CHANGE_TOPOLOGY,
            observation(requested_topology=TopologyName.PLANNER),
        )
        assert change.next_topology is TopologyName.PLANNER
        limited = recommended(
            permissive,
            LoopAction.CHANGE_TOPOLOGY,
            observation(requested_topology=TopologyName.PLANNER),
            iterations=3,
        )
        assert limited.termination_reason is Reason.ITERATION_LIMIT
        assert recommended(permissive, LoopAction.REPLAN).action is LoopAction.REPLAN
        assert recommended(
            permissive, LoopAction.REPLAN, replans=1
        ).termination_reason is (Reason.REPLAN_LIMIT)
        assert recommended(
            permissive, LoopAction.RETRY, retries=1
        ).termination_reason is (Reason.RETRY_LIMIT)
        assert recommended(strict, LoopAction.REPLAN).action is LoopAction.CONTINUE
        assert recommended(
            permissive, LoopAction.REFRESH, iterations=3
        ).termination_reason is (Reason.ITERATION_LIMIT)
        assert recommended(strict, LoopAction.CONTINUE).action is LoopAction.CONTINUE

    def test_progress_rules(self) -> None:
        one_shot = decide(LoopPolicy(), observation())
        assert one_shot.termination_reason is Reason.ONE_SHOT_COMPLETE
        stagnated = LoopEvaluation(reason="s", stagnation=True)
        policies = [
            (
                LoopPolicy(max_iterations=3, allow_replan=True, max_replans=1),
                LoopAction.REPLAN,
            ),
            (LoopPolicy(max_iterations=3, allow_refresh=True), LoopAction.REFRESH),
            (
                LoopPolicy(max_iterations=3, allow_human_intervention=True),
                LoopAction.HUMAN_INTERVENTION,
            ),
            (LoopPolicy(max_iterations=3), LoopAction.STOP),
        ]
        for policy, action in policies:
            assert decide(policy, observation(), stagnated).action is action
        limit = decide(LoopPolicy(max_iterations=3), observation(), iterations=3)
        assert limit.termination_reason is Reason.ITERATION_LIMIT
        assert (
            decide(LoopPolicy(max_iterations=3), observation()).action
            is LoopAction.CONTINUE
        )

    def test_oscillation(self) -> None:
        policy = LoopPolicy(max_iterations=10, oscillation_window=4)
        entries = []
        for index, topology in enumerate(
            [TopologyName.REACT, TopologyName.PLANNER] * 2, start=1
        ):
            entries.append(
                (
                    observation(iteration=index, topology=topology),
                    LoopEvaluation(reason="r"),
                )
            )
        oscillating = state_after(policy, *entries)
        decision = decide(
            policy, observation(iteration=5), state=oscillating, iterations=5
        )
        assert decision.termination_reason is Reason.OSCILLATION


class TestMachine:
    def test_begin_rejects_iterations_beyond_budget(self) -> None:
        machine = LoopMachine(
            LoopPolicy(), DefaultLoopObserver(), DefaultLoopController()
        )
        state = machine.begin(machine.start("goal", "loop"), LoopAction.CONTINUE)
        with pytest.raises(LoopStateError):
            machine.begin(state, LoopAction.RETRY)

    def test_controller_failures(self) -> None:
        class Broken:
            def decide(self, *args: Any) -> Any:
                raise RuntimeError("boom")

        class Empty:
            def decide(self, *args: Any) -> Any:
                return None

        for controller, message in ((Broken(), "failed"), (Empty(), "no LoopDecision")):
            machine = LoopMachine(LoopPolicy(), DefaultLoopObserver(), controller)
            state = machine.begin(machine.start("g", "l"), LoopAction.CONTINUE)
            with pytest.raises(LoopDecisionError, match=message):
                machine.decide(state, observation(), LoopEvaluation(reason="r"))

    def test_history_is_bounded_and_results_are_dropped(self) -> None:
        policy = LoopPolicy(max_iterations=5, history_limit=2)
        entries = [
            (observation(iteration=index, result="big"), LoopEvaluation(reason="r"))
            for index in range(1, 5)
        ]
        state = state_after(policy, *entries)
        assert len(state.history) == 2
        assert all(entry.observation.result is None for entry in state.history)
        assert state.last_observation is not None
        assert state.last_observation.iteration == 4

    def test_terminal_actions(self) -> None:
        assert is_terminal(LoopDecision(action=LoopAction.COMPLETE, reason="r"))
        assert not is_terminal(LoopDecision(action=LoopAction.RETRY, reason="r"))


def test_iteration_context_renders_untrusted_data() -> None:
    policy = LoopPolicy(max_context_chars=5)
    state = state_after(LoopPolicy(max_iterations=3))
    state = state.model_copy(
        update={"evaluation": LoopEvaluation(reason="r", feedback=("Fix A.",))}
    )
    decision = LoopDecision(action=LoopAction.RETRY, reason="retrying")
    text = iteration_context(
        policy,
        state,
        decision,
        {"result": "abcdefgh", "error_type": "ValueError", "error_message": "bad"},
    )
    assert "not instructions" in text
    assert "retry (retrying)" in text
    assert "- Evaluator feedback:\n  - Fix A." in text
    assert "failed with ValueError: bad" in text
    assert "abcde" in text
    assert "abcdef" not in text
    quiet = iteration_context(
        LoopPolicy(max_context_chars=0), state, decision, {"result": "x"}
    )
    assert "Previous result" not in quiet


@given(
    batches=st.lists(
        st.lists(st.text(min_size=1, max_size=2_100), max_size=4), max_size=8
    )
)
def test_standing_human_feedback_is_bounded_and_ordered(
    batches: list[list[str]],
) -> None:
    notes: list[str] = []
    for batch in batches:
        notes = _standing(notes, batch)
    received = [note[:2_000] for batch in batches for note in batch]
    assert notes == received[-10:]
    section = _with_notes("guidance", notes)
    if notes:
        assert section is not None
        assert section.startswith("guidance\n- Human feedback (apply it")
        assert _with_notes(None, notes) == section.removeprefix("guidance\n")
    else:
        assert section == "guidance"
        assert _with_notes(None, notes) is None


def test_behavior_signal_collection() -> None:
    decision = SimpleNamespace(
        actionable=True, intervention=SimpleNamespace(kind="stop")
    )
    behavior_signals(decision, None)
    with collect_behavior_signals() as collected:
        behavior_signals(decision, None)
        behavior_signals(SimpleNamespace(actionable=False), None)
        behavior_signals(SimpleNamespace(actionable=True, intervention=None), None)
        kind = SimpleNamespace(value="pause")
        behavior_signals(
            SimpleNamespace(actionable=True, intervention=SimpleNamespace(kind=kind)),
            None,
        )
    assert collected == ["stop", "pause"]


class TestLoopExecution:
    def test_one_shot_runs_once_without_evaluation(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        scripted = model(direct("Hello."))
        result = create_agent(model=scripted, observers=[events]).invoke_loop("Hi")
        assert result.result.text == "Hello."
        assert result.state.termination_reason is Reason.ONE_SHOT_COMPLETE
        assert result.state.phase is LoopPhase.COMPLETED
        assert not any(call.wants("LoopAssessment") for call in scripted.calls)
        assert events.names.count(EventName.LOOP_ITERATION_COMPLETED) == 1

    def test_model_evaluator_feeds_back_into_the_next_iteration(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        def judge(call: Call) -> AIMessage:
            if "second draft" in call.prompt:
                return assessment(True, 1.0)
            return assessment(False, 0.4, feedback=["Add a title."])

        scripted = model(direct("first draft", "second draft", judge=judge))
        agent = create_agent(model=scripted, loop_policy=LoopPolicy(max_iterations=3))
        result = agent.invoke_loop("Write a note")
        assert result.result.text == "second draft"
        assert result.state.iteration == 2
        assert result.state.termination_reason is Reason.OBJECTIVE_SATISFIED
        second = [c for c in scripted.calls if "Loop Engineering context" in c.prompt]
        assert len(second) == 1
        assert "Add a title." in second[0].prompt
        assert "first draft" in second[0].prompt

    def test_retry_after_failed_iteration(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        attempts: list[int] = []

        @tool
        def fetch() -> str:
            """Fetch data."""
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("upstream timeout")
            return "data"

        scripted = model(
            lambda call: (
                signals("Fetched data.", objective_complete=True)
                if call.last.type == "tool"
                else tool_call("fetch", {})
            )
        )
        agent = create_agent(
            model=scripted,
            tools=[fetch],
            pattern="react",
            loop_policy=LoopPolicy(max_iterations=2, allow_retry=True, max_retries=1),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        result = agent.invoke_loop("Fetch")
        assert result.state.iteration == 2
        assert result.state.budget.retries == 1
        assert result.state.history[0].decision.action is LoopAction.RETRY
        assert result.result.text == "Fetched data."

    def test_replan_and_stagnation(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        replanning = create_agent(
            model=model(
                direct(
                    signals("v1", replan_required=True),
                    signals("v2", objective_complete=True),
                )
            ),
            loop_policy=LoopPolicy(max_iterations=3, allow_replan=True, max_replans=1),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        replanned = replanning.invoke_loop("Design")
        assert replanned.state.budget.replans == 1
        assert replanned.result.text == "v2"

        stalling = create_agent(
            model=model(direct("same", "same", "same", "same")),
            loop_policy=LoopPolicy(max_iterations=5, stagnation_threshold=2),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        stalled = stalling.invoke_loop("Improve")
        assert stalled.state.termination_reason is Reason.STAGNATION
        assert stalled.state.iteration == 3

    def test_human_intervention_pauses_and_resumes_with_feedback(
        self,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        agloom_serde_events: list[dict[str, Any]],
    ) -> None:
        scripted = model(
            direct(
                signals("draft", human_intervention_required=True),
                signals("revised", objective_progress=0.6),
                signals("final", objective_complete=True),
            )
        )
        agent = create_agent(
            model=scripted,
            loop_policy=LoopPolicy(max_iterations=3, allow_human_intervention=True),
            loop_evaluator=DefaultLoopEvaluator(),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "human"}}
        paused = agent.invoke_loop("Draft a plan", config)
        assert paused.paused
        assert paused.state.phase is LoopPhase.WAITING
        payload = paused.interrupts[0].value
        assert payload["type"] == LOOP_INTERRUPT
        assert payload["action"] == "human_intervention"
        assert agent.resume(config, {"feedback": ["Use metric units."]}).text == "final"
        answers = [c for c in scripted.calls if not c.wants("TaskAnalysis")]
        assert len(answers) == 3
        assert "Use metric units." not in answers[0].prompt
        for later in answers[1:]:
            assert "Human feedback (apply it" in later.prompt
            assert "Use metric units." in later.prompt
        assert agloom_serde_events == []

    def test_resume_at_iteration_limit_stops(
        self, model: Callable[..., ScriptedChatModel], saver: Any
    ) -> None:
        agent = create_agent(
            model=model(direct(signals("draft", human_intervention_required=True))),
            loop_policy=LoopPolicy(allow_human_intervention=True),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "limit"}}
        assert agent.invoke_loop("Draft", config).paused
        assert agent.invoke_loop(None, config).paused
        with pytest.raises(ExecutionPaused):
            agent.invoke(None, config)
        final = agent.invoke_loop(Command(resume="ok"), config)
        assert final.state.termination_reason is Reason.ITERATION_LIMIT

    def test_human_intervention_without_checkpointer_returns_waiting_state(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(
                direct(
                    signals("draft", human_intervention_required=True),
                    signals("draft", human_intervention_required=True),
                )
            ),
            loop_policy=LoopPolicy(allow_human_intervention=True),
        )
        result = agent.invoke_loop("Draft")
        assert result.state.phase is LoopPhase.WAITING
        assert result.state.termination_reason is Reason.HUMAN_INTERVENTION
        assert agent.invoke("Draft").text == "draft"

    def test_topology_change(self, model: Callable[..., ScriptedChatModel]) -> None:
        plan = {"steps": [{"id": "a", "task": "Do A"}]}

        def respond(call: Call) -> Any:
            if call.wants("TaskAnalysis"):
                return analysis()
            if call.wants("Plan"):
                return tool_call("Plan", plan)
            if "assigned step" in call.system:
                return "step done"
            if "Combine the step results" in call.system:
                return "planned"
            return signals("direct", requested_topology="planner")

        agent = create_agent(
            model=model(respond),
            loop_policy=LoopPolicy(max_iterations=2, allow_topology_change=True),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        result = agent.invoke_loop("Do it")
        assert result.state.history[0].decision.next_topology is TopologyName.PLANNER
        assert result.state.active_topology is TopologyName.PLANNER
        assert result.result.text == "planned"

    def test_tool_budget(self, model: Callable[..., ScriptedChatModel]) -> None:
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        agent = create_agent(
            model=model(lambda call: tool_call("ping", {})),
            tools=[ping],
            pattern="react",
            loop_policy=LoopPolicy(max_tool_calls=0),
        )
        result = agent.invoke_loop("Ping")
        assert result.state.termination_reason is Reason.TOOL_CALL_LIMIT
        assert isinstance(result.error, LoopBudgetExceededError)
        with pytest.raises(LoopBudgetExceededError):
            agent.invoke("Ping")

    def test_behavior_signals_stop_the_loop(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        @tool
        def act() -> str:
            """Act."""
            behavior_signals(
                SimpleNamespace(
                    actionable=True, intervention=SimpleNamespace(kind="stop")
                ),
                None,
            )
            return "acted"

        agent = create_agent(
            model=model(
                lambda call: (
                    "done" if call.last.type == "tool" else tool_call("act", {})
                )
            ),
            tools=[act],
            pattern="react",
        )
        result = agent.invoke_loop("Act")
        assert result.state.termination_reason is Reason.BEHAVIOR_INTERVENTION
        assert result.state.history[0].observation.behavior_signals == ("stop",)

    def test_evaluation_and_observation_errors_are_wrapped(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Failing:
            def evaluate(self, *args: Any) -> Any:
                raise RuntimeError("judge down")

        failing = create_agent(model=model(direct("x")), loop_evaluator=Failing())
        with pytest.raises(LoopEvaluationError, match="evaluation failed"):
            failing.invoke("Hi")
        malformed = create_agent(
            model=model(direct(signals("x", objective_progress="most")))
        )
        with pytest.raises(LoopEvaluationError, match="observation failed"):
            malformed.invoke("Hi")

    async def test_async_loop_with_integrations(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        refreshed: list[int] = []
        submitted: list[dict[str, Any]] = []
        acknowledged: list[str] = []

        class Refresh:
            async def refresh(self) -> Any:
                refreshed.append(1)
                return SimpleNamespace(status="completed", refreshed_count=2)

        class Feedback:
            async def query(self, query: Any) -> list[Any]:
                assert query.target_id == "async-thread"
                if acknowledged:
                    return []
                return [
                    SimpleNamespace(feedback_id="f1", payload={"text": "Cite sources."})
                ]

            async def acknowledge(self, feedback_id: str) -> None:
                acknowledged.append(feedback_id)

            async def submit(self, **fields: Any) -> None:
                submitted.append(fields)

        def judge(call: Call) -> AIMessage:
            if json.loads(call.last.text)["result"] == "fresh":
                return assessment(True, 1.0)
            return assessment(
                False, 0.2, feedback=["Refresh first."], next_step="refresh"
            )

        scripted = model(direct("stale", "fresh", judge=judge))
        agent = create_agent(
            model=scripted,
            loop_policy=LoopPolicy(max_iterations=3, allow_refresh=True),
            refresh=Refresh(),  # type: ignore[arg-type]
            feedback=Feedback(),  # type: ignore[arg-type]
        )
        result = await agent.ainvoke_loop(
            "Report", {"configurable": {"thread_id": "async-thread"}}
        )
        assert result.result.text == "fresh"
        assert refreshed == [1]
        assert acknowledged == ["f1"]
        answers = [
            c
            for c in scripted.calls
            if not (c.wants("TaskAnalysis") or c.wants("LoopAssessment"))
        ]
        assert len(answers) == 2
        assert all("Cite sources." in answer.prompt for answer in answers)
        assert submitted
        assert submitted[0]["payload"]["feedback"] == ["Refresh first."]
        with pytest.raises(ConfigurationError, match="asynchronous"):
            agent.invoke("Report")

    async def test_refresh_without_engine_from_custom_controller(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Refreshing:
            def decide(self, *args: Any) -> LoopDecision:
                return LoopDecision(action=LoopAction.REFRESH, reason="custom")

        agent = create_agent(
            model=model(direct("a", "b")),
            loop_policy=LoopPolicy(max_iterations=2),
            loop_controller=Refreshing(),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        with pytest.raises(ConfigurationError, match="without a refresh engine"):
            await agent.ainvoke_loop("Go")
        with pytest.raises(ConfigurationError, match="refresh is asynchronous"):
            agent.invoke_loop("Go")


outcome_signals = st.fixed_dictionaries(
    {},
    optional={
        "objective_progress": st.floats(0, 1),
        "objective_complete": st.booleans(),
        "quality": st.floats(0, 1),
        "validation_passed": st.booleans(),
        "retryable": st.booleans(),
        "refresh_required": st.booleans(),
        "replan_required": st.booleans(),
        "human_intervention_required": st.booleans(),
        "requested_topology": st.sampled_from([t.value for t in TopologyName]),
    },
)
outcomes = st.one_of(
    st.builds(
        lambda loop, text, signals_: {
            "result": {"loop": loop, "text": text},
            "error_type": None,
            "duration": 1.0,
            "tool_calls": 1,
            "behavior_signals": signals_,
        },
        outcome_signals,
        st.sampled_from(["a", "b"]),
        st.lists(
            st.sampled_from(["stop", "pause", "retry", "redirect", "escalate"]),
            max_size=1,
        ),
    ),
    st.builds(
        lambda error: {
            "result": None,
            "error_type": error,
            "error_message": "x",
            "duration": 1.0,
            "tool_calls": 0,
        },
        st.sampled_from(
            [
                "TimeoutError",
                "ApprovalRequired",
                "PolicyViolationError",
                "ActionOutcomeUnknown",
            ]
        ),
    ),
)
policies = st.builds(
    LoopPolicy,
    max_iterations=st.integers(1, 6),
    max_retries=st.integers(0, 2),
    max_replans=st.integers(0, 2),
    history_limit=st.integers(1, 4),
    stagnation_threshold=st.integers(2, 3),
    allow_retry=st.booleans(),
    allow_replan=st.booleans(),
    allow_refresh=st.booleans(),
    allow_topology_change=st.booleans(),
    allow_human_intervention=st.booleans(),
)

_PERMISSIONS = {
    LoopAction.RETRY: "allow_retry",
    LoopAction.REPLAN: "allow_replan",
    LoopAction.REFRESH: "allow_refresh",
    LoopAction.CHANGE_TOPOLOGY: "allow_topology_change",
    LoopAction.WAIT: "allow_human_intervention",
}


@given(policies, st.lists(outcomes, min_size=1, max_size=12))
def test_loop_simulation_respects_policy_and_budgets(
    policy: LoopPolicy, iterations: list[dict[str, Any]]
) -> None:
    machine = LoopMachine(policy, DefaultLoopObserver(), DefaultLoopController())
    evaluator = DefaultLoopEvaluator()
    state = machine.start("objective", "loop")
    action = LoopAction.CONTINUE
    for outcome in iterations:
        state = machine.begin(state, action)
        item = machine.observe(state, outcome, None)
        evaluation = evaluator.evaluate(state, item, policy)
        state, decision = machine.decide(state, item, evaluation)
        assert len(state.history) <= policy.history_limit
        assert state.budget.retries <= policy.max_retries
        assert state.budget.replans <= policy.max_replans
        permission = _PERMISSIONS.get(decision.action)
        if permission is not None:
            assert getattr(policy, permission)
        if decision.action is LoopAction.HUMAN_INTERVENTION:
            assert policy.allow_human_intervention or outcome["error_type"] == (
                "ApprovalRequired"
            )
        if is_terminal(decision):
            assert decision.termination_reason is not None
            terminal = LoopMachine.terminate(state, decision)
            assert terminal.terminal
            break
        action = decision.action
    assert state.budget.iterations <= policy.max_iterations


class TestLoopEdgeCases:
    def test_evaluator_and_observer_errors(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Strict:
            def evaluate(self, *args: Any) -> Any:
                raise LoopEvaluationError("strict evaluator")

        class Observer:
            def observe(self, **fields: Any) -> Any:
                raise LoopEvaluationError("strict observer")

        with pytest.raises(LoopEvaluationError, match="strict evaluator"):
            create_agent(model=model(direct("x")), loop_evaluator=Strict()).invoke("go")
        with pytest.raises(LoopEvaluationError, match="strict observer"):
            create_agent(model=model(direct("x")), loop_observer=Observer()).invoke(
                "go"
            )

    async def test_async_evaluator_errors(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Strict:
            async def aevaluate(self, *args: Any) -> Any:
                raise LoopEvaluationError("async strict")

            def evaluate(self, *args: Any) -> Any:
                raise AssertionError("the async path must be used")

        class Broken:
            def evaluate(self, *args: Any) -> Any:
                raise RuntimeError("down")

        with pytest.raises(LoopEvaluationError, match="async strict"):
            await create_agent(
                model=model(direct("x")), loop_evaluator=Strict()
            ).ainvoke("go")
        with pytest.raises(LoopEvaluationError, match="evaluation failed"):
            await create_agent(
                model=model(direct("x")), loop_evaluator=Broken()
            ).ainvoke("go")

    async def test_async_human_pause_and_resume(
        self, model: Callable[..., ScriptedChatModel], saver: Any
    ) -> None:
        agent = create_agent(
            model=model(
                direct(
                    signals("draft", human_intervention_required=True),
                    signals("final", objective_complete=True),
                )
            ),
            loop_policy=LoopPolicy(max_iterations=2, allow_human_intervention=True),
            loop_evaluator=DefaultLoopEvaluator(),
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "async-human"}}
        assert (await agent.ainvoke_loop("Draft", config)).paused
        resumed = await agent.aresume(config, "Use metric units.")
        assert resumed.text == "final"

    def test_empty_results_carry_no_previous_result(self) -> None:
        state = state_after(LoopPolicy(max_iterations=3))
        decision = LoopDecision(action=LoopAction.CONTINUE, reason="go")
        text = iteration_context(
            LoopPolicy(), state, decision, {"result": AIMessage(content="")}
        )
        assert "Previous result" not in text

    async def test_feedback_edge_cases(
        self, model: Callable[..., ScriptedChatModel], caplog: pytest.LogCaptureFixture
    ) -> None:
        queried: list[Any] = []

        class Feedback:
            async def query(self, query: Any) -> list[Any]:
                queried.append(query)
                return [SimpleNamespace(feedback_id="f1", payload={"rating": 2})]

            async def acknowledge(self, feedback_id: str) -> None:
                return

            async def submit(self, **fields: Any) -> None:
                raise RuntimeError("store down")

        def judge(call: Call) -> AIMessage:
            return assessment(True, 1.0, feedback=["Fine."])

        agent = create_agent(
            model=model(direct("a", "b", judge=judge)),
            loop_policy=LoopPolicy(max_iterations=2),
            feedback=Feedback(),  # type: ignore[arg-type]
        )
        assert (await agent.ainvoke("no thread")).text == "a"
        assert queried == []
        threaded = await agent.ainvoke("thread", {"configurable": {"thread_id": "t"}})
        assert threaded.text == "b"
        assert len(queried) == 1
        assert "recording loop evaluation feedback failed" in caplog.text


def test_unserializable_values_are_fingerprinted_and_rendered() -> None:
    from agloom.loop.evaluator import _json

    circular: list[Any] = []
    circular.append(circular)
    assert fingerprint(circular, None)
    assert _json(circular).startswith("[")
    assert _json("text") == "text"


async def test_async_resume_at_the_iteration_limit(
    model: Callable[..., ScriptedChatModel], saver: Any
) -> None:
    agent = create_agent(
        model=model(direct(signals("draft", human_intervention_required=True))),
        loop_policy=LoopPolicy(allow_human_intervention=True),
        checkpointer=saver,
        checkpoint_authorizer=allow_threads,
    )
    config = {"configurable": {"thread_id": "async-limit"}}
    assert (await agent.ainvoke_loop("Draft", config)).paused
    final = await agent.ainvoke_loop(Command(resume="ok"), config)
    assert final.state.termination_reason is Reason.ITERATION_LIMIT


async def test_async_loop_continues_between_iterations(
    model: Callable[..., ScriptedChatModel],
) -> None:
    def judge(call: Call) -> AIMessage:
        if "second" in call.prompt:
            return assessment(True, 1.0)
        return assessment(False, 0.5, feedback=["Expand it."])

    agent = create_agent(
        model=model(direct("first", "second", judge=judge)),
        loop_policy=LoopPolicy(max_iterations=3),
    )
    result = await agent.ainvoke_loop("Write")
    assert result.state.iteration == 2
    assert result.state.history[0].decision.action is LoopAction.CONTINUE


@pytest.mark.parametrize("pattern", ["pipeline", "hybrid"])
def test_topologies_preserve_answer_signals(
    model: Callable[..., ScriptedChatModel], pattern: str
) -> None:
    options: dict[str, Any] = (
        {"stages": [PipelineStage(name="only", instruction="Do it.")]}
        if pattern == "pipeline"
        else {"composition": ["react", "pipeline"]}
    )
    agent = create_agent(
        model=model(lambda call: signals("done", objective_complete=True)),
        pattern=pattern,
        loop_policy=LoopPolicy(max_iterations=3),
        loop_evaluator=DefaultLoopEvaluator(),
        **options,
    )
    result = agent.invoke_loop("go")
    assert result.state.iteration == 1
    assert result.state.termination_reason is Reason.OBJECTIVE_SATISFIED
