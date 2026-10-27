import json

from backend.agent import loop_policy
from backend.models import Citation, Claim, Iteration, Source, Synthesis
from backend.validation import (
    EvidenceLoopObserver,
    approval_ready,
    extract_synthesis,
    loop_signals,
    score_synthesis,
)
from langchain_core.messages import AIMessage

from agloom import (
    DefaultLoopController,
    DefaultLoopEvaluator,
    LoopAction,
    LoopBudget,
    LoopPolicy,
    LoopState,
)
from tests.support import SOURCES, synthesis


def source() -> Source:
    return Source(
        id="SRC-1",
        title="MFA",
        uri="memory://mfa",
        revision="1",
        body="Production access requires phishing-resistant MFA.",
    )


def supported(claim_id: str = "C1", quote: str = "phishing-resistant MFA") -> Claim:
    return Claim(
        id=claim_id,
        topic="access",
        text="MFA required",
        status="supported",
        citations=[Citation(source_id="SRC-1", quote=quote)],
    )


class FixedRevisions:
    def __init__(self, changed: bool) -> None:
        self.changed = changed

    def has_newer_revisions(self, current: list[Source]) -> bool:
        return self.changed


def test_deterministic_accuracy_and_exact_quote_enforcement():
    result = Synthesis(summary="MFA is required.", claims=[supported()])
    assert score_synthesis(result, [source()], ["access"]) == score_synthesis(
        result, [source()], ["access"]
    )
    score = score_synthesis(result, [source()], ["access"])
    assert score.citation_precision == score.coverage == 1
    assert score.quality == 1
    assert approval_ready(score)


def test_unknown_or_nonverbatim_citation_fails_validation():
    result = Synthesis(
        summary="Unsupported.",
        claims=[
            Claim(
                id="C1",
                topic="access",
                text="Claim",
                status="supported",
                citations=[Citation(source_id="SRC-X", quote="not present")],
            )
        ],
    )
    score = score_synthesis(result, [source()], ["access", "retention"])
    assert score.citation_precision == 0
    assert score.coverage == 0.5
    assert score.violations
    assert not approval_ready(score)


def test_unverified_control_results_do_not_pass_the_stricter_quality_gate():
    result = Synthesis(
        summary="Four controls are evidenced; SOC 2 control results are not provided.",
        claims=[supported(f"C{index}") for index in range(4)]
        + [
            Claim(
                id="C5",
                topic="access",
                text="SOC 2 control test results are not supplied",
                status="unverified",
            )
        ],
    )
    score = score_synthesis(result, [source()], ["access"])
    assert score.citation_precision == score.coverage == 1
    assert score.claim_support == 0.8
    assert score.quality == 0.94
    assert score.quality < 0.95


def test_gaps_trigger_replan_and_changed_sources_trigger_refresh():
    score = score_synthesis(
        Synthesis(summary="Missing coverage"), [source()], ["access"]
    )
    signals = loop_signals(score, source_changed=True)
    assert signals["replan_required"] is True
    assert signals["refresh_required"] is True
    assert signals["objective_complete"] is False
    assert signals["validation_passed"] is False


def test_only_published_evidence_completes_the_objective():
    score = score_synthesis(
        Synthesis(summary="ok", claims=[supported()]), [source()], ["access"]
    )
    pending = loop_signals(score)
    assert approval_ready(score)
    assert pending["objective_complete"] is False
    assert pending["objective_progress"] == 1.0
    assert pending["replan_required"] is False
    assert any("publish_evidence" in item for item in pending["feedback"])
    published = loop_signals(score, published=True, source_changed=True)
    assert published["objective_complete"] is True
    assert published["validation_passed"] is True
    assert published["objective_progress"] == 1.0
    assert published["refresh_required"] is False


def test_full_coverage_with_violations_replans_instead_of_completing():
    # Full coverage reports progress 1.0, but the explicit incomplete signal
    # keeps the loop going while a citation is invalid.
    result = Synthesis(
        summary="Covered but miscited.",
        claims=[supported(), supported("C2", quote="an invented quote")],
    )
    observer = EvidenceLoopObserver([source()], ["access"])
    observation = observer.observe(
        iteration=1, result=AIMessage(content=result.model_dump_json())
    )
    assert observation.objective_progress == 1.0
    assert observation.objective_complete is False
    policy = loop_policy()
    state = LoopState(
        loop_id="loop",
        objective="verify",
        iteration=1,
        budget=LoopBudget.from_policy(policy).model_copy(update={"iterations": 1}),
    )
    evaluation = DefaultLoopEvaluator().evaluate(state, observation, policy)
    decision = DefaultLoopController().decide(
        state, observation, evaluation, policy, state.budget
    )
    assert evaluation.objective_complete is False
    assert decision.action is LoopAction.REPLAN


def test_observer_exposes_publication_aware_loop_signals():
    result = Synthesis(summary="Verified.", claims=[supported(quote="MFA")])
    observer = EvidenceLoopObserver([source()], ["access"])
    observed = observer.observe(
        iteration=1,
        result=AIMessage(content=f"```json\n{json.dumps(result.model_dump())}\n```"),
    )
    assert observed.validation_passed is False
    assert observed.objective_complete is False
    assert observed.metadata["accuracy"]["quality"] == 1
    observer.mark_published(result)
    published = observer.observe(iteration=2, result=AIMessage(content="Published."))
    assert published.objective_complete is True
    assert published.validation_passed is True


def test_observer_streams_deterministic_accuracy_progress():
    result = Synthesis(summary="Partially verified.", claims=[supported(quote="MFA")])
    events = []
    observer = EvidenceLoopObserver(
        [source()],
        ["access", "retention"],
        events=lambda event_type, payload: events.append((event_type, payload)),
    )
    observer.observe(iteration=2, result=result)
    event_type, payload = events[0]
    assert event_type == "iteration"
    assert payload["iteration"] == 2
    assert payload["progress"] == 0.5
    assert payload["quality"] == 0.875
    assert payload["citation_precision"] == 1
    assert payload["gaps"] == ["retention"]
    assert payload["violations"] == []
    assert payload["approval_ready"] is False
    assert payload["message"] == "Validation found gaps that require another iteration."


def test_failed_iterations_keep_the_original_error_type():
    class ApprovalRejected(Exception):
        pass

    observer = EvidenceLoopObserver([source()], ["access"])
    observed = observer.observe(iteration=1, error=ApprovalRejected("rejected"))
    assert observed.success is False
    assert observed.error_type == "ApprovalRejected"


def test_extract_synthesis_parses_ai_message_content_blocks():
    payload = {
        "summary": "Verified.",
        "claims": [supported(quote="MFA").model_dump()],
        "covered_topics": ["access"],
    }
    message = AIMessage(content=[{"type": "text", "text": json.dumps(payload)}])
    assert extract_synthesis(message) == Synthesis.model_validate(payload)
    assert extract_synthesis(AIMessage(content="Published.")) is None


def test_feedback_state_carries_prior_gaps_and_violations():
    observer = EvidenceLoopObserver([source()], ["access", "retention"])
    observer.observe(
        iteration=1, result=AIMessage(content='{"summary":"incomplete","claims":[]}')
    )
    feedback = observer.feedback_payload()
    assert feedback["iteration"] == 1
    assert feedback["gaps"] == ["access", "retention"]
    assert feedback["approval_ready"] is False
    assert feedback["sources"][0]["revision"] == "1"


def test_submission_gate_accepts_only_one_ready_request_per_iteration():
    observer = EvidenceLoopObserver(SOURCES, ["encryption", "incident response"])
    assert observer.review_submission(synthesis()).ready is True
    second = observer.review_submission(synthesis())
    assert second.ready is False
    assert "Only one publication request" in second.message
    observer.observe(iteration=1, result=AIMessage(content=json.dumps(synthesis())))
    invalid = observer.review_submission(synthesis(invalid_quote=True))
    assert invalid.ready is False
    report = json.loads(invalid.message)
    assert report["status"] == "not_submitted"
    assert report["violations"]


def test_submission_gate_rejects_schema_errors_and_stale_sources():
    observer = EvidenceLoopObserver(SOURCES, ["encryption"])
    malformed = observer.review_submission({"claims": "not a list"})
    assert malformed.ready is False
    assert "schema" in malformed.message
    stale = EvidenceLoopObserver(
        SOURCES, ["encryption"], revision_provider=FixedRevisions(changed=True)
    )
    review = stale.review_submission(synthesis(["encryption"]))
    assert review.ready is False
    assert "Newer evidence revisions" in review.message


def test_decided_publication_requests_pass_to_the_policy():
    observer = EvidenceLoopObserver(SOURCES, ["encryption"])
    observer.approval_decided = True
    assert observer.review_submission(synthesis(["encryption"])).ready is True


def test_restore_and_timeline_entries_survive_restarts():
    observer = EvidenceLoopObserver(SOURCES, ["encryption", "incident response"])
    observer.restore(
        Iteration(
            iteration=2,
            action="replan",
            progress=0.5,
            quality=0.6,
            objective_complete=False,
            gaps=["incident response"],
            violations=["C1: invalid citation VENDOR-SEC-001"],
            citation_precision=0.5,
            coverage=0.5,
        )
    )
    feedback = observer.feedback_payload()
    assert feedback["iteration"] == 2
    assert feedback["gaps"] == ["incident response"]
    observer.observe(iteration=3, result=AIMessage(content=json.dumps(synthesis())))
    entry = observer.timeline_entry(3, "continue", "validation passed")
    assert entry.quality == 1
    assert entry.coverage == 1
    assert entry.citations == 2
    assert entry.reason == "validation passed"
    unknown = observer.timeline_entry(9, "stop", "no observation")
    assert unknown.quality == 0


def test_loop_policy_is_bounded_and_never_completes_on_convergence():
    policy = loop_policy()
    assert policy.max_iterations == 5
    assert policy.allow_replan
    assert policy.allow_refresh
    assert policy.allow_retry
    assert policy.allow_human_intervention is False
    assert policy.complete_on_convergence is False
    assert loop_policy("one_shot") == LoopPolicy(max_duration=300, max_tool_calls=30)
