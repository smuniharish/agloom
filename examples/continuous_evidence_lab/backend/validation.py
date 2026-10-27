"""Deterministic evidence validation and the Loop Engineering observer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from pydantic import ValidationError
from xstructured import StructuredParser, XStructuredError

from agloom import DefaultLoopObserver, LoopObservation

from .models import Accuracy, Iteration, Source, Synthesis, empty_accuracy

QUALITY_GATE = 0.8

EventSink = Callable[[str, dict[str, Any]], None]


class RevisionProvider(Protocol):
    def has_newer_revisions(self, current: list[Source]) -> bool: ...


def score_synthesis(
    synthesis: Synthesis, sources: list[Source], required_topics: list[str]
) -> Accuracy:
    source_by_id = {source.id: source for source in sources}
    valid = total = supported = 0
    violations: list[str] = []
    covered: set[str] = set()
    for claim in synthesis.claims:
        if claim.status == "supported":
            supported += 1
        if claim.topic in required_topics and claim.status != "unverified":
            covered.add(claim.topic)
        if claim.status in {"supported", "contradicted"} and not claim.citations:
            violations.append(f"{claim.id}: factual status requires a citation")
        for citation in claim.citations:
            total += 1
            source = source_by_id.get(citation.source_id)
            if source and citation.quote.casefold() in source.body.casefold():
                valid += 1
            else:
                violations.append(f"{claim.id}: invalid citation {citation.source_id}")
    total_claims = len(synthesis.claims)
    precision = valid / total if total else 0.0
    support = supported / total_claims if total_claims else 0.0
    coverage = len(covered) / len(required_topics) if required_topics else 1.0
    quality = round(0.45 * precision + 0.30 * support + 0.25 * coverage, 4)
    gaps = [topic for topic in required_topics if topic not in covered]
    return Accuracy(
        citation_precision=precision,
        claim_support=support,
        coverage=coverage,
        quality=quality,
        valid_citations=valid,
        total_citations=total,
        supported_claims=supported,
        total_claims=total_claims,
        gaps=gaps,
        violations=violations,
    )


def approval_ready(accuracy: Accuracy) -> bool:
    """Whether a synthesis passes the deterministic gate for reviewer approval."""
    return (
        accuracy.citation_precision == 1.0
        and accuracy.coverage == 1.0
        and not accuracy.violations
        and accuracy.quality >= QUALITY_GATE
    )


def loop_signals(
    accuracy: Accuracy, *, published: bool = False, source_changed: bool = False
) -> dict[str, Any]:
    """Map deterministic accuracy to Loop Engineering signals.

    Progress is topic coverage, but the objective is complete only when
    reviewer-approved evidence has been published: the explicit
    `objective_complete` signal takes precedence over full coverage. Gaps and
    violations request a replan, and changed source revisions request a
    refresh first.
    """
    ready = approval_ready(accuracy)
    feedback = [*accuracy.gaps, *accuracy.violations]
    if ready and not published:
        feedback.append(
            "validation passed: submit this synthesis with publish_evidence "
            "for reviewer approval"
        )
    return {
        "objective_progress": 1.0 if published else accuracy.coverage,
        "objective_complete": published,
        "validation_passed": published,
        "quality": accuracy.quality,
        "refresh_required": source_changed and not published,
        "replan_required": not published and bool(accuracy.gaps or accuracy.violations),
        "feedback": tuple(feedback),
    }


@dataclass(frozen=True)
class SubmissionReview:
    """The deterministic verdict on one publication request."""

    ready: bool
    message: str
    accuracy: Accuracy | None = None


class EvidenceLoopObserver:
    """Derives every loop signal from deterministic citation and coverage checks.

    The same validator gates publication requests before they reach the
    approval policy and serves the `get_validation_feedback` tool, so the
    model can never declare its own output accurate.
    """

    def __init__(
        self,
        sources: list[Source],
        topics: list[str],
        *,
        revision_provider: RevisionProvider | None = None,
        events: EventSink | None = None,
    ) -> None:
        self.sources = list(sources)
        self.topics = list(topics)
        self.revision_provider = revision_provider
        self.events = events
        self.latest_accuracy = empty_accuracy(self.topics)
        self.latest_iteration = 0
        self.latest_synthesis: Synthesis | None = None
        self.published: Synthesis | None = None
        self.approval_decided = False
        self._observed: dict[int, tuple[Accuracy, float, bool]] = {}
        self._submission: Synthesis | None = None
        self._submissions = 0
        self._default = DefaultLoopObserver()

    def restore(
        self, latest: Iteration | None, *, published: Synthesis | None = None
    ) -> None:
        """Reload validation state persisted before a restart."""
        if latest is not None:
            self.latest_iteration = latest.iteration
            self.latest_accuracy = Accuracy(
                citation_precision=latest.citation_precision,
                claim_support=0,
                coverage=latest.coverage,
                quality=latest.quality,
                valid_citations=latest.citations,
                gaps=latest.gaps,
                violations=latest.violations,
            )
        if published is not None:
            self.published = published
            self.latest_synthesis = published

    def feedback_payload(self) -> dict[str, Any]:
        """State read by the model before every synthesis attempt."""
        return {
            "iteration": self.latest_iteration,
            "gaps": self.latest_accuracy.gaps,
            "violations": self.latest_accuracy.violations,
            "approval_ready": approval_ready(self.latest_accuracy),
            "published": self.published is not None,
            "instruction": (
                "Resolve every listed gap and violation. Cite only verbatim quotes "
                "from the current sources below, then submit the synthesis once "
                "with publish_evidence."
            ),
            "sources": [source.model_dump(mode="json") for source in self.sources],
        }

    def replace_source(self, source: Source) -> None:
        by_id = {item.id: item for item in self.sources}
        by_id[source.id] = source
        self.sources[:] = list(by_id.values())

    def delete_source(self, source_id: str) -> None:
        self.sources[:] = [item for item in self.sources if item.id != source_id]

    def review_submission(self, arguments: Mapping[str, Any]) -> SubmissionReview:
        """Decide whether a publication request may reach the approval policy."""
        if self.approval_decided or self.published is not None:
            return SubmissionReview(True, "a reviewer already decided this request")
        self._submissions += 1
        if self._submissions > 1:
            return SubmissionReview(
                False,
                "Only one publication request is accepted per iteration. Answer "
                "with your synthesis; the loop validates it and continues.",
            )
        try:
            synthesis = Synthesis.model_validate(arguments)
        except ValidationError as error:
            return SubmissionReview(
                False,
                "The publication request does not match the synthesis schema "
                f"({error.error_count()} error(s)). Answer with a corrected "
                "synthesis.",
            )
        self._submission = synthesis
        if self._source_changed():
            return SubmissionReview(
                False,
                "Newer evidence revisions are available. Answer with your "
                "synthesis; the loop refreshes the sources before the next attempt.",
            )
        accuracy = score_synthesis(synthesis, self.sources, self.topics)
        if approval_ready(accuracy):
            return SubmissionReview(True, "approval-ready", accuracy)
        return SubmissionReview(
            False,
            json.dumps(
                {
                    "status": "not_submitted",
                    "reason": "deterministic validation failed",
                    "gaps": accuracy.gaps,
                    "violations": accuracy.violations,
                    "citation_precision": accuracy.citation_precision,
                    "coverage": accuracy.coverage,
                    "quality": accuracy.quality,
                    "instruction": "Answer with your synthesis; the next iteration "
                    "receives this feedback.",
                },
                sort_keys=True,
            ),
            accuracy,
        )

    def mark_published(self, synthesis: Synthesis) -> None:
        self.published = synthesis
        self.latest_synthesis = synthesis

    def timeline_entry(self, iteration: int, action: str, reason: str) -> Iteration:
        """Combine the validation of one iteration with the loop's decision."""
        accuracy, progress, complete = self._observed.get(
            iteration, (empty_accuracy(self.topics), 0.0, False)
        )
        return Iteration(
            iteration=iteration,
            action=action,
            progress=progress,
            quality=accuracy.quality,
            objective_complete=complete,
            gaps=accuracy.gaps,
            violations=accuracy.violations,
            citations=accuracy.valid_citations,
            citation_precision=accuracy.citation_precision,
            coverage=accuracy.coverage,
            reason=reason,
        )

    def _source_changed(self) -> bool:
        return bool(
            self.revision_provider
            and self.revision_provider.has_newer_revisions(self.sources)
        )

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.events is not None:
            self.events(event_type, payload)

    def observe(
        self,
        *,
        iteration: int,
        result: Any = None,
        error: BaseException | None = None,
        topology: Any = None,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LoopObservation:
        base = self._default.observe(
            iteration=iteration,
            result=result,
            error=error,
            topology=topology,
            duration=duration,
            metadata=metadata,
        )
        self._submissions = 0
        submission, self._submission = self._submission, None
        if error is not None:
            self._emit(
                "iteration",
                {
                    "iteration": iteration,
                    "message": f"Iteration ended with {base.error_type}.",
                },
            )
            return base
        synthesis = self.published or extract_synthesis(result) or submission
        accuracy = (
            score_synthesis(synthesis, self.sources, self.topics)
            if synthesis is not None
            else empty_accuracy(self.topics).model_copy(
                update={"violations": ["structured synthesis unavailable"]}
            )
        )
        published = self.published is not None
        self.latest_accuracy = accuracy
        self.latest_iteration = iteration
        if synthesis is not None:
            self.latest_synthesis = synthesis
        signals = loop_signals(
            accuracy, published=published, source_changed=self._source_changed()
        )
        self._observed[iteration] = (
            accuracy,
            signals["objective_progress"],
            signals["objective_complete"],
        )
        self._emit(
            "iteration",
            {
                "iteration": iteration,
                "progress": signals["objective_progress"],
                "quality": accuracy.quality,
                "citation_precision": accuracy.citation_precision,
                "coverage": accuracy.coverage,
                "gaps": accuracy.gaps,
                "violations": accuracy.violations,
                "approval_ready": approval_ready(accuracy),
                "message": _iteration_message(accuracy, published),
            },
        )
        fingerprint = (
            hashlib.sha256(
                json.dumps(synthesis.model_dump(mode="json"), sort_keys=True).encode()
            ).hexdigest()
            if synthesis is not None
            else base.result_fingerprint
        )
        return base.model_copy(
            update={
                **signals,
                "result_fingerprint": fingerprint,
                "metadata": {
                    **base.metadata,
                    "accuracy": accuracy.model_dump(mode="json"),
                },
            }
        )


def _iteration_message(accuracy: Accuracy, published: bool) -> str:
    if published:
        return "Reviewer-approved evidence was published."
    if approval_ready(accuracy):
        return "Deterministic validation passed; publication needs reviewer approval."
    return "Validation found gaps that require another iteration."


def extract_synthesis(result: Any) -> Synthesis | None:
    """Return the synthesis carried by an answer, a mapping, or a parse result."""
    candidates = [result, getattr(result, "structured", None)]
    if isinstance(result, Mapping):
        candidates.extend([result.get("structured"), result.get("structured_output")])
    for candidate in candidates:
        if isinstance(candidate, Synthesis):
            return candidate
        if isinstance(candidate, Mapping):
            try:
                return Synthesis.model_validate(candidate)
            except ValidationError:
                continue
        text = _text_content(candidate)
        if text:
            try:
                parsed = StructuredParser(Synthesis).parse(text).value
            except (XStructuredError, ValidationError, ValueError):
                continue
            if isinstance(parsed, Synthesis):
                return parsed
    return None


def _text_content(value: Any) -> str | None:
    content = value.content if isinstance(value, BaseMessage) else value
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) or None
    return None
