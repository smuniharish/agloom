"""API contracts and the structured investigation report."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

CITATION = re.compile(r"\[((?:ADR|RUNBOOK|RELEASE)-\d+)\]")


class InvestigationRequest(BaseModel):
    """A release-readiness question."""

    question: str = Field(min_length=3, max_length=2000)


class EvidenceItem(BaseModel):
    """One factual claim and the evidence document that supports it."""

    claim: str = Field(min_length=1, max_length=1000)
    citation: str = Field(
        description="Exactly one bracketed evidence ID, for example [ADR-101]."
    )


class InvestigationReport(BaseModel):
    """The model's structured release-readiness assessment."""

    decision: Literal["go", "no_go"] = Field(
        description="go only when every release requirement is evidenced."
    )
    risk: Literal["low", "medium", "high", "critical"]
    summary: str = Field(min_length=1, max_length=3000)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=20)
    recommendation: str = Field(min_length=1, max_length=2000)

    def citations(self) -> list[str]:
        """Return the evidence IDs cited by the report, in order."""
        return [
            match.group(1)
            for item in self.evidence
            if (match := CITATION.fullmatch(item.citation)) is not None
        ]

    def grounded(self, allowed: set[str]) -> bool:
        """Return whether every citation is exactly one allowed evidence ID."""
        cited = self.citations()
        return len(cited) == len(self.evidence) and set(cited) <= allowed


DecisionStatus = Literal[
    "recorded", "awaiting_approval", "rejected", "expired", "outcome_unknown"
]


class ReleaseDecision(BaseModel):
    """The governed publication of the decision to the release register."""

    status: DecisionStatus
    action_id: str | None = None
    policy_rule: str | None = None
    decided_by: str | None = None
    recorded_at: datetime | None = None


class Investigation(BaseModel):
    """A completed investigation and the state of its release decision."""

    investigation_id: UUID
    question: str
    report: InvestigationReport
    mcp_checks: dict[str, str]
    release_decision: ReleaseDecision
    created_at: datetime


class ApprovalRequest(BaseModel):
    """A release manager's decision on a pending go decision."""

    approve: bool
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=2000)


class FeedbackRequest(BaseModel):
    """Reviewer feedback on an investigation."""

    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)
