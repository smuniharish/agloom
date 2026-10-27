"""API contracts and the structured audit report."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

Verdict = Literal["pass", "fail", "needs_review"]


class AuditRequest(BaseModel):
    """A URL to audit, optionally limited to some policies."""

    url: HttpUrl
    policy_ids: list[str] = Field(default_factory=list, max_length=20)


class Evidence(BaseModel):
    """One retrieved policy."""

    policy_id: str
    title: str
    excerpt: str
    similarity: float = Field(ge=0, le=1)


class Finding(BaseModel):
    """The assessment of one policy requirement."""

    policy_id: str
    requirement: str
    status: Verdict
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)


class AuditReport(BaseModel):
    """The model's structured audit."""

    verdict: Verdict = Field(
        description="pass only when every applicable requirement is evidenced."
    )
    summary: str = Field(min_length=1, max_length=3000)
    findings: list[Finding] = Field(min_length=1, max_length=40)
    limitations: list[str] = Field(default_factory=list)

    def grounded(self, policy_ids: set[str]) -> bool:
        """Return whether every finding references retrieved policies only."""
        return all(
            finding.policy_id in policy_ids and set(finding.evidence_ids) <= policy_ids
            for finding in self.findings
        )


RecordStatus = Literal[
    "recorded", "awaiting_approval", "rejected", "expired", "outcome_unknown"
]


class ComplianceRecord(BaseModel):
    """The governed publication of the audit to the compliance register."""

    status: RecordStatus
    action_id: str | None = None
    policy_rule: str | None = None
    decided_by: str | None = None
    recorded_at: datetime | None = None


class Audit(BaseModel):
    """A completed audit and the state of its compliance record."""

    audit_id: UUID
    url: str
    report: AuditReport
    evidence: list[Evidence]
    mcp_checks: dict[str, str]
    record: ComplianceRecord
    created_at: datetime


class ApprovalRequest(BaseModel):
    """A compliance officer's decision on a pending passing audit."""

    approve: bool
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=2000)


class FeedbackRequest(BaseModel):
    """Reviewer feedback on an audit."""

    rating: Literal["up", "down"]
    correction: str | None = Field(default=None, max_length=2000)
