"""Strict API and model output contracts."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class AuditRequest(BaseModel):
    url: HttpUrl
    policy_ids: list[str] = Field(default_factory=list, max_length=20)


class Evidence(BaseModel):
    policy_id: str
    title: str
    excerpt: str
    similarity: float = Field(ge=0, le=1)


class Finding(BaseModel):
    policy_id: str
    requirement: str
    status: Literal["pass", "fail", "needs_review"]
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)


class AuditReport(BaseModel):
    verdict: Literal["pass", "fail", "needs_review"]
    summary: str = Field(min_length=1, max_length=3000)
    findings: list[Finding] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class AuditResponse(BaseModel):
    audit_id: UUID
    url: str
    report: AuditReport
    evidence: list[Evidence]
    provenance_id: str


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    correction: str | None = Field(default=None, max_length=2000)
