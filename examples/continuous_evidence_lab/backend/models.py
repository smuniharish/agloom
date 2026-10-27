from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

RunStatus = Literal[
    "running",
    "awaiting_approval",
    "approved",
    "rejected",
    "unvalidated",
    "failed",
    "cancelled",
]
ExecutionStatus = Literal[
    "queued", "running", "paused", "completed", "failed", "cancelled"
]
ObjectiveState = Literal["incomplete", "awaiting_approval", "validated", "failed"]
TERMINAL_EXECUTION: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


class Source(BaseModel):
    id: str
    title: str
    uri: str
    revision: str
    body: str
    retrieved_at: datetime | None = None
    similarity: float = Field(default=1, ge=0, le=1)


class Citation(BaseModel):
    source_id: str = Field(description="ID of a supplied source.")
    quote: str = Field(
        min_length=1,
        max_length=600,
        description="A verbatim quote from that source.",
    )


class Claim(BaseModel):
    id: str
    topic: str = Field(description="The required topic the claim covers.")
    text: str
    status: Literal["supported", "contradicted", "unverified"]
    citations: list[Citation] = Field(default_factory=list)


class Synthesis(BaseModel):
    """A cited evidence synthesis covering the required topics."""

    summary: str = Field(min_length=1, max_length=4000)
    claims: list[Claim] = Field(default_factory=list)
    covered_topics: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Accuracy(BaseModel):
    citation_precision: float = Field(ge=0, le=1)
    claim_support: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    valid_citations: int = 0
    total_citations: int = 0
    supported_claims: int = 0
    total_claims: int = 0
    gaps: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class Iteration(BaseModel):
    iteration: int
    action: str
    progress: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    objective_complete: bool
    gaps: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    citations: int = 0
    citation_precision: float = Field(default=0, ge=0, le=1)
    coverage: float = Field(default=0, ge=0, le=1)
    reason: str = ""


class RunRequest(BaseModel):
    objective: str = Field(min_length=8, max_length=1000)
    topics: list[str] = Field(min_length=1, max_length=12)
    target_url: HttpUrl | None = None
    mode: Literal["loop", "one_shot"] = "loop"


class Approval(BaseModel):
    """The reviewer decision gating publication of one run's evidence."""

    action_id: str
    tool: str
    policy_rule: str
    expires_at: datetime | None = None
    decision: Literal["approve", "reject"] | None = None
    decided_by: str | None = None
    decided_at: datetime | None = None
    note: str | None = None


class RunView(BaseModel):
    id: UUID
    objective: str
    topics: list[str]
    mode: Literal["loop", "one_shot"] = "loop"
    status: RunStatus
    execution_status: ExecutionStatus = "queued"
    objective_state: ObjectiveState
    termination: str | None = None
    error: str | None = None
    synthesis: Synthesis | None = None
    accuracy: Accuracy
    timeline: list[Iteration] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    approval: Approval | None = None
    published_at: datetime | None = None
    created_at: datetime


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=1000)


def empty_accuracy(topics: list[str]) -> Accuracy:
    """Return the accuracy of a run that has produced no synthesis yet."""
    return Accuracy(
        citation_precision=0, claim_support=0, coverage=0, quality=0, gaps=topics
    )
