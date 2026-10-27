from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

RunStatus = Literal[
    "queued",
    "running",
    "awaiting_approval",
    "awaiting_reconciliation",
    "completed",
    "failed",
    "cancelled",
    "rejected",
]
ACTIVE_RUN_STATUSES: frozenset[str] = frozenset(
    {"queued", "running", "awaiting_approval", "awaiting_reconciliation"}
)
WAITING_RUN_STATUSES: frozenset[str] = frozenset(
    {"awaiting_approval", "awaiting_reconciliation"}
)
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled", "rejected"}
)


class AcceptanceCriterion(BaseModel):
    id: str = Field(pattern=r"^AC-\d+$")
    description: str = Field(min_length=3, max_length=500)
    kind: Literal["file", "content", "command", "browser"]
    target: str = Field(min_length=1, max_length=500)
    expected: str | None = Field(default=None, max_length=500)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    objective: str = Field(min_length=10, max_length=4000)
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1, max_length=30)


class TaskCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    objective: str = Field(min_length=10, max_length=4000)


class Project(BaseModel):
    id: UUID
    name: str
    objective: str
    acceptance_criteria: list[AcceptanceCriterion]
    created_at: datetime


class Task(BaseModel):
    id: UUID
    project_id: UUID
    title: str
    objective: str
    status: RunStatus
    created_at: datetime


class Evidence(BaseModel):
    id: UUID
    run_id: UUID
    iteration: int
    criterion_id: str
    source: Literal["filesystem", "command", "playwright", "agent"]
    passed: bool
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RunProof(BaseModel):
    iterations: int = 0
    acceptance_criteria: int = 0
    passed_criteria: int = 0
    validation_pass_rate: float = Field(default=0, ge=0, le=1)
    tool_calls: int = 0
    model_calls: int = 0
    elapsed_seconds: float = Field(default=0, ge=0)
    termination_reason: str | None = None


class Approval(BaseModel):
    """A consequential tool call that waits for, or was decided by, a person.

    `kind="approval"`: a reviewer approves or rejects the call before it runs.
    `kind="reconciliation"`: the call raised, so agentbarrier cannot tell whether
    it took effect (`status="unknown"`); an operator records whether it did.
    """

    action_id: str
    tool: str
    kind: Literal["approval", "reconciliation"] = "approval"
    arguments: dict[str, Any] = Field(default_factory=dict)
    policy_rule: str = ""
    error: str | None = None
    expires_at: datetime | None = None
    status: Literal[
        "pending",
        "approved",
        "rejected",
        "denied",
        "expired",
        "executing",
        "succeeded",
        "unknown",
    ] = "pending"
    decided_by: str | None = None
    decision_reason: str | None = None


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str | None = Field(default=None, max_length=1000)


class ReconciliationDecision(BaseModel):
    """An operator's finding about a call whose outcome is unknown.

    `committed`: the call took effect; `result` (any JSON value) is replayed to
    the agent as the call's result. `not_committed`: it did not; the call runs
    again once the policy allows it (a release needs a new approval).
    """

    outcome: Literal["committed", "not_committed"]
    reason: str = Field(min_length=3, max_length=1000)
    result: Any = None


class LoopRun(BaseModel):
    id: UUID
    task_id: UUID
    status: RunStatus
    proof: RunProof
    result: dict[str, Any] = Field(default_factory=dict)
    approvals: list[Approval] = Field(default_factory=list)
    created_at: datetime
    finished_at: datetime | None = None


class Artifact(BaseModel):
    id: UUID
    project_id: UUID
    path: str
    kind: str
    sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
