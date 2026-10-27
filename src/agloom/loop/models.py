"""Immutable Loop Engineering contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agloom.models import TopologyName


class LoopPhase(StrEnum):
    """Lifecycle phase of one loop."""

    CREATED = "created"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    WAITING = "waiting"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    FAILED = "failed"


class LoopAction(StrEnum):
    """What the loop does after an iteration."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    RETRY = "retry"
    REPLAN = "replan"
    REFRESH = "refresh"
    CHANGE_TOPOLOGY = "change_topology"
    WAIT = "wait"
    HUMAN_INTERVENTION = "human_intervention"
    STOP = "stop"
    FAIL = "fail"


class ProgressStatus(StrEnum):
    """Progress of the latest iteration relative to the previous one."""

    UNKNOWN = "unknown"
    PROGRESS = "progress"
    NO_PROGRESS = "no_progress"
    REGRESSION = "regression"
    STAGNATION = "stagnation"
    CONVERGENCE = "convergence"


class ObjectiveStatus(StrEnum):
    """Status of the loop objective."""

    UNKNOWN = "unknown"
    IN_PROGRESS = "in_progress"
    AT_RISK = "at_risk"
    SATISFIED = "satisfied"


class LoopTerminationReason(StrEnum):
    """Why a loop stopped iterating."""

    OBJECTIVE_SATISFIED = "objective_satisfied"
    VALIDATED = "validated"
    CONVERGED = "converged"
    ONE_SHOT_COMPLETE = "one_shot_complete"
    EXPLICIT_STOP = "explicit_stop"
    ITERATION_LIMIT = "iteration_limit"
    TIMEOUT = "timeout"
    TOOL_CALL_LIMIT = "tool_call_limit"
    RETRY_LIMIT = "retry_limit"
    REPLAN_LIMIT = "replan_limit"
    STAGNATION = "stagnation"
    OSCILLATION = "oscillation"
    POLICY_VIOLATION = "policy_violation"
    HUMAN_INTERVENTION = "human_intervention"
    BEHAVIOR_INTERVENTION = "behavior_intervention"
    EXECUTION_FAILED = "execution_failed"


class LoopPolicy(BaseModel):
    """Bounds and permitted adaptations for one loop.

    The default is a one-shot loop: one iteration, no adaptation, and no extra
    evaluation call.

    Attributes:
        max_iterations: Maximum number of iterations.
        max_duration: Maximum active execution time in seconds, checked when
            each iteration ends; an iteration in progress is not interrupted,
            so set timeouts on the chat model and tools for per-call deadlines.
            Time spent waiting for a human or an approval is not counted.
        max_tool_calls: Maximum tool calls across all iterations. See
            `LoopBudget` for how iterations resumed after a pause are counted.
        max_retries: Maximum retries, after failed iterations or when an
            evaluator recommends retrying.
        max_replans: Maximum replans.
        history_limit: Number of history entries kept in the loop state.
        stagnation_threshold: Consecutive iterations without progress that
            count as stagnation.
        oscillation_window: Number of recent decisions inspected for
            alternating actions or topologies.
        progress_threshold: Minimum progress change that counts as progress.
        quality_threshold: Minimum quality for convergence.
        max_context_chars: Maximum characters of the previous result carried
            into the next iteration.
        allow_retry: Whether failed iterations may be retried.
        allow_replan: Whether the loop may ask for a different approach.
        allow_refresh: Whether the loop may refresh sources before continuing.
        allow_topology_change: Whether the loop may switch topology.
        allow_human_intervention: Whether the loop may wait for a human.
        complete_on_convergence: Whether converged results complete the loop.
    """

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=1, ge=1, le=1_000)
    max_duration: float = Field(default=300.0, gt=0, le=86_400)
    max_tool_calls: int = Field(default=1_000, ge=0, le=1_000_000)
    max_retries: int = Field(default=0, ge=0, le=100)
    max_replans: int = Field(default=0, ge=0, le=100)
    history_limit: int = Field(default=20, ge=1, le=1_000)
    stagnation_threshold: int = Field(default=3, ge=2, le=100)
    oscillation_window: int = Field(default=4, ge=4, le=100)
    progress_threshold: float = Field(default=0.01, gt=0, le=1)
    quality_threshold: float = Field(default=0.8, ge=0, le=1)
    max_context_chars: int = Field(default=4_000, ge=0, le=200_000)
    allow_retry: bool = False
    allow_replan: bool = False
    allow_refresh: bool = False
    allow_topology_change: bool = False
    allow_human_intervention: bool = False
    complete_on_convergence: bool = True

    @field_validator("oscillation_window")
    @classmethod
    def _require_even_window(cls, value: int) -> int:
        if value % 2:
            raise ValueError("oscillation_window must be even")
        return value


class LoopBudget(BaseModel):
    """Resource usage of one loop against its limits.

    Usage is recorded when an iteration finishes. When an iteration pauses (for
    an approval or a breakpoint) or crashes and is later resumed, the resumed
    run counts its own tool calls and active time and continues with the
    tool-call budget that remained when the iteration started; the tool calls
    and time from before the pause are not added.
    """

    model_config = ConfigDict(frozen=True)

    max_iterations: int
    max_duration: float
    max_tool_calls: int
    max_retries: int
    max_replans: int
    iterations: int = Field(default=0, ge=0)
    elapsed: float = Field(default=0.0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    replans: int = Field(default=0, ge=0)

    @classmethod
    def from_policy(cls, policy: LoopPolicy) -> LoopBudget:
        """Return an unused budget for a policy."""
        return cls(
            max_iterations=policy.max_iterations,
            max_duration=policy.max_duration,
            max_tool_calls=policy.max_tool_calls,
            max_retries=policy.max_retries,
            max_replans=policy.max_replans,
        )


class LoopObservation(BaseModel):
    """Normalized signals from one iteration.

    Signals come from the result itself (a `loop` mapping in a dictionary
    result or in an `AIMessage`'s `additional_kwargs`), from the execution
    (errors, tool calls, topology), and from evaluators.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    iteration: int = Field(ge=1)
    success: bool
    result: Any = Field(default=None, repr=False, exclude=True)
    result_fingerprint: str
    error_type: str | None = None
    error_message: str | None = None
    topology: TopologyName | None = None
    duration: float = Field(default=0.0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    objective_progress: float | None = Field(default=None, ge=0, le=1)
    objective_complete: bool | None = None
    quality: float | None = Field(default=None, ge=0, le=1)
    validation_passed: bool | None = None
    retryable: bool = False
    refresh_required: bool = False
    replan_required: bool = False
    human_intervention_required: bool = False
    requested_topology: TopologyName | None = None
    feedback: tuple[str, ...] = ()
    behavior_signals: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_error(self) -> LoopObservation:
        if self.success and self.error_type is not None:
            raise ValueError("successful observations cannot contain an error")
        if not self.success and not self.error_type:
            raise ValueError("failed observations require error_type")
        return self


class LoopEvaluation(BaseModel):
    """Objective-relative interpretation of one observation."""

    model_config = ConfigDict(frozen=True)

    progress_status: ProgressStatus = ProgressStatus.UNKNOWN
    objective_status: ObjectiveStatus = ObjectiveStatus.UNKNOWN
    progress: bool = False
    objective_progress: float | None = Field(default=None, ge=0, le=1)
    objective_complete: bool = False
    stagnation: bool = False
    regression: bool = False
    convergence: bool = False
    quality: float | None = Field(default=None, ge=0, le=1)
    consecutive_no_progress: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    recommended_action: LoopAction | None = None
    feedback: tuple[str, ...] = ()
    reason: str


class LoopDecision(BaseModel):
    """A validated controller decision."""

    model_config = ConfigDict(frozen=True)

    action: LoopAction
    reason: str = Field(min_length=1)
    next_topology: TopologyName | None = None
    termination_reason: LoopTerminationReason | None = None

    @model_validator(mode="after")
    def _validate_topology(self) -> LoopDecision:
        if (self.action is LoopAction.CHANGE_TOPOLOGY) != (
            self.next_topology is not None
        ):
            raise ValueError("next_topology is required exactly for CHANGE_TOPOLOGY")
        return self


class LoopHistoryEntry(BaseModel):
    """One observe, evaluate, and decide record. Results are not retained."""

    model_config = ConfigDict(frozen=True)

    observation: LoopObservation
    evaluation: LoopEvaluation
    decision: LoopDecision


class LoopState(BaseModel):
    """The state of one loop."""

    model_config = ConfigDict(frozen=True)

    loop_id: str
    objective: str
    phase: LoopPhase = LoopPhase.CREATED
    iteration: int = Field(default=0, ge=0)
    objective_status: ObjectiveStatus = ObjectiveStatus.UNKNOWN
    progress_status: ProgressStatus = ProgressStatus.UNKNOWN
    progress: float | None = Field(default=None, ge=0, le=1)
    budget: LoopBudget
    history: tuple[LoopHistoryEntry, ...] = ()
    active_topology: TopologyName | None = None
    evaluation: LoopEvaluation | None = None
    decision: LoopDecision | None = None
    termination_reason: LoopTerminationReason | None = None

    @property
    def last_observation(self) -> LoopObservation | None:
        """The observation of the latest recorded iteration."""
        return self.history[-1].observation if self.history else None

    @property
    def terminal(self) -> bool:
        """Whether the loop has stopped iterating."""
        return self.termination_reason is not None


class LoopResult(BaseModel):
    """The final result of a loop together with its state.

    Attributes:
        result: The latest iteration result.
        state: The loop state.
        interrupts: Pending LangGraph interrupts when the loop paused.
        error: The exception that ended the loop, when it failed.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    result: Any = Field(default=None, repr=False)
    state: LoopState
    interrupts: tuple[Any, ...] = ()
    error: BaseException | None = Field(default=None, exclude=True)

    @property
    def paused(self) -> bool:
        """Whether the loop is waiting for a resume."""
        return bool(self.interrupts)
