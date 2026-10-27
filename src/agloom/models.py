"""Immutable configuration and execution-selection models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agloom.errors import ConfigurationError


class TopologyName(StrEnum):
    """The eight execution topologies.

    DIRECT is an execution mode, not a topology.
    """

    REACT = "react"
    SUPERVISOR = "supervisor"
    PIPELINE = "pipeline"
    PLANNER = "planner"
    REFLECTION = "reflection"
    SWARM = "swarm"
    BLACKBOARD = "blackboard"
    HYBRID = "hybrid"


TOPOLOGY_NAMES: tuple[TopologyName, ...] = tuple(TopologyName)
WORKER_TOPOLOGIES: frozenset[TopologyName] = frozenset(
    {TopologyName.SUPERVISOR, TopologyName.SWARM, TopologyName.BLACKBOARD}
)


class ExecutionMode(StrEnum):
    """How one request is executed."""

    DIRECT = "direct"
    TOPOLOGY = "topology"


class ExecutionLimits(BaseModel):
    """Hard bounds for one invocation and every recursive child it starts.

    The recursion budget covers the whole invocation, including every Loop
    Engineering iteration. Resuming a paused thread starts a new invocation.

    Attributes:
        max_depth: Maximum recursive child depth. `0` disables recursion; above
            `0`, each Planner step runs as a recursive child execution.
        max_children: Maximum number of recursive child executions in one
            invocation.
        max_workers: Maximum number of branches executed in parallel by one
            fan-out (Planner steps or Blackboard contributors).
        max_plan_steps: Maximum number of steps in one Planner plan.
        max_handoffs: Maximum delegations in one Supervisor or Swarm run.
        max_execution_time: Wall-clock limit in seconds for recursive work,
            measured from the start of the first child execution.
        allowed_nested_topologies: Topologies that recursive children may select.
    """

    model_config = ConfigDict(frozen=True)

    max_depth: int = Field(default=0, ge=0, le=8)
    max_children: int = Field(default=16, ge=1, le=256)
    max_workers: int = Field(default=4, ge=1, le=32)
    max_plan_steps: int = Field(default=8, ge=1, le=64)
    max_handoffs: int = Field(default=8, ge=1, le=64)
    max_execution_time: float = Field(default=120.0, gt=0, le=3_600)
    allowed_nested_topologies: tuple[TopologyName, ...] = TOPOLOGY_NAMES

    @field_validator("allowed_nested_topologies")
    @classmethod
    def _require_unique_topologies(
        cls, value: tuple[TopologyName, ...]
    ) -> tuple[TopologyName, ...]:
        if not value:
            raise ValueError("allowed_nested_topologies must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("allowed_nested_topologies must be unique")
        return value

    @property
    def recursion_enabled(self) -> bool:
        """Whether recursive child execution is enabled."""
        return self.max_depth > 0


class WorkerSpec(BaseModel):
    """A worker used by the Supervisor, Swarm, and Blackboard topologies.

    Attributes:
        name: Unique worker name shown to routing decisions.
        description: The worker's scope.
        instructions: Additional instructions for the worker.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    instructions: str = Field(default="", max_length=4_000)

    @field_validator("name", "description", "instructions")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("name", "description")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("worker name and description must not be blank")
        return value


class PipelineStage(BaseModel):
    """One stage of an ordered Pipeline.

    Attributes:
        name: Unique stage name.
        instruction: What the stage must do with the previous stage's output.
        transform: Optional function applied to the stage's text output.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(min_length=1, max_length=80)
    instruction: str = Field(min_length=1, max_length=4_000)
    transform: Callable[[str], str] | None = None

    @field_validator("name", "instruction")
    @classmethod
    def _require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("pipeline stage name and instruction must not be blank")
        return value


class SuggestedWorker(BaseModel):
    """A worker proposed by task analysis when none were configured."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80, description="Unique worker name.")
    description: str = Field(
        min_length=1, max_length=500, description="The worker's distinct scope."
    )
    instructions: str = Field(
        default="", max_length=2_000, description="Concrete worker instructions."
    )


class SuggestedStage(BaseModel):
    """A pipeline stage proposed by task analysis when none were configured."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80, description="Unique stage name.")
    instruction: str = Field(
        min_length=1, max_length=2_000, description="Outcome-oriented instruction."
    )


class ExecutionBlueprint(BaseModel):
    """Bounded, task-specific recommendations produced by task analysis."""

    model_config = ConfigDict(frozen=True)

    instructions: str = Field(
        default="",
        max_length=4_000,
        description=(
            "Task-specific objective, constraints, and success criteria. "
            "Never grants capabilities or weakens safety rules."
        ),
    )
    workers: tuple[SuggestedWorker, ...] = Field(
        default=(),
        max_length=16,
        description="Workers to use when none are configured.",
    )
    stages: tuple[SuggestedStage, ...] = Field(
        default=(), max_length=16, description="Stages to use when none are configured."
    )
    composition: tuple[TopologyName, ...] = Field(
        default=(),
        description="Two to four non-hybrid topologies, only when Hybrid is required.",
    )

    @model_validator(mode="after")
    def _validate_blueprint(self) -> ExecutionBlueprint:
        if len({worker.name for worker in self.workers}) != len(self.workers):
            raise ValueError("suggested worker names must be unique")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("suggested stage names must be unique")
        if self.composition:
            _validate_composition(self.composition)
        return self


class TaskAnalysis(BaseModel):
    """Structured observations about a task. Contains no execution decision."""

    model_config = ConfigDict(frozen=True)

    complexity: int = Field(
        ge=0, le=10, description="Difficulty from 0 (trivial) to 10 (very complex)."
    )
    estimated_steps: int = Field(
        ge=1, le=128, description="Estimated number of distinct steps."
    )
    requires_tools: bool = Field(
        default=False, description="An available tool or capability is needed."
    )
    requires_decomposition: bool = Field(
        default=False, description="The task must be split into dependent subtasks."
    )
    requires_parallelism: bool = Field(
        default=False, description="Independent subtasks can run at the same time."
    )
    requires_delegation: bool = Field(
        default=False,
        description="Specialist workers should be coordinated by a supervisor.",
    )
    requires_handoffs: bool = Field(
        default=False,
        description="Peer workers should hand control to each other dynamically.",
    )
    requires_shared_workspace: bool = Field(
        default=False,
        description="Several workers should contribute to one shared workspace.",
    )
    requires_ordering: bool = Field(
        default=False, description="Work must pass through fixed ordered stages."
    )
    requires_validation: bool = Field(
        default=False, description="The answer should be reviewed and revised."
    )
    requires_composition: bool = Field(
        default=False,
        description="Two or more topologies must be composed in sequence.",
    )
    missing_capabilities: tuple[str, ...] = Field(
        default=(),
        description=(
            "Actions the user explicitly asked for that no listed tool or "
            "capability can perform."
        ),
    )
    rationale: str = Field(
        default="", max_length=2_000, description="One-sentence justification."
    )
    blueprint: ExecutionBlueprint = Field(default_factory=ExecutionBlueprint)

    @field_validator("missing_capabilities")
    @classmethod
    def _normalize_missing(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("missing capability names must not be blank")
        return tuple(dict.fromkeys(normalized))


class ExecutionDecision(BaseModel):
    """A strategy result that selects DIRECT or exactly one topology."""

    model_config = ConfigDict(frozen=True)

    mode: ExecutionMode
    topology: TopologyName | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_mode(self) -> ExecutionDecision:
        if (self.mode is ExecutionMode.TOPOLOGY) != (self.topology is not None):
            raise ValueError("topology must be set exactly when mode is topology")
        return self


class ArchitectureSpec(BaseModel):
    """A validated, compiler-ready description of one topology execution."""

    model_config = ConfigDict(frozen=True)

    topology: TopologyName
    composition: tuple[TopologyName, ...] = ()
    workers: tuple[WorkerSpec, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    system_prompt: str | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> ArchitectureSpec:
        if self.topology is TopologyName.HYBRID:
            _validate_composition(self.composition)
        elif self.composition:
            raise ValueError("composition is only valid for the Hybrid topology")
        needs_workers = self.topology in WORKER_TOPOLOGIES or any(
            child in WORKER_TOPOLOGIES for child in self.composition
        )
        if needs_workers and not self.workers:
            raise ValueError(f"the {self.topology.value} topology requires workers")
        return self

    @property
    def cache_key(self) -> tuple[object, ...]:
        """A hashable identity for compiled-graph caching."""
        return (
            self.topology,
            self.composition,
            self.workers,
            tuple(
                (stage.name, stage.instruction, stage.transform)
                for stage in self.stages
            ),
            self.system_prompt,
        )


class Strategy(Protocol):
    """Selects DIRECT or one topology from a task analysis."""

    def decide(
        self,
        analysis: TaskAnalysis,
        candidates: Sequence[TopologyName] | None = None,
        *,
        allow_direct: bool = True,
    ) -> ExecutionDecision:
        """Return the execution decision for one analyzed task."""
        ...


def parse_topology(value: str | TopologyName) -> TopologyName:
    """Parse one topology name. DIRECT is rejected because it is not a topology."""
    if isinstance(value, TopologyName):
        return value
    try:
        return TopologyName(value.strip().lower())
    except (AttributeError, ValueError) as error:
        raise ConfigurationError(f"unknown topology {value!r}") from error


def parse_pattern(
    pattern: str | TopologyName | Sequence[str | TopologyName] | None,
) -> tuple[TopologyName, ...] | None:
    """Parse one explicit topology or a non-empty, duplicate-free allow-list."""
    if pattern is None:
        return None
    if isinstance(pattern, str):
        return (parse_topology(pattern),)
    if not isinstance(pattern, Sequence):
        raise ConfigurationError(
            "pattern must be a topology name or a sequence of names"
        )
    parsed = tuple(parse_topology(item) for item in pattern)
    if not parsed:
        raise ConfigurationError("pattern must not be an empty sequence")
    if len(set(parsed)) != len(parsed):
        raise ConfigurationError("pattern candidates must be unique")
    return parsed


def parse_composition(
    composition: Sequence[str | TopologyName] | None,
) -> tuple[TopologyName, ...]:
    """Parse and validate an explicit Hybrid composition."""
    if composition is None:
        return ()
    if isinstance(composition, str):
        raise ConfigurationError("composition must be a sequence of topology names")
    parsed = tuple(parse_topology(item) for item in composition)
    try:
        _validate_composition(parsed)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    return parsed


def _validate_composition(composition: Sequence[TopologyName]) -> None:
    if not 2 <= len(composition) <= 4:
        raise ValueError("Hybrid composition requires two to four topologies")
    if TopologyName.HYBRID in composition:
        raise ValueError("Hybrid composition cannot contain Hybrid")
    if len(set(composition)) != len(composition):
        raise ValueError("Hybrid composition topologies must be unique")
