"""Core immutable configuration and execution contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agloom.errors import ConfigurationError


class TopologyName(StrEnum):
    REACT = "react"
    SUPERVISOR = "supervisor"
    PIPELINE = "pipeline"
    PLANNER = "planner"
    REFLECTION = "reflection"
    SWARM = "swarm"
    BLACKBOARD = "blackboard"
    HYBRID = "hybrid"


TOPOLOGY_NAMES: tuple[TopologyName, ...] = tuple(TopologyName)


class ExecutionMode(StrEnum):
    DIRECT = "direct"
    TOPOLOGY = "topology"


class Strategy(Protocol):
    """Selects DIRECT or one of the supplied topology candidates."""

    def decide(
        self,
        analysis: TaskAnalysis,
        candidates: Sequence[TopologyName] | None = None,
        *,
        allow_direct: bool = False,
    ) -> ExecutionDecision: ...


class RecursionPolicy(BaseModel):
    """Hard bounds applied to a single invocation and all of its descendants."""

    model_config = ConfigDict(frozen=True)

    max_depth: int = Field(default=3, ge=1, le=8)
    max_workers: int = Field(default=4, ge=1, le=32)
    max_subtasks: int = Field(default=16, ge=1, le=128)
    max_execution_time: float = Field(default=120.0, gt=0, le=3600)
    allowed_nested_topologies: tuple[TopologyName, ...] = TOPOLOGY_NAMES


class SuggestedWorker(BaseModel):
    """Analyzer-proposed model role for an automatically selected topology."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    instructions: str = Field(default="", max_length=2_000)

    @field_validator("name", "description", "instructions")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class SuggestedStage(BaseModel):
    """Analyzer-proposed stage for an automatically selected pipeline."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80)
    instruction: str = Field(min_length=1, max_length=2_000)

    @field_validator("name", "instruction")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class ExecutionBlueprint(BaseModel):
    """Bounded, task-specific recommendations that do not select execution."""

    model_config = ConfigDict(frozen=True)

    instructions: str = Field(default="", max_length=4_000)
    workers: tuple[SuggestedWorker, ...] = Field(default=(), max_length=32)
    stages: tuple[SuggestedStage, ...] = Field(default=(), max_length=16)
    composition: tuple[TopologyName, ...] = ()

    @field_validator("instructions")
    @classmethod
    def normalize_instructions(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_names_and_composition(self) -> ExecutionBlueprint:
        worker_names = [worker.name for worker in self.workers]
        if len(set(worker_names)) != len(worker_names):
            raise ValueError("suggested worker names must be unique")
        stage_names = [stage.name for stage in self.stages]
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("suggested stage names must be unique")
        if TopologyName.HYBRID in self.composition:
            raise ValueError("Hybrid cannot contain Hybrid as a child")
        if self.composition and not 2 <= len(self.composition) <= 4:
            raise ValueError("suggested composition requires two to four topologies")
        return self


class TaskAnalysis(BaseModel):
    """Structured observations about a task; contains no execution decision."""

    model_config = ConfigDict(frozen=True)

    complexity: int = Field(ge=0, le=10)
    estimated_steps: int = Field(ge=1, le=128)
    requires_decomposition: bool = False
    requires_delegation: bool = False
    requires_parallelism: bool = False
    requires_tools: bool = False
    requires_validation: bool = False
    requires_ordering: bool = False
    requires_shared_workspace: bool = False
    requires_composition: bool = False
    missing_capabilities: tuple[str, ...] = ()
    context_demand: int = Field(default=0, ge=0, le=10)
    rationale: str = ""
    blueprint: ExecutionBlueprint = Field(default_factory=ExecutionBlueprint)

    @field_validator("missing_capabilities")
    @classmethod
    def require_named_missing_capabilities(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("missing capability names must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("missing capability names must be unique")
        return normalized


class ExecutionDecision(BaseModel):
    """A strategy result that selects DIRECT or one actual topology."""

    model_config = ConfigDict(frozen=True)

    mode: ExecutionMode
    topology: TopologyName | None = None
    reason: str

    @model_validator(mode="after")
    def validate_mode_topology(self) -> ExecutionDecision:
        if (self.mode is ExecutionMode.TOPOLOGY) != (self.topology is not None):
            raise ValueError("topology must be set exactly when mode is topology")
        return self


class PipelineStage(BaseModel):
    """One callable stage in an explicitly configured ordered pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    instruction: str
    transform: Callable[[Any], Any] | None = None

    @field_validator("name", "instruction")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pipeline stage name and instruction must be nonempty")
        return value


class WorkerSpec(BaseModel):
    """A bounded worker definition shared by delegation topologies."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    instructions: str = ""

    @field_validator("name", "description")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("worker name and description must be nonempty")
        return value


class ArchitectureSpec(BaseModel):
    """Validated, compiler-ready description of one execution architecture."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    decision: ExecutionDecision
    composition: tuple[TopologyName, ...] = ()
    checkpointer: Any = None
    workers: tuple[WorkerSpec, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    system_prompt: str | None = None

    @model_validator(mode="after")
    def validate_composition(self) -> ArchitectureSpec:
        if self.decision.topology is TopologyName.HYBRID:
            if len(self.composition) < 2:
                raise ValueError("Hybrid requires at least two child topologies")
            if TopologyName.HYBRID in self.composition:
                raise ValueError("Hybrid cannot contain Hybrid as a child")
            if len(self.composition) > 4:
                raise ValueError("Hybrid supports at most four child topologies")
        elif self.composition:
            raise ValueError("composition is only valid for Hybrid")
        return self


@dataclass(frozen=True)
class AgentConfig:
    """Validated developer options held by one agent instance."""

    model: BaseChatModel
    tools: tuple[BaseTool | Callable[..., Any], ...] = ()
    pattern_candidates: tuple[TopologyName, ...] | None = None
    recursion: bool = False
    recursion_policy: RecursionPolicy = field(default_factory=RecursionPolicy)
    composition: tuple[TopologyName, ...] = ()
    workers: tuple[WorkerSpec, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    checkpointer: Any = None
    checkpoint_authorizer: Callable[[str], bool] | None = None
    interrupt_before: tuple[str, ...] = ()
    interrupt_after: tuple[str, ...] = ()
    event_sink: Any = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    middleware: tuple[Any, ...] = ()
    explainability: Any = None
    system_prompt: str | None = None
    max_reflections: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.max_reflections <= 5:
            raise ConfigurationError("max_reflections must be between 1 and 5")
        if len({worker.name for worker in self.workers}) != len(self.workers):
            raise ConfigurationError("worker names must be unique")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ConfigurationError("pipeline stage names must be unique")
        if any(not name.strip() for name in self.capabilities):
            raise ConfigurationError("capability names must not be empty")
        object.__setattr__(
            self,
            "capabilities",
            MappingProxyType(dict(self.capabilities)),
        )
        if self.pattern_candidates == ():
            raise ConfigurationError("pattern must not be an empty list")
        if self.pattern_candidates is not None and (
            len(set(self.pattern_candidates)) != len(self.pattern_candidates)
        ):
            raise ConfigurationError("pattern candidates must be unique")
        if self.composition and len(set(self.composition)) != len(self.composition):
            raise ConfigurationError("Hybrid child topologies must be unique")
        if (
            self.interrupt_before or self.interrupt_after
        ) and self.checkpointer is None:
            raise ConfigurationError(
                "interrupt_before/interrupt_after require a LangGraph checkpointer"
            )
        if self.checkpointer is not None and (
            self.pattern_candidates is None or len(self.pattern_candidates) != 1
        ):
            raise ConfigurationError(
                "checkpointing requires one explicit topology for stable resume"
            )


def parse_topology(value: str | TopologyName) -> TopologyName:
    """Parse a topology name without permitting DIRECT as a topology."""

    if isinstance(value, TopologyName):
        return value
    try:
        return TopologyName(value.strip().lower())
    except (AttributeError, ValueError) as error:
        raise ConfigurationError(f"unknown topology {value!r}") from error


def parse_pattern(
    pattern: str | TopologyName | Sequence[str | TopologyName] | None,
) -> tuple[TopologyName, ...] | None:
    if pattern is None:
        return None
    if isinstance(pattern, (str, TopologyName)):
        return (parse_topology(pattern),)
    if not isinstance(pattern, Sequence):
        raise ConfigurationError("pattern must be a topology name or a sequence")
    result = tuple(parse_topology(item) for item in pattern)
    if not result:
        raise ConfigurationError("pattern must not be an empty list")
    if len(set(result)) != len(result):
        raise ConfigurationError("pattern candidates must be unique")
    return result


def normalize_composition(
    composition: Sequence[str | TopologyName] | None,
) -> tuple[TopologyName, ...]:
    if composition is None:
        return ()
    return tuple(parse_topology(item) for item in composition)


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a shallow immutable copy for topology-specific configuration."""

    from types import MappingProxyType

    return MappingProxyType(dict(value))
