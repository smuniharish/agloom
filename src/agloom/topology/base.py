"""Shared topology build contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from agloom.models import PipelineStage, TopologyName, WorkerSpec


class ExecutionState(TypedDict, total=False):
    task: str
    messages: list[Any]
    output: Any
    subtasks: list[str]
    results: list[Any]
    selected_worker: str
    handoff_count: int
    blackboard: dict[str, Any]
    review: str


@dataclass(frozen=True)
class TopologyBuildContext:
    model: BaseChatModel
    tools: tuple[BaseTool | Any, ...]
    workers: tuple[WorkerSpec, ...]
    stages: tuple[PipelineStage, ...]
    max_reflections: int
    recursion_enabled: bool
    composition: tuple[TopologyName, ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    middleware: tuple[Any, ...] = ()
    checkpointer: Any = None
    interrupt_before: tuple[str, ...] = ()
    interrupt_after: tuple[str, ...] = ()
    compile_child: Callable[[TopologyName], CompiledStateGraph] | None = None


@dataclass(frozen=True)
class TopologyArtifact:
    name: TopologyName
    graph: CompiledStateGraph


class Topology(Protocol):
    """One topology builder; runtime concerns remain outside this contract."""

    name: TopologyName

    def build(self, context: TopologyBuildContext) -> TopologyArtifact: ...
