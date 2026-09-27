"""Compile validated specifications to shared LangGraph artifacts."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from agloom.errors import AgloomError, CompilationError, TopologyError
from agloom.models import (
    AgentConfig,
    ArchitectureSpec,
    ExecutionMode,
    TopologyName,
)
from agloom.topology.base import Topology, TopologyArtifact, TopologyBuildContext
from agloom.topology.builtins import BUILTIN_TOPOLOGIES


class ArchitectureCompiler:
    """Validates topology selection and dispatches through a builder registry."""

    def __init__(self, topologies: dict[TopologyName, Topology] | None = None) -> None:
        self._topologies = dict(topologies or BUILTIN_TOPOLOGIES)
        if set(self._topologies) != set(TopologyName):
            raise ValueError("compiler registry must contain exactly eight topologies")

    @property
    def topologies(self) -> tuple[TopologyName, ...]:
        return tuple(self._topologies)

    def register(self, topology: Topology) -> None:
        if topology.name is TopologyName.HYBRID and topology.name in self._topologies:
            self._topologies[topology.name] = topology
            return
        if topology.name in self._topologies:
            raise TopologyError(
                f"topology {topology.name.value!r} is already registered"
            )
        self._topologies[topology.name] = topology

    def compile(
        self,
        spec: ArchitectureSpec,
        config: AgentConfig,
    ) -> TopologyArtifact | None:
        if spec.decision.mode is ExecutionMode.DIRECT:
            if spec.decision.topology is not None:
                raise CompilationError("DIRECT specification cannot name a topology")
            return None

        topology_name = spec.decision.topology
        if topology_name is None:
            raise CompilationError("topology specification is missing its topology")
        topology = self._topologies.get(topology_name)
        if topology is None:
            raise CompilationError(
                f"topology {topology_name.value!r} is not registered"
            )
        if config.middleware and topology_name is not TopologyName.REACT:
            raise CompilationError(
                "LangChain middleware is supported only by the ReAct topology"
            )

        def compile_child(child_name: TopologyName) -> CompiledStateGraph:
            if child_name is TopologyName.HYBRID:
                raise CompilationError("Hybrid cannot contain another Hybrid")
            child = self._topologies.get(child_name)
            if child is None:
                raise CompilationError(
                    f"Hybrid child {child_name.value!r} is not registered"
                )
            if config.middleware and child_name is not TopologyName.REACT:
                raise CompilationError(
                    "LangChain middleware in Hybrid requires a ReAct child"
                )
            child_context = TopologyBuildContext(
                model=config.model,
                tools=config.tools,
                workers=spec.workers,
                stages=spec.stages,
                max_reflections=config.max_reflections,
                recursion_enabled=config.recursion,
                composition=(),
                capabilities=config.capabilities,
                system_prompt=spec.system_prompt,
                middleware=config.middleware,
                checkpointer=None,
            )
            return child.build(child_context).graph

        context = TopologyBuildContext(
            model=config.model,
            tools=config.tools,
            workers=spec.workers,
            stages=spec.stages,
            max_reflections=config.max_reflections,
            recursion_enabled=config.recursion,
            composition=spec.composition,
            capabilities=config.capabilities,
            system_prompt=spec.system_prompt,
            middleware=config.middleware,
            checkpointer=spec.checkpointer,
            interrupt_before=config.interrupt_before,
            interrupt_after=config.interrupt_after,
            compile_child=compile_child,
        )
        try:
            artifact = topology.build(context)
        except AgloomError:
            raise
        except Exception as error:
            raise CompilationError(
                f"failed to compile {topology_name.value} topology"
            ) from error
        if artifact.name is not topology_name:
            raise CompilationError(
                f"{topology_name.value} builder returned {artifact.name.value}"
            )
        if config.explainability is not None:
            try:
                instrumented = config.explainability.instrument(artifact.graph)
            except Exception as error:
                raise CompilationError(
                    "explainability instrumentation failed"
                ) from error
            artifact = TopologyArtifact(artifact.name, instrumented)
        return artifact
