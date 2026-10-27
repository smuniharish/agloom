"""Compile architecture specifications to LangGraph graphs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from langchain_core.runnables import Runnable

from agloom.errors import AgloomError, CompilationError, ConfigurationError
from agloom.models import ArchitectureSpec, TopologyName
from agloom.topology.base import Topology, TopologyArtifact, TopologyBuildContext
from agloom.topology.builtins import BUILTIN_TOPOLOGIES


class ArchitectureCompiler:
    """Builds topology graphs from validated specifications.

    Args:
        topologies: Replacement builders keyed by topology name. Builders not
            replaced use the built-in implementation.

    Raises:
        ConfigurationError: If a replacement builder reports a different name.
    """

    def __init__(
        self, topologies: Mapping[TopologyName, Topology] | None = None
    ) -> None:
        resolved: dict[TopologyName, Topology] = dict(BUILTIN_TOPOLOGIES)
        for name, builder in (topologies or {}).items():
            if builder.name is not name:
                raise ConfigurationError(
                    f"the builder registered for {name.value!r} "
                    f"reports {builder.name.value!r}"
                )
            resolved[name] = builder
        self._topologies = MappingProxyType(resolved)

    @property
    def topologies(self) -> Mapping[TopologyName, Topology]:
        """The builder used for each topology."""
        return self._topologies

    def compile(
        self, spec: ArchitectureSpec, context: TopologyBuildContext
    ) -> TopologyArtifact:
        """Compile one specification.

        Raises:
            CompilationError: If the builder fails or returns another topology.
        """

        def compile_child(child: TopologyName) -> Runnable[Any, Any]:
            child_spec = ArchitectureSpec(
                topology=child,
                workers=spec.workers,
                stages=spec.stages,
                system_prompt=spec.system_prompt,
            )
            return self._build(
                child_spec, context.child().specialize(child_spec, None)
            ).graph

        return self._build(spec, context.specialize(spec, compile_child))

    def _build(
        self, spec: ArchitectureSpec, context: TopologyBuildContext
    ) -> TopologyArtifact:
        builder = self._topologies[spec.topology]
        try:
            artifact = builder.build(context)
        except AgloomError:
            raise
        except Exception as error:
            raise CompilationError(
                f"failed to compile the {spec.topology.value} topology"
            ) from error
        if artifact.name is not spec.topology:
            raise CompilationError(
                f"the {spec.topology.value} builder returned {artifact.name.value}"
            )
        return artifact
