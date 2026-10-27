"""Hybrid: an explicit sequence of two to four child topologies."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agloom.errors import CompilationError
from agloom.models import TopologyName
from agloom.topology._shared import State, node, topology_answer, topology_input
from agloom.topology.base import TopologyArtifact, TopologyBuildContext, compose_prompt


class HybridState(TypedDict, total=False):
    """State of the Hybrid topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    output: str
    loop: dict[str, Any]


class HybridTopology:
    """Runs the configured child topologies in order, passing each result forward.

    Graph nodes: one `child_<i>_<topology>` node per child topology.
    """

    name = TopologyName.HYBRID

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Hybrid graph."""
        if context.compile_child is None:
            raise CompilationError("Hybrid compilation requires a child compiler")
        builder = StateGraph[Any](HybridState)
        previous = START
        for index, child in enumerate(context.composition):
            child_id = f"child_{index}_{child.value}"
            react = child is TopologyName.REACT

            def build_input(state: State, react: bool = react) -> dict[str, Any]:
                prior = state.get("output")
                child_context = compose_prompt(
                    state.get("context"),
                    f"Result of the previous stage:\n{prior}" if prior else None,
                )
                return topology_input(
                    react, state["task"], state.get("messages") or [], child_context
                )

            builder.add_node(
                child_id,
                node(
                    child_id,
                    build_input,
                    context.compile_child(child),
                    lambda state, result, react=react: topology_answer(react, result),
                ),
            )
            builder.add_edge(previous, child_id)
            previous = child_id
        builder.add_edge(previous, END)
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
