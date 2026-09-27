"""Hybrid topology composing isolated child LangGraph topologies."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.errors import ConfigurationError
from agloom.models import TopologyName
from agloom.topology._shared import (
    child_input,
    compile_graph,
    extract_output,
    node,
    without_checkpointer_thread,
)
from agloom.topology.base import ExecutionState, TopologyArtifact, TopologyBuildContext


class HybridTopology:
    name = TopologyName.HYBRID

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        if context.compile_child is None:
            raise ConfigurationError("Hybrid compilation requires a child compiler")
        if len(context.composition) < 2:
            raise ConfigurationError("Hybrid requires at least two child topologies")
        builder = StateGraphType[Any](ExecutionState)
        compiled = [context.compile_child(topology) for topology in context.composition]
        previous = START
        for index, child in enumerate(compiled):
            node_name = f"child_{index}_{context.composition[index].value}"

            def make_child(child_graph: Any):
                def run(state: ExecutionState, config: RunnableConfig):
                    result = child_graph.invoke(
                        child_input(state),
                        config=without_checkpointer_thread(config),
                    )
                    return {"output": extract_output(result)}

                async def arun(state: ExecutionState, config: RunnableConfig):
                    result = await child_graph.ainvoke(
                        child_input(state),
                        config=without_checkpointer_thread(config),
                    )
                    return {"output": extract_output(result)}

                return node(run, arun)

            builder.add_node(node_name, make_child(child))
            builder.add_edge(previous, node_name)
            previous = node_name
        builder.add_edge(previous, END)
        return TopologyArtifact(self.name, compile_graph(context, builder))
