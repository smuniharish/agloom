"""Ordered pipeline topology."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.models import TopologyName
from agloom.topology._shared import (
    amodel_text,
    compile_graph,
    default_stage,
    model_text,
    node,
    safe_name,
)
from agloom.topology.base import ExecutionState, TopologyArtifact, TopologyBuildContext


class PipelineTopology:
    name = TopologyName.PIPELINE

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        stages = context.stages or (
            default_stage("analyze", "Analyze the request and identify requirements."),
            default_stage("draft", "Produce a complete response for the request."),
            default_stage("finalize", "Finalize and verify the response."),
        )
        builder = StateGraphType[Any](ExecutionState)
        previous = START
        for index, stage in enumerate(stages):
            node_name = f"stage_{index}_{safe_name(stage.name)}"

            def make_node(stage_item: Any):
                def run(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
                    response = model_text(context, stage_item.instruction, state)
                    output = response.content
                    if stage_item.transform is not None:
                        output = stage_item.transform(output)
                    return {"output": output}

                async def arun(
                    state: ExecutionState, _: RunnableConfig
                ) -> dict[str, Any]:
                    response = await amodel_text(context, stage_item.instruction, state)
                    output = response.content
                    if stage_item.transform is not None:
                        output = stage_item.transform(output)
                    return {"output": output}

                return node(run, arun)

            builder.add_node(node_name, make_node(stage))
            builder.add_edge(previous, node_name)
            previous = node_name
        builder.add_edge(previous, END)
        return TopologyArtifact(self.name, compile_graph(context, builder))
