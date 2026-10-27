"""Pipeline: ordered stages, each building on the previous stage's output."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agloom.models import PipelineStage, TopologyName
from agloom.topology._shared import (
    State,
    agent_input,
    answer_update,
    node,
    node_name,
    request,
)
from agloom.topology.base import TopologyArtifact, TopologyBuildContext

DEFAULT_STAGES = (
    PipelineStage(
        name="analyze", instruction="Analyze the request and list its requirements."
    ),
    PipelineStage(
        name="draft", instruction="Produce a complete response that meets them."
    ),
    PipelineStage(name="finalize", instruction="Verify and finalize the response."),
)


class PipelineState(TypedDict, total=False):
    """State of the Pipeline topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    output: str
    loop: dict[str, Any]


class PipelineTopology:
    """Runs ordered, tool-capable stages; each stage receives the previous output.

    Graph nodes: one `stage_<i>_<name>` node per stage. Without configured or
    suggested stages, the stages analyze, draft, and finalize are used.
    """

    name = TopologyName.PIPELINE

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Pipeline graph."""
        stages = context.stages or DEFAULT_STAGES
        builder = StateGraph[Any](PipelineState)
        previous = START
        for index, stage in enumerate(stages):
            stage_id = node_name("stage", index, stage.name)
            agent = context.agent(stage.instruction, name=f"agloom_{stage_id}")

            def finish(
                state: State, result: Any, item: PipelineStage = stage
            ) -> dict[str, Any]:
                return answer_update(result, item.transform)

            builder.add_node(
                stage_id,
                node(
                    stage_id,
                    lambda state: agent_input(
                        request(
                            state, ("Output of the previous stage", state.get("output"))
                        )
                    ),
                    agent,
                    finish,
                ),
            )
            builder.add_edge(previous, stage_id)
            previous = stage_id
        builder.add_edge(previous, END)
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
