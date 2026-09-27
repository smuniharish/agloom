"""Swarm topology with bounded peer-to-peer handoffs."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.models import TopologyName, WorkerSpec
from agloom.topology._shared import (
    amodel_text,
    compile_graph,
    model_text,
    node,
    require_workers,
    swarm_route,
    worker_instruction,
)
from agloom.topology.base import ExecutionState, TopologyArtifact, TopologyBuildContext


class SwarmTopology:
    name = TopologyName.SWARM

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        require_workers(context.workers, self.name)
        builder = StateGraphType[Any](ExecutionState)
        ids = {
            worker.name: f"peer_{index}" for index, worker in enumerate(context.workers)
        }
        for worker in context.workers:
            node_id = ids[worker.name]

            def make_peer(peer: WorkerSpec):
                instruction = (
                    f"{worker_instruction(peer)} "
                    "Continue from the shared prior result. "
                    "End your response with exactly one routing directive: NEXT: "
                    f"<worker-name> or NEXT: STOP. Eligible peers: {', '.join(ids)}."
                )

                def run(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
                    response = model_text(context, instruction, state)
                    return {
                        "output": response.content,
                        "handoff_count": state.get("handoff_count", 0) + 1,
                    }

                async def arun(
                    state: ExecutionState, _: RunnableConfig
                ) -> dict[str, Any]:
                    response = await amodel_text(context, instruction, state)
                    return {
                        "output": response.content,
                        "handoff_count": state.get("handoff_count", 0) + 1,
                    }

                return node(run, arun)

            builder.add_node(node_id, make_peer(worker))
            builder.add_conditional_edges(
                node_id,
                lambda state, config: swarm_route(state, config, context.workers, ids),
                {**{peer_id: peer_id for peer_id in ids.values()}, END: END},
            )
        builder.add_edge(START, next(iter(ids.values())))
        return TopologyArtifact(self.name, compile_graph(context, builder))
