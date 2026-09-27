"""Blackboard topology with worker-owned shared contributions."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.errors import ExecutionError
from agloom.models import TopologyName, WorkerSpec
from agloom.topology._shared import (
    amodel_text,
    compile_graph,
    model_text,
    node,
    require_workers,
    safe_name,
    worker_instruction,
)
from agloom.topology.base import ExecutionState, TopologyArtifact, TopologyBuildContext


class BlackboardTopology:
    name = TopologyName.BLACKBOARD

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        require_workers(context.workers, self.name)
        builder = StateGraphType[Any](ExecutionState)
        previous = START
        for index, worker in enumerate(context.workers):
            node_name = f"worker_{index}_{safe_name(worker.name)}"

            def make_contributor(peer: WorkerSpec):
                instruction = (
                    f"{worker_instruction(peer)} Read the current shared "
                    "blackboard snapshot, contribute your findings, and do not "
                    "overwrite another worker's entry."
                )

                def run(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
                    board = dict(state.get("blackboard", {}))
                    if peer.name in board:
                        raise ExecutionError(
                            f"blackboard entry {peer.name!r} already has an owner"
                        )
                    response = model_text(
                        context,
                        instruction,
                        {
                            "task": state["task"],
                            "output": f"Blackboard snapshot: {board}",
                        },
                    )
                    board[peer.name] = response.content
                    return {"blackboard": board}

                async def arun(
                    state: ExecutionState, _: RunnableConfig
                ) -> dict[str, Any]:
                    board = dict(state.get("blackboard", {}))
                    if peer.name in board:
                        raise ExecutionError(
                            f"blackboard entry {peer.name!r} already has an owner"
                        )
                    response = await amodel_text(
                        context,
                        instruction,
                        {
                            "task": state["task"],
                            "output": f"Blackboard snapshot: {board}",
                        },
                    )
                    board[peer.name] = response.content
                    return {"blackboard": board}

                return node(run, arun)

            builder.add_node(node_name, make_contributor(worker))
            builder.add_edge(previous, node_name)
            previous = node_name

        def aggregate(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            response = model_text(
                context,
                "Synthesize the shared blackboard contributions into one answer.",
                {
                    "task": state["task"],
                    "output": state.get("blackboard", {}),
                },
            )
            return {"output": response.content}

        async def aaggregate(
            state: ExecutionState, _: RunnableConfig
        ) -> dict[str, Any]:
            response = await amodel_text(
                context,
                "Synthesize the shared blackboard contributions into one answer.",
                {
                    "task": state["task"],
                    "output": state.get("blackboard", {}),
                },
            )
            return {"output": response.content}

        builder.add_node("aggregate", node(aggregate, aaggregate))
        builder.add_edge(previous, "aggregate")
        builder.add_edge("aggregate", END)
        return TopologyArtifact(self.name, compile_graph(context, builder))
