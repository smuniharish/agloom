"""Supervisor topology with a finite, explicitly configured worker set."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.models import TopologyName
from agloom.topology._shared import (
    amodel_text,
    async_choose_worker,
    choose_worker,
    compile_graph,
    model_text,
    node,
    require_workers,
    worker_by_name,
    worker_instruction,
)
from agloom.topology.base import ExecutionState, TopologyArtifact, TopologyBuildContext


class SupervisorTopology:
    name = TopologyName.SUPERVISOR

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        require_workers(context.workers, self.name)

        def choose(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            worker = choose_worker(context, state["task"])
            return {"selected_worker": worker.name}

        async def achoose(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            worker = await async_choose_worker(context, state["task"])
            return {"selected_worker": worker.name}

        def work(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            worker = worker_by_name(context.workers, state["selected_worker"])
            response = model_text(
                context,
                worker_instruction(worker),
                {"task": state["task"], "output": None},
            )
            return {"output": response.content}

        async def awork(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            worker = worker_by_name(context.workers, state["selected_worker"])
            response = await amodel_text(
                context,
                worker_instruction(worker),
                {"task": state["task"], "output": None},
            )
            return {"output": response.content}

        builder = StateGraphType[Any](ExecutionState)
        builder.add_node("supervisor", node(choose, achoose))
        builder.add_node("worker", node(work, awork))
        builder.add_edge(START, "supervisor")
        builder.add_edge("supervisor", "worker")
        builder.add_edge("worker", END)
        return TopologyArtifact(self.name, compile_graph(context, builder))
