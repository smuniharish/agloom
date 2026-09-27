"""Planner topology with bounded, optionally recursive subtask execution."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.errors import RecursionError
from agloom.models import TopologyName
from agloom.topology._shared import (
    amodel_text,
    child_callback,
    compile_graph,
    model_messages,
    model_text,
    node,
    parse_subtasks,
    run_subtask,
)
from agloom.topology.base import (
    ExecutionState,
    TopologyArtifact,
    TopologyBuildContext,
)


class PlannerTopology:
    name = TopologyName.PLANNER

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        planner_instruction = (
            "Break the task into a short, ordered plan. Emit one actionable, "
            "self-contained subtask per line, with no more than 16 lines. "
            "Preserve all supplied facts and constraints needed by each child. "
            "Do not turn supplied facts into requests for external verification "
            "or require capabilities that the original task did not request."
        )

        def plan(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            response = model_text(context, planner_instruction, state)
            return {"subtasks": parse_subtasks(response)}

        async def aplan(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            response = await amodel_text(context, planner_instruction, state)
            return {"subtasks": parse_subtasks(response)}

        def execute(state: ExecutionState, config: RunnableConfig) -> dict[str, Any]:
            callback = (
                child_callback(config, async_mode=False)
                if context.recursion_enabled
                else None
            )
            results = []
            for subtask in state["subtasks"]:
                if callback is None and context.recursion_enabled:
                    raise RecursionError("recursive child executor was not configured")
                result = (
                    callback(subtask)
                    if callback is not None
                    else run_subtask(context, subtask)
                )
                results.append(result)
            return {"results": results, "output": "\n".join(map(str, results))}

        async def aexecute(
            state: ExecutionState, config: RunnableConfig
        ) -> dict[str, Any]:
            callback = (
                child_callback(config, async_mode=True)
                if context.recursion_enabled
                else None
            )
            results = []
            for subtask in state["subtasks"]:
                if context.recursion_enabled:
                    if callback is None:
                        raise RecursionError(
                            "recursive child executor was not configured"
                        )
                    results.append(await callback(subtask))
                else:
                    response = await context.model.ainvoke(
                        model_messages(
                            "Complete this planned subtask precisely.",
                            subtask,
                            system_prompt=context.system_prompt,
                        )
                    )
                    results.append(response.content)
            return {"results": results, "output": "\n".join(map(str, results))}

        builder = StateGraphType[Any](ExecutionState)
        builder.add_node("plan", node(plan, aplan))
        builder.add_node("execute", node(execute, aexecute))
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", END)
        return TopologyArtifact(self.name, compile_graph(context, builder))
