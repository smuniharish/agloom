"""Reflection topology for bounded draft, evaluation, and revision."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph import StateGraph as StateGraphType

from agloom.models import TopologyName
from agloom.topology._shared import amodel_text, compile_graph, model_text, node
from agloom.topology.base import ExecutionState, TopologyArtifact, TopologyBuildContext


class ReflectionTopology:
    name = TopologyName.REFLECTION

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        def draft(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            response = model_text(
                context, "Create a complete initial response to the task.", state
            )
            return {"output": response.content}

        async def adraft(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
            response = await amodel_text(
                context, "Create a complete initial response to the task.", state
            )
            return {"output": response.content}

        builder = StateGraphType[Any](ExecutionState)
        builder.add_node("draft", node(draft, adraft))
        previous = "draft"
        for index in range(context.max_reflections):
            evaluate_name = f"evaluate_{index + 1}"
            revise_name = f"revise_{index + 1}"

            def make_evaluator(round_number: int):
                instruction = (
                    "Evaluate response quality for the task. Identify concrete "
                    f"errors or say APPROVED. This is review round {round_number}."
                )

                def run(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
                    response = model_text(context, instruction, state)
                    return {"review": response.content}

                async def arun(
                    state: ExecutionState, _: RunnableConfig
                ) -> dict[str, Any]:
                    response = await amodel_text(context, instruction, state)
                    return {"review": response.content}

                return node(run, arun)

            def revise(state: ExecutionState, _: RunnableConfig) -> dict[str, Any]:
                response = model_text(
                    context,
                    "Revise the response using the evaluation. "
                    "Return the final result.",
                    {
                        "task": state["task"],
                        "output": (
                            f"{state.get('output')}\nReview: {state.get('review')}"
                        ),
                    },
                )
                return {"output": response.content}

            async def arevise(
                state: ExecutionState, _: RunnableConfig
            ) -> dict[str, Any]:
                response = await amodel_text(
                    context,
                    "Revise the response using the evaluation. "
                    "Return the final result.",
                    {
                        "task": state["task"],
                        "output": (
                            f"{state.get('output')}\nReview: {state.get('review')}"
                        ),
                    },
                )
                return {"output": response.content}

            builder.add_node(evaluate_name, make_evaluator(index + 1))
            builder.add_node(revise_name, node(revise, arevise))
            builder.add_edge(previous, evaluate_name)
            builder.add_edge(evaluate_name, revise_name)
            previous = revise_name
        builder.add_edge(START, "draft")
        builder.add_edge(previous, END)
        return TopologyArtifact(self.name, compile_graph(context, builder))
