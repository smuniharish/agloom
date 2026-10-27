"""Planner: a dependency-aware plan executed in parallel waves."""

from __future__ import annotations

from collections.abc import Sequence
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from agloom._runtime import current_run
from agloom.errors import ExecutionError
from agloom.models import TopologyName
from agloom.topology._shared import (
    State,
    agent_input,
    agent_text,
    answer_update,
    ensure_active,
    merge_dicts,
    model_input,
    node,
    request,
)
from agloom.topology.base import TopologyArtifact, TopologyBuildContext, compose_prompt

_PLANNING = """\
Break the request into the smallest set of self-contained steps, at most
{limit}. Give every step a short unique id, a complete instruction that
includes every fact it needs, and the ids of the steps whose results it
requires. Steps without unmet dependencies run in parallel. Do not add steps
that only verify facts supplied in the request."""

_STEP = "Complete exactly the assigned step of a larger plan."
_SYNTHESIS = "Combine the step results into one complete answer to the request."
_REPAIRABLE = (ExecutionError, OutputParserException, ValidationError)
_PROBLEM_CHARS = 500


def _problem(error: Exception) -> str:
    return str(error).strip()[:_PROBLEM_CHARS] or type(error).__name__


class PlanStep(BaseModel):
    """One step of a plan."""

    id: str = Field(min_length=1, max_length=40, description="Short unique step id.")
    task: str = Field(
        min_length=1, description="Self-contained instruction for the step."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of the steps whose results this step needs.",
    )


class Plan(BaseModel):
    """A dependency-ordered plan."""

    steps: list[PlanStep] = Field(min_length=1, description="The plan steps.")


class PlannerState(TypedDict, total=False):
    """State of the Planner topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    plan: list[dict[str, Any]]
    results: Annotated[dict[str, str], merge_dicts]
    output: str
    loop: dict[str, Any]


def validate_plan(steps: Sequence[PlanStep], limit: int) -> list[dict[str, Any]]:
    """Validate a plan and return its steps as plain dictionaries.

    Raises:
        ExecutionError: If the plan is too long, has duplicate or unknown ids,
            or contains a dependency cycle.
    """
    if len(steps) > limit:
        raise ExecutionError(f"the plan has {len(steps)} steps; the limit is {limit}")
    ids = [step.id for step in steps]
    if len(set(ids)) != len(ids):
        raise ExecutionError("plan step ids must be unique")
    known = set(ids)
    graph: dict[str, set[str]] = {}
    for step in steps:
        unknown = set(step.depends_on) - known
        if unknown:
            raise ExecutionError(
                f"plan step {step.id!r} depends on unknown steps {sorted(unknown)}"
            )
        graph[step.id] = set(step.depends_on)
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as error:
        raise ExecutionError("the plan contains a dependency cycle") from error
    return [step.model_dump() for step in steps]


def ready_steps(
    plan: Sequence[dict[str, Any]], results: dict[str, str]
) -> list[dict[str, Any]]:
    """Return the unexecuted steps whose dependencies have results."""
    return [
        step
        for step in plan
        if step["id"] not in results
        and all(dependency in results for dependency in step["depends_on"])
    ]


class PlannerTopology:
    """Plans dependent steps, runs ready steps in parallel waves, then synthesizes.

    Graph nodes: `plan`, `step`, `join`, and `synthesize`. Steps of one wave run
    concurrently, bounded by `ExecutionLimits.max_workers`; plans are bounded by
    `ExecutionLimits.max_plan_steps`. An invalid plan (unknown or duplicate step
    ids, a dependency cycle, too many steps, or an unparsable plan) gets one
    correction round with the validation error before the run fails. With
    recursion enabled, every step runs as a recursive child execution.
    """

    name = TopologyName.PLANNER

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Planner graph."""
        limit = context.limits.max_plan_steps
        planner = context.decision(Plan, name="agloom_planner_plan")
        worker = context.agent(_STEP, name="agloom_planner_step")
        planning = compose_prompt(context.system_prompt, _PLANNING.format(limit=limit))

        def plan_input(state: State, problem: str | None) -> list[BaseMessage]:
            correction = (
                f"The previous plan was rejected: {problem}\n"
                "Return a corrected plan that resolves this."
                if problem
                else None
            )
            return model_input(planning, request(state, ("Correction", correction)))

        def accept(plan: Any) -> dict[str, Any]:
            if not isinstance(plan, Plan):
                raise ExecutionError("the planner returned no plan")
            return {"plan": validate_plan(plan.steps, limit)}

        def make_plan(state: State, config: RunnableConfig) -> dict[str, Any]:
            ensure_active(config)
            try:
                return accept(planner.invoke(plan_input(state, None), config))
            except _REPAIRABLE as error:
                problem = _problem(error)
            return accept(planner.invoke(plan_input(state, problem), config))

        async def amake_plan(state: State, config: RunnableConfig) -> dict[str, Any]:
            ensure_active(config)
            try:
                return accept(await planner.ainvoke(plan_input(state, None), config))
            except _REPAIRABLE as error:
                problem = _problem(error)
            return accept(await planner.ainvoke(plan_input(state, problem), config))

        def step_prompt(payload: State) -> str:
            inputs = "\n\n".join(
                f"[{dependency}] {result}"
                for dependency, result in payload["inputs"].items()
            )
            return request(
                payload,
                ("Your step", payload["step"]["task"]),
                ("Results of prerequisite steps", inputs),
            )

        def finish_step(payload: State, text: str) -> dict[str, Any]:
            return {"results": {payload["step"]["id"]: text}}

        def run_step(payload: State, config: RunnableConfig) -> dict[str, Any]:
            ensure_active(config)
            run = current_run(config)
            prompt = step_prompt(payload)
            if run is not None and run.child is not None:
                return finish_step(payload, run.child.run(prompt, config))
            return finish_step(
                payload, agent_text(worker.invoke(agent_input(prompt), config))
            )

        async def arun_step(payload: State, config: RunnableConfig) -> dict[str, Any]:
            ensure_active(config)
            run = current_run(config)
            prompt = step_prompt(payload)
            if run is not None and run.child is not None:
                return finish_step(payload, await run.child.arun(prompt, config))
            result = await worker.ainvoke(agent_input(prompt), config)
            return finish_step(payload, agent_text(result))

        def dispatch(state: State) -> list[Send] | str:
            results = state.get("results") or {}
            ready = ready_steps(state["plan"], results)
            if not ready:
                return "synthesize"
            # At most max_workers steps run at once; the rest follow after `join`.
            return [
                Send(
                    "step",
                    {
                        "task": state["task"],
                        "messages": state.get("messages") or [],
                        "context": state.get("context"),
                        "step": step,
                        "inputs": {name: results[name] for name in step["depends_on"]},
                    },
                )
                for step in ready[: context.limits.max_workers]
            ]

        def plan_results(state: State) -> str:
            ordered = [
                f"[{step['id']}] {step['task']}\n{state['results'][step['id']]}"
                for step in state["plan"]
            ]
            return "\n\n".join(ordered)

        builder = StateGraph[Any](PlannerState)
        builder.add_node(
            "plan", RunnableLambda(make_plan, afunc=amake_plan, name="plan")
        )
        builder.add_node("step", RunnableLambda(run_step, afunc=arun_step, name="step"))
        builder.add_node("join", lambda state: {})
        builder.add_node(
            "synthesize",
            node(
                "synthesize",
                lambda state: request(state, ("Step results", plan_results(state))),
                context.writer(_SYNTHESIS, name="agloom_planner_synthesize"),
                lambda state, message: answer_update(message),
            ),
        )
        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", dispatch, ["step", "synthesize"])
        builder.add_edge("step", "join")
        builder.add_conditional_edges("join", dispatch, ["step", "synthesize"])
        builder.add_edge("synthesize", END)
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
