"""Supervisor: a coordinator delegating to specialist workers in a bounded loop."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from pydantic import Field, create_model
from typing_extensions import TypedDict

from agloom.errors import ConfigurationError
from agloom.models import TopologyName
from agloom.topology._shared import (
    State,
    agent_input,
    agent_text,
    answer_update,
    choices,
    model_input,
    node,
    node_name,
    request,
    transcript,
    worker_instruction,
)
from agloom.topology.base import TopologyArtifact, TopologyBuildContext, compose_prompt

FINISH = "FINISH"

_INSTRUCTION = """\
You coordinate a team of workers to fully answer the request.
Workers:
{roster}
Choose the single worker who should act next and give them a precise
assignment, or choose FINISH once the completed work fully answers the request
and write the complete final answer. You can delegate at most {limit} times.
Never ask a worker to repeat completed work."""

_SYNTHESIS = "Write the complete final answer to the request from the workers' results."


class SupervisorState(TypedDict, total=False):
    """State of the Supervisor topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    contributions: Annotated[list[dict[str, str]], operator.add]
    handoffs: int
    next: str
    assignment: str
    output: str
    loop: dict[str, Any]


class SupervisorTopology:
    """A supervisor routes work to tool-capable workers until the request is answered.

    Graph nodes: `supervisor`, one `worker_<i>_<name>` node per worker, and
    `synthesize`. Delegations are bounded by `ExecutionLimits.max_handoffs`.
    """

    name = TopologyName.SUPERVISOR

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Supervisor graph."""
        workers = context.workers
        if not workers:
            raise ConfigurationError(
                "the supervisor topology requires at least one worker"
            )
        if any(worker.name == FINISH for worker in workers):
            raise ConfigurationError(f"{FINISH!r} is reserved and cannot name a worker")
        limit = context.limits.max_handoffs
        route = create_model(
            "SupervisorRoute",
            next=(
                choices([*(worker.name for worker in workers), FINISH]),
                Field(description="The worker that acts next, or FINISH."),
            ),
            assignment=(
                str,
                Field(
                    default="",
                    description="Precise assignment for the selected worker.",
                ),
            ),
            final_answer=(
                str,
                Field(
                    default="", description="The complete final answer, with FINISH."
                ),
            ),
        )
        router = context.decision(route, name="agloom_supervisor_route")
        roster = "\n".join(
            f"- {worker.name}: {worker.description}" for worker in workers
        )
        system = compose_prompt(
            context.system_prompt, _INSTRUCTION.format(roster=roster, limit=limit)
        )
        ids = {
            worker.name: node_name("worker", index, worker.name)
            for index, worker in enumerate(workers)
        }

        def decide(state: State, decision: Any) -> dict[str, Any]:
            if decision.next == FINISH:
                answer = "" if context.screen_answers else decision.final_answer
                return {"next": FINISH, "output": answer.strip(), "loop": {}}
            return {"next": decision.next, "assignment": decision.assignment}

        builder = StateGraph[Any](SupervisorState)
        builder.add_node(
            "supervisor",
            node(
                "supervisor",
                lambda state: model_input(
                    system,
                    request(
                        state,
                        (
                            "Completed work",
                            transcript(state.get("contributions") or []),
                        ),
                        (
                            "Remaining delegations",
                            str(limit - state.get("handoffs", 0)),
                        ),
                    ),
                ),
                router,
                decide,
                skip=lambda state: (
                    {"next": FINISH} if state.get("handoffs", 0) >= limit else None
                ),
            ),
        )
        for worker in workers:
            agent = context.agent(
                worker_instruction(worker), name=f"agloom_{ids[worker.name]}"
            )
            builder.add_node(
                ids[worker.name],
                node(
                    ids[worker.name],
                    lambda state: agent_input(
                        request(
                            state,
                            ("Your assignment", state.get("assignment")),
                            (
                                "Completed work",
                                transcript(state.get("contributions") or []),
                            ),
                        )
                    ),
                    agent,
                    lambda state, result, author=worker.name: {
                        "contributions": [
                            {"author": author, "text": agent_text(result)}
                        ],
                        "handoffs": state.get("handoffs", 0) + 1,
                    },
                ),
            )
            builder.add_edge(ids[worker.name], "supervisor")
        builder.add_node(
            "synthesize",
            node(
                "synthesize",
                lambda state: request(
                    state,
                    ("Worker results", transcript(state.get("contributions") or [])),
                ),
                context.writer(_SYNTHESIS, name="agloom_supervisor_synthesize"),
                lambda state, message: answer_update(message),
                skip=lambda state: {} if state.get("output") else None,
            ),
        )
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            lambda state: ids.get(state.get("next", FINISH), "synthesize"),
            [*ids.values(), "synthesize"],
        )
        builder.add_edge("synthesize", END)
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
