"""Swarm: peers that hand control to each other through structured handoffs."""

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
    answer_update,
    choices,
    node,
    node_name,
    request,
    structured_response,
    transcript,
    worker_instruction,
)
from agloom.topology.base import TopologyArtifact, TopologyBuildContext, compose_prompt

STOP = "STOP"

_INSTRUCTION = """\
You work in a swarm of peers:
{roster}
Contribute within your scope, then hand off to the peer best suited to continue,
or choose STOP once the request is fully answered and write the complete final
answer. Never repeat work another peer completed."""

_SYNTHESIS = "Write the complete final answer to the request from the peers' work."


class SwarmState(TypedDict, total=False):
    """State of the Swarm topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    contributions: Annotated[list[dict[str, str]], operator.add]
    handoffs: int
    active: str
    output: str
    loop: dict[str, Any]


class SwarmTopology:
    """Decentralized peers work in turn and choose who continues.

    Graph nodes: one `peer_<i>_<name>` node per worker and `finish`. The first
    worker starts. Handoffs are bounded by `ExecutionLimits.max_handoffs`.
    """

    name = TopologyName.SWARM

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Swarm graph."""
        workers = context.workers
        if not workers:
            raise ConfigurationError("the swarm topology requires at least one worker")
        if any(worker.name == STOP for worker in workers):
            raise ConfigurationError(f"{STOP!r} is reserved and cannot name a worker")
        limit = context.limits.max_handoffs
        turn = create_model(
            "SwarmTurn",
            contribution=(str, Field(description="Your contribution to the request.")),
            next=(
                choices([*(worker.name for worker in workers), STOP]),
                Field(description="The peer that continues, or STOP."),
            ),
            final_answer=(
                str,
                Field(default="", description="The complete final answer, with STOP."),
            ),
        )
        roster = "\n".join(
            f"- {worker.name}: {worker.description}" for worker in workers
        )
        ids = {
            worker.name: node_name("peer", index, worker.name)
            for index, worker in enumerate(workers)
        }

        def advance(state: State, result: Any, author: str) -> dict[str, Any]:
            decision = structured_response(result)
            update: dict[str, Any] = {
                "contributions": [{"author": author, "text": decision.contribution}],
                "handoffs": state.get("handoffs", 0) + 1,
                "active": decision.next,
            }
            if (
                decision.next == STOP
                and decision.final_answer.strip()
                and not context.screen_answers
            ):
                update["output"] = decision.final_answer.strip()
            return update

        def route(state: State) -> str:
            active = state.get("active", STOP)
            if active == STOP or state.get("handoffs", 0) >= limit:
                return "finish"
            return ids[active]

        builder = StateGraph[Any](SwarmState)
        for worker in workers:
            peer = context.agent(
                compose_prompt(
                    worker_instruction(worker), _INSTRUCTION.format(roster=roster)
                )
                or "",
                name=f"agloom_{ids[worker.name]}",
                response_format=turn,
            )
            builder.add_node(
                ids[worker.name],
                node(
                    ids[worker.name],
                    lambda state: agent_input(
                        request(
                            state,
                            (
                                "Work so far",
                                transcript(state.get("contributions") or []),
                            ),
                        )
                    ),
                    peer,
                    lambda state, result, author=worker.name: advance(
                        state, result, author
                    ),
                ),
            )
            builder.add_conditional_edges(
                ids[worker.name], route, [*ids.values(), "finish"]
            )
        builder.add_node(
            "finish",
            node(
                "finish",
                lambda state: request(
                    state, ("Peer work", transcript(state.get("contributions") or []))
                ),
                context.writer(_SYNTHESIS, name="agloom_swarm_finish"),
                lambda state, message: answer_update(message),
                skip=lambda state: {} if state.get("output") else None,
            ),
        )
        builder.add_edge(START, ids[workers[0].name])
        builder.add_edge("finish", END)
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
