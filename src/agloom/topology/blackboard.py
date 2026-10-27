"""Blackboard: parallel contributions to a shared board, then synthesis."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agloom.errors import ConfigurationError
from agloom.models import TopologyName
from agloom.topology._shared import (
    State,
    agent_input,
    agent_text,
    answer_update,
    merge_dicts,
    node,
    node_name,
    request,
    worker_instruction,
)
from agloom.topology.base import TopologyArtifact, TopologyBuildContext, compose_prompt

_CONTRIBUTE = "Contribute your findings to the shared board, staying within your scope."
_SYNTHESIS = (
    "Combine the board contributions into one complete answer to the request. "
    "Resolve conflicting contributions explicitly."
)


class BlackboardState(TypedDict, total=False):
    """State of the Blackboard topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    board: Annotated[dict[str, str], merge_dicts]
    output: str
    loop: dict[str, Any]


class BlackboardTopology:
    """Workers contribute to a shared board in parallel; a synthesis combines them.

    Graph nodes: one `contributor_<i>_<name>` node per worker and `synthesize`.
    Contributors run concurrently, bounded by `ExecutionLimits.max_workers`.
    """

    name = TopologyName.BLACKBOARD

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Blackboard graph."""
        workers = context.workers
        if not workers:
            raise ConfigurationError(
                "the blackboard topology requires at least one worker"
            )
        order = [worker.name for worker in workers]

        def board_text(state: State) -> str:
            board = state.get("board") or {}
            return "\n\n".join(
                f"[{name}] {board[name]}" for name in order if name in board
            )

        builder = StateGraph[Any](BlackboardState)
        contributors = []
        for index, worker in enumerate(workers):
            contributor_id = node_name("contributor", index, worker.name)
            agent = context.agent(
                compose_prompt(worker_instruction(worker), _CONTRIBUTE) or "",
                name=f"agloom_{contributor_id}",
            )
            builder.add_node(
                contributor_id,
                node(
                    contributor_id,
                    lambda state: agent_input(request(state)),
                    agent,
                    lambda state, result, author=worker.name: {
                        "board": {author: agent_text(result)}
                    },
                ),
            )
            contributors.append(contributor_id)
        # At most max_workers contributors run at once; each group starts when the
        # previous group has finished.
        size = context.limits.max_workers
        previous: str | list[str] = START
        for start in range(0, len(contributors), size):
            group = contributors[start : start + size]
            for contributor_id in group:
                builder.add_edge(previous, contributor_id)
            previous = group
        builder.add_node(
            "synthesize",
            node(
                "synthesize",
                lambda state: request(state, ("Shared board", board_text(state))),
                context.writer(_SYNTHESIS, name="agloom_blackboard_synthesize"),
                lambda state, message: answer_update(message),
            ),
        )
        builder.add_edge(previous, "synthesize")
        builder.add_edge("synthesize", END)
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
