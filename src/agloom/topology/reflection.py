"""Reflection: draft, structured review, and revision with early exit."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agloom.models import TopologyName
from agloom.topology._shared import (
    State,
    agent_input,
    answer_update,
    model_input,
    node,
    request,
)
from agloom.topology.base import TopologyArtifact, TopologyBuildContext, compose_prompt

_DRAFT = "Write a complete, correct answer to the request."
_REVIEW = (
    "Review the answer against the request. Approve it only when it is complete, "
    "correct, and needs no change; otherwise list concrete issues to fix."
)
_REVISE = (
    "Revise the answer so that every listed issue is resolved. Return the full answer."
)


class Review(BaseModel):
    """The verdict of one review round."""

    approved: bool = Field(description="True when the answer needs no change.")
    issues: list[str] = Field(
        default_factory=list, description="Concrete issues to fix; empty when approved."
    )


class ReflectionState(TypedDict, total=False):
    """State of the Reflection topology."""

    task: str
    messages: list[AnyMessage]
    context: str | None
    output: str
    loop: dict[str, Any]
    approved: bool
    issues: list[str]
    rounds: int


class ReflectionTopology:
    """Drafts an answer, reviews it, and revises until approved or out of rounds.

    Graph nodes: `draft`, `review`, and `revise`. The run ends as soon as a
    review approves the answer, and after at most
    `max_reflections` review-and-revise rounds.
    """

    name = TopologyName.REFLECTION

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the Reflection graph."""
        rounds = context.max_reflections
        reviewer = context.decision(Review, name="agloom_reflection_review")
        drafter = context.agent(_DRAFT, name="agloom_reflection_draft")
        reviser = context.agent(_REVISE, name="agloom_reflection_revise")
        review_prompt = compose_prompt(context.system_prompt, _REVIEW)

        def after_review(state: State) -> str:
            return END if state.get("approved") else "revise"

        def after_revise(state: State) -> str:
            return END if state.get("rounds", 0) >= rounds else "review"

        builder = StateGraph[Any](ReflectionState)
        builder.add_node(
            "draft",
            node(
                "draft",
                lambda state: agent_input(request(state)),
                drafter,
                lambda state, result: answer_update(result),
            ),
        )
        builder.add_node(
            "review",
            node(
                "review",
                lambda state: model_input(
                    review_prompt,
                    request(state, ("Answer to review", state.get("output"))),
                ),
                reviewer,
                lambda state, review: {
                    "approved": review.approved,
                    "issues": list(review.issues),
                    "rounds": state.get("rounds", 0) + 1,
                },
            ),
        )
        builder.add_node(
            "revise",
            node(
                "revise",
                lambda state: agent_input(
                    request(
                        state,
                        ("Current answer", state.get("output")),
                        (
                            "Issues to fix",
                            "\n".join(
                                f"- {issue}" for issue in state.get("issues") or []
                            ),
                        ),
                    )
                ),
                reviser,
                lambda state, result: answer_update(result),
            ),
        )
        builder.add_edge(START, "draft")
        builder.add_edge("draft", "review")
        builder.add_conditional_edges("review", after_review, ["revise", END])
        builder.add_conditional_edges("revise", after_revise, ["review", END])
        return TopologyArtifact(self.name, builder.compile(**context.compile_options()))
