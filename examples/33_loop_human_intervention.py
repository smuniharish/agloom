"""Human intervention: pause the loop for a person, resume with their decision.

With `LoopPolicy(allow_human_intervention=True)` and a checkpointer, an
iteration that needs a human pauses the loop durably (an interrupt of type
`LOOP_INTERRUPT`); `invoke_loop` returns the pause as data. The reviewer's
answer, passed to `agent.resume(config, "...")`, becomes feedback for the next
iteration. Here a risk rule (a `LoopObserver`) requires finance sign-off for
refunds above $500.

Run:
    uv run python examples/33_loop_human_intervention.py
"""

from typing import Any

from _support import real_model, verified_text
from langgraph.checkpoint.memory import InMemorySaver

from agloom import (
    LOOP_INTERRUPT,
    AgentEvent,
    DefaultLoopObserver,
    EventName,
    LoopObservation,
    LoopPolicy,
    create_agent,
)

drafts: dict[str, float] = {}


def draft_refund(order_id: str, amount: float) -> str:
    """Prepare (but do not send) a refund for an order."""
    drafts[order_id] = amount
    return f"Refund of ${amount:.2f} drafted for order {order_id}; not sent yet."


class FinanceSignOff(DefaultLoopObserver):
    """Refunds above $500 need a finance authorization (FIN-...) in the answer."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        large = any(amount > 500 for amount in drafts.values())
        if large and "FIN-" not in kwargs["result"].text:
            return observation.model_copy(update={"human_intervention_required": True})
        return observation.model_copy(update={"objective_complete": True})


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(),
        tools=[draft_refund],
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=3, allow_human_intervention=True),
        loop_observer=FinanceSignOff(),
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=lambda thread_id: thread_id.startswith("refund-"),
        observers=[events.append],
    )
    config = {"configurable": {"thread_id": "refund-A-2002"}}

    paused = agent.invoke_loop(
        "Order A-2002 ($640) arrived broken. Draft a full refund and tell the "
        "customer what happens next.",
        config,
    )
    request = paused.interrupts[0].value
    assert paused.paused
    assert request["type"] == LOOP_INTERRUPT, request
    assert request["action"] == "human_intervention", request
    print(f"Paused at iteration {request['iteration']}: {request['reason']}")

    answer = agent.resume(
        config,
        "Finance approved the $640 refund under authorization FIN-7731. Confirm "
        "it to the customer and cite the authorization.",
    )
    print(verified_text(answer, "FIN-7731"))
    completed = [e for e in events if e.name == EventName.LOOP_COMPLETED]
    assert completed[-1].payload["termination_reason"] == "objective_satisfied"


if __name__ == "__main__":
    main()
