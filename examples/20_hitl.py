"""Human in the loop: review tool calls before they run; breakpoints for debugging.

To let a person approve, edit, or reject tool calls, add LangChain's
`HumanInTheLoopMiddleware(interrupt_on={...})` and a checkpointer. A call that
needs review pauses the run with `ExecutionPaused`; its payload lists the
`action_requests` (tool name and arguments). `agent.resume(config,
{"decisions": [...]})` continues with `approve`, `edit` (an `edited_action`),
or `reject` (a message for the model). For approvals bound to exact arguments
with exactly-once execution, use the `PolicyManager` (examples 41-44).

Static breakpoints (`interrupt_before`/`interrupt_after` with one explicit
`pattern`) are debugging stops between topology nodes: the run pauses with a
`BREAKPOINT_INTERRUPT` and `agent.resume(config)` continues it.

Run:
    uv run python examples/20_hitl.py
"""

from typing import Any

from _support import real_model, verified_text
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from agloom import BREAKPOINT_INTERRUPT, Agent, ExecutionPaused, create_agent

orders: list[tuple[str, int]] = []


def restock(sku: str, quantity: int) -> str:
    """Order more units of a product from the supplier."""
    orders.append((sku, quantity))
    return f"Purchase order placed for {quantity} x {sku}."


def authorize(thread_id: str) -> bool:
    """Allow only this example's threads."""
    return thread_id.startswith("hitl-")


def pending_review(agent: Agent, thread_id: str, request: str) -> dict[str, Any]:
    """Run a request until a tool call waits for review; return that call."""
    try:
        agent.invoke(request, {"configurable": {"thread_id": thread_id}})
    except ExecutionPaused as paused:
        (call,) = paused.interrupts[0].value["action_requests"]
        print(f"Review requested: {call['name']} {call['args']}")
        return call
    raise AssertionError("the tool call should wait for review")


def review_tool_calls() -> None:
    agent = create_agent(
        model=real_model(),
        tools=[restock],
        pattern="react",
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"restock": True})],
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=authorize,
    )

    pending_review(agent, "hitl-approve", "Restock 40 units of SKU-778.")
    agent.resume(
        {"configurable": {"thread_id": "hitl-approve"}},
        {"decisions": [{"type": "approve"}]},
    )
    assert orders == [("SKU-778", 40)], orders

    call = pending_review(agent, "hitl-edit", "Restock 500 units of SKU-901.")
    edited = {"name": call["name"], "args": {**call["args"], "quantity": 200}}
    agent.resume(
        {"configurable": {"thread_id": "hitl-edit"}},
        {"decisions": [{"type": "edit", "edited_action": edited}]},
    )
    assert orders[-1] == ("SKU-901", 200), orders

    pending_review(agent, "hitl-reject", "Restock 1000 units of SKU-123.")
    answer = agent.resume(
        {"configurable": {"thread_id": "hitl-reject"}},
        {
            "decisions": [
                {
                    "type": "reject",
                    "message": "Rejected: purchasing is frozen until Monday. "
                    "Do not retry; tell the requester.",
                }
            ]
        },
    )
    assert all(sku != "SKU-123" for sku, _ in orders), orders
    print("Approved, edited to 200 units, and rejected:", orders)
    print(verified_text(answer, "Monday"))


def debug_with_a_breakpoint() -> None:
    agent = create_agent(
        model=real_model(),
        pattern="reflection",
        interrupt_after=["draft"],
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=authorize,
    )
    config = {"configurable": {"thread_id": "hitl-announcement"}}

    try:
        agent.invoke(
            "Write a two-sentence announcement that the office closes at 3 PM "
            "on Friday for maintenance.",
            config,
        )
    except ExecutionPaused as paused:
        stop = paused
    else:
        raise AssertionError("the run should stop after the draft")
    assert stop.interrupts[0].value["type"] == BREAKPOINT_INTERRUPT
    print(f"Stopped after the draft on thread {stop.thread_id}")

    answer = agent.resume(config)
    print(verified_text(answer, ("3 PM", "3:00", "3pm", "15:00"), "Friday"))


def main() -> None:
    review_tool_calls()
    debug_with_a_breakpoint()


if __name__ == "__main__":
    main()
