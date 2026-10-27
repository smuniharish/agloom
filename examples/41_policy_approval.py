"""Approvals: pause for a human, then execute exactly once, or never.

Declare consequential tools with a business idempotency key. A
`REQUIRE_APPROVAL` rule pauses the run with the action bound to its exact
arguments (`ExecutionPaused`, interrupt type `APPROVAL_INTERRUPT`). A reviewer
decides in agentbarrier (`store.decide(...)` here; also its CLI, API, or
dashboard), and `agent.resume()` executes an approved call exactly once;
repeating it replays the stored result. Rejected actions never run.

Run:
    uv run python examples/41_policy_approval.py
"""

import tempfile
from pathlib import Path
from typing import Any

from _support import real_model, verified_text
from agentbarrier import Decision
from agentbarrier.errors import ApprovalRejected
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langgraph.checkpoint.memory import InMemorySaver

from agloom import (
    APPROVAL_INTERRUPT,
    Agent,
    ExecutionPaused,
    PolicyManager,
    create_agent,
)

refunds: list[str] = []

REFUND_POLICY = RuntimePolicy(
    version="payments-2026-10",
    rules=(
        PolicyRule(
            "review refunds over $100",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="refund",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
        ),
        PolicyRule("allow small refunds", PolicyEffect.ALLOW, tool="refund"),
    ),
)


def refund(request_id: str, order_id: str, amount: float) -> str:
    """Refund an order. The request_id makes the refund idempotent."""
    refunds.append(request_id)
    return f"Refunded ${amount:.2f} for order {order_id} (request {request_id})."


def request_refund(agent: Agent, thread_id: str, request: str) -> dict[str, Any]:
    """Run a refund request and return the approval it waits for."""
    try:
        agent.invoke(request, {"configurable": {"thread_id": thread_id}})
    except ExecutionPaused as paused:
        approval = paused.interrupts[0].value
        assert approval["type"] == APPROVAL_INTERRUPT, approval
        return approval
    raise AssertionError("refunds over $100 must wait for approval")


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        store = SQLiteRuntimeStore(Path(folder) / "agentbarrier.db")
        agent = create_agent(
            model=real_model(),
            tools=[refund],
            pattern="react",
            policy=PolicyManager(
                policy=REFUND_POLICY,
                store=store,
                consequential={"refund": "request_id"},
                namespace="payments",
                organization_id="acme",
                requested_by="support-agent",
            ),
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=lambda thread_id: thread_id.startswith("refund-"),
        )

        approval = request_refund(
            agent, "refund-R-1", "Refund $250 for order A-1001, request id R-1."
        )
        print(f"Awaiting approval: {approval['tool']} {approval['arguments']}")
        store.decide(approval["action_id"], Decision.APPROVE, decided_by="alice")
        answer = agent.resume({"configurable": {"thread_id": "refund-R-1"}})
        print(verified_text(answer, "R-1"))
        assert refunds == ["R-1"], refunds

        replayed = agent.execute_capability("refund", approval["arguments"])
        assert refunds == ["R-1"], "a repeated approved action is replayed"
        print(f"Replayed without a second refund: {replayed['content']}")

        approval = request_refund(
            agent, "refund-R-2", "Refund $400 for order A-1002, request id R-2."
        )
        store.decide(
            approval["action_id"],
            Decision.REJECT,
            decided_by="alice",
            reason="Duplicate claim",
        )
        try:
            agent.resume({"configurable": {"thread_id": "refund-R-2"}})
        except ApprovalRejected as rejected:
            print(f"Rejected, never executed: {rejected}")
        else:
            raise AssertionError("a rejected refund must not run")
        assert refunds == ["R-1"], refunds
        store.close()


if __name__ == "__main__":
    main()
