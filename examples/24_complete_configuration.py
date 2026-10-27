"""A complete, production-style configuration in one place.

A customer-support agent combining what production deployments usually need,
with every external component owned by the application:

* tools, automatic selection within an allow-list (ReAct, Planner,
  Reflection), and bounds on plan length and parallel fan-out;
* ContextSage middleware that keeps long conversations within budget;
* a PolicyManager: order lookups are allowed, refunds above $100 need a human
  approval and execute exactly once (agentbarrier), everything else is denied;
* a checkpointer with a tenant-aware thread authorizer;
* Loop Engineering with the model evaluator, a retry, and budgets;
* human feedback through feedback-manager;
* Prometheus metrics and an event callable.

Run:
    uv run python examples/24_complete_configuration.py
"""

import asyncio
import tempfile
from pathlib import Path

from _support import real_model, verified_text
from agentbarrier import Decision
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from contextsage import IntelligentSummarizationMiddleware
from feedback_manager import (
    FeedbackCategory,
    FeedbackManager,
    FeedbackSource,
    FeedbackStatus,
    FeedbackTarget,
    FeedbackTargetType,
)
from langgraph.checkpoint.memory import InMemorySaver
from prometheus_client import CollectorRegistry, generate_latest

from agloom import (
    APPROVAL_INTERRUPT,
    AgentEvent,
    EventName,
    ExecutionLimits,
    ExecutionPaused,
    LoopPolicy,
    PolicyManager,
    PrometheusObserver,
    create_agent,
)

TENANT = "acme"
ORDERS = {"A-1001": ("delivered damaged", 250.00)}
REFUNDS: list[str] = []

SUPPORT_POLICY = RuntimePolicy(
    version="support-2026-10",
    rules=(
        PolicyRule(
            "review large refunds",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="issue_refund",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
        ),
        PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="issue_refund"),
        PolicyRule("allow order lookups", PolicyEffect.ALLOW, tool="lookup_order"),
    ),
)


def lookup_order(order_id: str) -> str:
    """Return the delivery status and the total of an order."""
    status, total = ORDERS.get(order_id, ("unknown", 0.0))
    return f"Order {order_id}: {status}, total ${total:.2f}."


def issue_refund(request_id: str, order_id: str, amount: float) -> str:
    """Refund an order. The request_id makes the refund idempotent."""
    REFUNDS.append(request_id)
    return f"Refunded ${amount:.2f} for order {order_id} (request {request_id})."


def authorize(thread_id: str) -> bool:
    """Threads are named '<tenant>:<conversation>'; admit only this tenant."""
    return thread_id.startswith(f"{TENANT}:")


async def main() -> None:
    model = real_model()
    events: list[AgentEvent] = []
    metrics = CollectorRegistry()
    feedback = FeedbackManager()
    thread = f"{TENANT}:conversation-88"
    config = {"configurable": {"thread_id": thread}}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        store = SQLiteRuntimeStore(Path(folder) / "agentbarrier.db")
        agent = create_agent(
            model=model,
            tools=[lookup_order, issue_refund],
            system_prompt="You are Acme's support agent. Be concise and factual.",
            pattern=["react", "planner", "reflection"],
            limits=ExecutionLimits(max_plan_steps=4, max_workers=2),
            middleware=[
                IntelligentSummarizationMiddleware(
                    model, trigger=("tokens", 60_000), keep=("messages", 20)
                )
            ],
            policy=PolicyManager(
                policy=SUPPORT_POLICY,
                store=store,
                consequential={"issue_refund": "request_id"},
                namespace="support-agent",
                organization_id=TENANT,
                requested_by="support-agent",
            ),
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=authorize,
            loop_policy=LoopPolicy(
                max_iterations=2,
                allow_retry=True,
                max_retries=1,
                max_tool_calls=12,
                max_duration=240,
            ),
            feedback=feedback,
            observers=[PrometheusObserver(registry=metrics), events.append],
        )

        note = await feedback.submit(
            source=FeedbackSource.HUMAN,
            category=FeedbackCategory.COMMENT,
            target=FeedbackTarget(type=FeedbackTargetType.THREAD, id=thread),
            payload={"text": "Priority account: apologize for the damaged delivery."},
        )
        try:
            await agent.ainvoke(
                "Order A-1001 arrived damaged. Please refund the full order total "
                "with request id RF-1001.",
                config,
            )
        except ExecutionPaused as paused:
            approval = paused.interrupts[0].value
        else:
            raise AssertionError("a $250 refund must wait for approval")
        assert approval["type"] == APPROVAL_INTERRUPT, approval
        print(f"Approval needed: {approval['tool']} {approval['arguments']}")
        store.decide(approval["action_id"], Decision.APPROVE, decided_by="lead")

        answer = await agent.aresume(config)
        print(verified_text(answer, "250"))
        store.close()

    acknowledged = await feedback.get(note.feedback_id)
    names = [event.name for event in events]
    exposition = generate_latest(metrics).decode()
    assert REFUNDS == ["RF-1001"], REFUNDS
    assert EventName.APPROVAL_REQUIRED in names, names
    assert EventName.ACTION_EXECUTED in names, names
    assert acknowledged is not None
    assert acknowledged.status == FeedbackStatus.ACKNOWLEDGED
    assert 'tool="issue_refund"' in exposition
    print("Refund executed exactly once:", REFUNDS)


if __name__ == "__main__":
    asyncio.run(main())
