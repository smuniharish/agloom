"""Policy store on PostgreSQL: shared approvals, exactly-once effects, reconciliation.

In production, keep agentbarrier's runtime state in PostgreSQL with
`agentbarrier.runtime.PostgresRuntimeStore` (install `agloom[postgres]`, or
`agloom[postgres-binary]` on hosts without a PostgreSQL client library).
Every service instance, reviewer, and operator then share approvals,
execution claims, and the tamper-evident receipt chain:

* A wire over $10,000 waits for approval (`APPROVAL_INTERRUPT`). A reviewer
  approves it through their own connection, and the resumed run sends the
  wire exactly once.
* When a consequential tool fails, its outcome is unknown: did the bank book
  the wire before the connection failed? The run pauses with a
  `RECONCILIATION_INTERRUPT` until an operator checks with the bank and
  reconciles the action through their own connection. `agent.resume(config)`
  then retries the call. A committed action replays the operator's evidence
  to the model; one that was not committed runs exactly once. Either way the
  bank books the wire once. Without a checkpointer, and from
  `PolicyManager.execute`, the failure raises agentbarrier's
  `ActionOutcomeUnknown` with the action instead.

Run (with a PostgreSQL URL in `AGLOOM_POSTGRES_URL`):
    uv run python examples/44_policy_postgres.py
"""

import os
import uuid

from _support import real_model, verified_text
from agentbarrier import Decision
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    PostgresRuntimeStore,
    RuntimePolicy,
    RuntimeReconciliation,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agloom import (
    APPROVAL_INTERRUPT,
    RECONCILIATION_INTERRUPT,
    Agent,
    ExecutionPaused,
    PolicyManager,
    create_agent,
)

SCHEMA = "agloom_examples"
STATEMENT = "BS-4471"
TREASURY_POLICY = RuntimePolicy(
    version="treasury-2026-10",
    rules=(
        PolicyRule(
            "review wires over $10,000",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="send_wire",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 10_000),),
        ),
        PolicyRule("allow wires", PolicyEffect.ALLOW, tool="send_wire"),
    ),
)
wires: list[str] = []
"""Wires the bank booked."""
faults: dict[str, str] = {}
"""One-off bank connection faults by transfer id."""


def send_wire(transfer_id: str, beneficiary: str, amount: float) -> str:
    """Send a bank wire. The transfer_id makes the wire idempotent."""
    fault = faults.pop(transfer_id, None)
    if fault == "lost request":
        raise ConnectionError("the request did not reach the bank")
    wires.append(transfer_id)
    if fault == "lost confirmation":
        raise TimeoutError("the bank did not confirm the wire in time")
    return f"Wire {transfer_id} of ${amount:,.2f} sent to {beneficiary}."


def new_transfer() -> tuple[str, RunnableConfig]:
    transfer_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
    return transfer_id, {"configurable": {"thread_id": f"treasury-{transfer_id}"}}


def approve_a_large_wire(agent: Agent, store: PostgresRuntimeStore, url: str) -> None:
    transfer_id, config = new_transfer()
    try:
        agent.invoke(
            f"Wire $12,500 to Globex Ltd with transfer id {transfer_id}.", config
        )
    except ExecutionPaused as paused:
        approval = paused.interrupts[0].value
    else:
        raise AssertionError("wires over $10,000 must wait for approval")
    assert approval["type"] == APPROVAL_INTERRUPT, approval
    print(f"Awaiting approval: {approval['tool']} {approval['arguments']}")

    reviewer = PostgresRuntimeStore(url, schema=SCHEMA)
    reviewer.decide(approval["action_id"], Decision.APPROVE, decided_by="treasury")
    reviewer.close()

    answer = agent.resume(config)
    print(verified_text(answer, transfer_id))
    assert wires.count(transfer_id) == 1, wires
    receipts = store.receipts(action_id=approval["action_id"])
    print(f"Executed once with {len(receipts)} audit receipts.")


def reconcile_an_unknown_outcome(agent: Agent, url: str, *, booked: bool) -> None:
    transfer_id, config = new_transfer()
    faults[transfer_id] = "lost confirmation" if booked else "lost request"
    try:
        agent.invoke(
            f"Wire $2,500 to Initech with transfer id {transfer_id} and quote "
            "the bank's confirmation.",
            config,
        )
    except ExecutionPaused as paused:
        unknown = paused.interrupts[0].value
    else:
        raise AssertionError("a failed wire must wait for reconciliation")
    assert unknown["type"] == RECONCILIATION_INTERRUPT, unknown
    print(f"Outcome unknown after {unknown['error']}: {unknown['arguments']}")

    # The operator checks with the bank, then records what it found.
    operator = PostgresRuntimeStore(url, schema=SCHEMA)
    if booked:
        operator.reconcile(
            unknown["action_id"],
            RuntimeReconciliation.COMMITTED,
            resolved_by="treasury-ops",
            reason=f"Bank statement {STATEMENT} lists the wire.",
            result=f"Wire {transfer_id} confirmed on bank statement {STATEMENT}.",
        )
    else:
        operator.reconcile(
            unknown["action_id"],
            RuntimeReconciliation.NOT_COMMITTED,
            resolved_by="treasury-ops",
            reason="The bank has no record of the wire.",
        )
    operator.close()

    answer = agent.resume(config)
    print(verified_text(answer, STATEMENT if booked else transfer_id))
    assert wires.count(transfer_id) == 1, wires


def main() -> None:
    url = os.environ.get("AGLOOM_POSTGRES_URL")
    if not url:
        raise SystemExit(
            "Set AGLOOM_POSTGRES_URL to a PostgreSQL URL, for example "
            "postgresql://agloom@localhost:5432/agloom"
        )
    store = PostgresRuntimeStore(url, schema=SCHEMA, create_schema=True, migrate=True)
    agent = create_agent(
        model=real_model(),
        tools=[send_wire],
        pattern="react",
        policy=PolicyManager(
            policy=TREASURY_POLICY,
            store=store,
            consequential={"send_wire": "transfer_id"},
            namespace="treasury",
            organization_id="acme",
        ),
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=lambda thread_id: thread_id.startswith("treasury-"),
    )

    approve_a_large_wire(agent, store, url)
    reconcile_an_unknown_outcome(agent, url, booked=True)
    reconcile_an_unknown_outcome(agent, url, booked=False)
    assert store.verify_receipt_chain()
    print("Receipt chain verified.")
    store.close()


if __name__ == "__main__":
    main()
