"""Custom policies: bring your own rules as Python objects or versioned JSON files.

Policies belong to the application. Write them in Python (`PolicyRule`,
`ArgumentCondition`) or keep them as reviewed JSON files and pass the path
(`examples/policies/procurement.json`). Consequential tools take a business
idempotency key: an argument name or, as here, a callable deriving it from
the arguments. With `on_denied="inform_model"`, a denied call is returned to
the model as an error instead of failing the run, so it can explain or adjust.

Run:
    uv run python examples/42_policy_custom_rules.py
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _support import real_model, verified_text
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)

from agloom import AgentEvent, EventName, PolicyManager, create_agent

POLICY_FILE = Path(__file__).with_name("policies") / "procurement.json"

PROCUREMENT_POLICY = RuntimePolicy(
    version="procurement-2026-10",
    rules=(
        PolicyRule(
            "approved vendors only",
            PolicyEffect.DENY,
            tool="create_purchase_order",
            conditions=(
                ArgumentCondition(
                    "vendor", ConditionOperator.NOT_IN, ["Globex", "Initech"]
                ),
            ),
        ),
        PolicyRule(
            "review orders over 5000",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="create_purchase_order",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 5000),),
            approval_ttl_seconds=86400,
        ),
        PolicyRule(
            "allow purchase orders", PolicyEffect.ALLOW, tool="create_purchase_order"
        ),
        PolicyRule("allow invoice reads", PolicyEffect.ALLOW, tool="lookup_*"),
    ),
)
INVOICES = {"U-77": ("Umbrella Corp", 900.0), "G-1042": ("Globex", 1200.0)}
orders: list[str] = []


def lookup_invoice(invoice: str) -> str:
    """Return the vendor and amount of a vendor invoice."""
    vendor, amount = INVOICES[invoice]
    return f"Invoice {invoice}: vendor {vendor}, amount ${amount:.2f}."


def create_purchase_order(vendor: str, invoice: str, amount: float) -> str:
    """Create a purchase order that pays a vendor invoice."""
    orders.append(f"{vendor}:{invoice}")
    return f"Purchase order created: {vendor}, invoice {invoice}, ${amount:.2f}."


def invoice_key(arguments: Mapping[str, Any]) -> str:
    """One purchase order per vendor invoice, however often it is requested."""
    return f"{arguments['vendor']}:{arguments['invoice']}".casefold()


def main() -> None:
    assert RuntimePolicy.from_file(POLICY_FILE) == PROCUREMENT_POLICY
    events: list[AgentEvent] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        store = SQLiteRuntimeStore(Path(folder) / "agentbarrier.db")
        agent = create_agent(
            model=real_model(),
            tools=[lookup_invoice, create_purchase_order],
            pattern="react",
            policy=PolicyManager(
                policy=POLICY_FILE,
                store=store,
                consequential={"create_purchase_order": invoice_key},
                namespace="procurement",
                organization_id="acme",
                on_denied="inform_model",
            ),
            observers=[events.append],
        )

        answer = agent.invoke("Pay invoice U-77 by creating its purchase order.")
        denied = [
            e
            for e in events
            if e.name == EventName.TOOL_CALL_DENIED
            and e.payload["tool"] == "create_purchase_order"
        ]
        assert denied, "the policy must deny the unapproved vendor"
        assert not orders, orders
        print(f"Policy denial ({denied[0].payload['reason']}) returned to the model.")
        print("Unapproved vendor:", verified_text(answer, "Umbrella"))

        answer = agent.invoke("Pay invoice G-1042 by creating its purchase order.")
        assert orders == ["Globex:G-1042"], orders
        print("Approved vendor:", verified_text(answer, "G-1042"))

        (action,) = [a for a in store.list_actions() if a.result_available]
        assert action.idempotency_key == "globex:g-1042", action.idempotency_key
        agent.execute_capability("create_purchase_order", dict(action.arguments))
        assert orders == ["Globex:G-1042"], "an invoice is never ordered twice"
        print(
            f"Recorded under key {action.idempotency_key!r} by rule "
            f"{action.policy_rule!r} ({action.policy_version}); a repeat replayed."
        )
        store.close()


if __name__ == "__main__":
    main()
