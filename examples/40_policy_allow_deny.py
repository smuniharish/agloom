"""Policy: allow and deny tool calls with an agentbarrier runtime policy.

`PolicyManager` enforces an agentbarrier `RuntimePolicy` on every tool call:
by the model, inside any topology, and through `execute_capability`. Rules
match tool names (glob patterns) and argument conditions; the first matching
rule wins and the default effect is deny, so every tool must be allowed
explicitly. Tools denied for every argument are hidden from the model, and a
denied call raises `PolicyViolationError` before the tool runs.

Run:
    uv run python examples/40_policy_allow_deny.py
"""

import tempfile
from pathlib import Path

from _support import real_model, verified_text
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)

from agloom import PolicyManager, PolicyViolationError, create_agent

CUSTOMERS = {"C-100": "Berlin", "C-200": "Munich", "C-300": "Berlin"}
exports: list[str] = []

CRM_POLICY = RuntimePolicy(
    version="crm-2026-10",
    rules=(
        PolicyRule("allow searches", PolicyEffect.ALLOW, tool="search_*"),
        PolicyRule(
            "allow internal exports",
            PolicyEffect.ALLOW,
            tool="export_customers",
            conditions=(
                ArgumentCondition(
                    "recipient", ConditionOperator.ENDS_WITH, "@acme.example"
                ),
            ),
        ),
        PolicyRule("never delete customers", PolicyEffect.DENY, tool="delete_*"),
    ),
)


def search_customers(city: str) -> str:
    """Find customer IDs by city."""
    found = [cid for cid, home in CUSTOMERS.items() if home == city]
    return f"Customers in {city}: {', '.join(found) or 'none'}."


def export_customers(recipient: str) -> str:
    """Email the customer list to a recipient."""
    exports.append(recipient)
    return f"Customer list sent to {recipient}."


def delete_customer(customer_id: str) -> str:
    """Delete a customer record."""
    return f"Deleted {customer_id}."


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        store = SQLiteRuntimeStore(Path(folder) / "agentbarrier.db")
        agent = create_agent(
            model=real_model(),
            tools=[search_customers, export_customers, delete_customer],
            pattern="react",
            policy=PolicyManager(policy=CRM_POLICY, store=store),
        )
        assert "delete_customer" not in agent.capabilities.names, "hidden"

        answer = agent.invoke(
            "Which customers are in Berlin? Then email the customer list to "
            "analytics@acme.example."
        )
        print(verified_text(answer, "C-100", "C-300"))
        assert exports == ["analytics@acme.example"], exports

        try:
            agent.execute_capability(
                "export_customers", {"recipient": "lists@broker.example"}
            )
        except PolicyViolationError as denied:
            print(f"Denied before running: {denied}")
        else:
            raise AssertionError("external exports must be denied")
        assert exports == ["analytics@acme.example"], exports
        store.close()


if __name__ == "__main__":
    main()
