from _support import real_model
from behaviorweave import (
    BehaviorEvent,
    InterventionType,
    PolicyRule,
)
from langchain_core.tools import tool

from agloom import create_agent


@tool
def search_orders(customer_id: str) -> str:
    """Return an order summary for a customer."""

    return f"{customer_id}: one order is ready to ship"


agent = create_agent(
    model=real_model(),
    tools=[search_orders],
    behavior_options={
        "policies": (
            PolicyRule(
                policy_id="pause-repeated-search",
                pattern_id="repeated_tool_call",
                threshold=3,
                intervention=InterventionType.PAUSE,
                message="Pause after three identical searches.",
            ),
        )
    },
)
engine = agent.capabilities.resolve("behavior_engine")

decisions = [
    (
        search_orders.invoke({"customer_id": "C-42"}),
        engine.process(
            BehaviorEvent.tool_call(
                "search_orders",
                {"customer_id": "C-42"},
                scope="support-agent",
            )
        ),
    )
    for _ in range(3)
]
tool_results = [result for result, _ in decisions]
policy_decisions = [decision for _, decision in decisions]
final = policy_decisions[-1]

assert all("ready to ship" in result for result in tool_results)
assert policy_decisions[0].intervention.kind is InterventionType.NOOP
assert policy_decisions[1].intervention.kind is InterventionType.NOOP
assert final.intervention.kind is InterventionType.PAUSE
assert final.explanation is not None
assert final.explanation.count == 3
print(
    f"Observed {final.explanation.count} repeated searches; "
    f"intervention: {final.intervention.kind.value}."
)
print(final.intervention.message)
