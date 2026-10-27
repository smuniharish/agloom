"""Behavior guardrails: BehaviorWeave stops runaway tool use and the loop obeys.

`behaviorweave.integrations.langchain.BehaviorWeaveMiddleware` watches every
tool call and outcome against behavior policies. With
`on_decision=agloom.behavior_signals`, its decisions also become Loop
Engineering signals: `stop` ends the loop, `pause`/`human_review` wait for a
human, `redirect` replans, and `retry` retries, always within the
`LoopPolicy`. Here a failing billing API trips a failure-streak policy.

Run:
    uv run python examples/16_behavior.py
"""

from _support import real_model
from behaviorweave import BehaviorEngine, InterventionType, PolicyRule
from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware
from langchain_core.tools import StructuredTool, ToolException

from agloom import (
    DefaultLoopEvaluator,
    LoopPolicy,
    LoopTerminationReason,
    behavior_signals,
    create_agent,
)

attempts: list[str] = []


def fetch_invoice(customer_id: str) -> str:
    """Fetch the latest invoice of a customer from the billing API."""
    attempts.append(customer_id)
    raise ToolException("billing API timed out after 30 seconds")


def main() -> None:
    engine = BehaviorEngine(
        policies=[
            PolicyRule(
                policy_id="stop-on-failing-billing",
                pattern_id="failure_streak",
                threshold=2,
                intervention=InterventionType.STOP,
                message="The billing API keeps failing; stop and escalate.",
            )
        ]
    )
    guard = BehaviorWeaveMiddleware(
        engine, scope="billing-assistant", on_decision=behavior_signals
    )
    agent = create_agent(
        model=real_model(),
        tools=[
            StructuredTool.from_function(fetch_invoice, handle_tool_error=True),
        ],
        pattern="react",
        middleware=[guard],
        loop_policy=LoopPolicy(max_iterations=3),
        loop_evaluator=DefaultLoopEvaluator(),
    )

    result = agent.invoke_loop(
        "Fetch the latest invoices of customers C-100 and C-200 and summarize them."
    )

    state = result.state
    signals = [s for entry in state.history for s in entry.observation.behavior_signals]
    assert "stop" in signals, signals
    assert state.termination_reason is LoopTerminationReason.BEHAVIOR_INTERVENTION
    assert len(attempts) >= 2, attempts
    print(f"Billing calls attempted: {len(attempts)}; behavior signals: {signals}")
    print(f"Loop stopped after {state.iteration} iteration(s): {state.decision.reason}")


if __name__ == "__main__":
    main()
