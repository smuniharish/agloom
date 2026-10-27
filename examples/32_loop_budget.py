"""Budgets: limits on tool calls, iterations, and active time.

`LoopPolicy` bounds every loop: `max_tool_calls` (shared by all iterations and
checked before each tool runs), `max_iterations`, `max_duration` (active
seconds, checked when each iteration ends; waiting for a human does not
count), `max_retries`, and `max_replans`. Exhausting a budget ends an
unfinished loop with a precise termination reason instead of a runaway agent;
a result that already satisfies the objective still completes.

Run:
    uv run python examples/32_loop_budget.py
"""

from _support import real_model

from agloom import (
    DefaultLoopEvaluator,
    LoopBudgetExceededError,
    LoopPolicy,
    LoopTerminationReason,
    create_agent,
)

lookups: list[str] = []


def stock_level(sku: str) -> str:
    """Return the units in stock for one product."""
    lookups.append(sku)
    return f"{sku}: 12 units in stock."


def main() -> None:
    model = real_model()

    tool_budget = create_agent(
        model=model,
        tools=[stock_level],
        pattern="react",
        loop_policy=LoopPolicy(max_tool_calls=2),
    ).invoke_loop(
        "Check the stock of SKU-1, SKU-2, SKU-3, and SKU-4, one tool call per SKU."
    )
    assert tool_budget.state.termination_reason is LoopTerminationReason.TOOL_CALL_LIMIT
    assert isinstance(tool_budget.error, LoopBudgetExceededError)
    assert len(lookups) == 2, lookups
    print(f"Tool-call budget: stopped after {len(lookups)} calls ({tool_budget.error})")

    iteration_budget = create_agent(
        model=model,
        loop_policy=LoopPolicy(max_iterations=2),
        loop_evaluator=DefaultLoopEvaluator(),
    ).invoke_loop("Suggest a name for an internal code-review bot.")
    state = iteration_budget.state
    assert state.termination_reason is LoopTerminationReason.ITERATION_LIMIT
    assert state.iteration == 2
    print(f"Iteration budget: stopped after {state.iteration} iterations")

    time_budget = create_agent(
        model=model,
        loop_policy=LoopPolicy(max_iterations=5, max_duration=1.0),
        loop_evaluator=DefaultLoopEvaluator(),
    ).invoke_loop("Write a haiku about deadlines.")
    state = time_budget.state
    assert state.termination_reason is LoopTerminationReason.TIMEOUT
    assert state.iteration == 1
    print(f"Duration budget: stopped after {state.budget.elapsed:.1f}s of 1.0s")


if __name__ == "__main__":
    main()
