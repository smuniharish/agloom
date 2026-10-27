"""Retry: recover from a transient failure within a retry budget.

When an iteration fails (here a rates service returns 503 once), the loop
retries if `LoopPolicy(allow_retry=True)` and retries remain. The next
attempt receives the failure as context. Policy denials, budget exhaustion,
and unknown outcomes are never retried. Without retries, the run fails fast.

Run:
    uv run python examples/28_loop_retry.py
"""

from _support import real_model, verified_text

from agloom import (
    LoopAction,
    LoopPolicy,
    LoopTerminationReason,
    ModelLoopEvaluator,
    create_agent,
)

requests: list[str] = []


def exchange_rate(base: str, quote: str) -> str:
    """Return the current exchange rate between two currencies."""
    requests.append(f"{base}/{quote}")
    if len(requests) == 1:
        raise ConnectionError("rates service unavailable (HTTP 503)")
    return f"1 {base} = 1.0842 {quote}"


def main() -> None:
    model = real_model()
    agent = create_agent(
        model=model,
        tools=[exchange_rate],
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=3, allow_retry=True, max_retries=2),
        loop_evaluator=ModelLoopEvaluator(
            model, criteria="Complete when the answer states the EUR/USD rate."
        ),
    )

    result = agent.invoke_loop("What is the current EUR to USD exchange rate?")

    state = result.state
    failed, recovered = state.history[0], state.history[1]
    assert failed.observation.error_type == "ConnectionError"
    assert failed.decision.action is LoopAction.RETRY
    assert recovered.observation.success
    assert state.budget.retries == 1
    assert state.termination_reason is LoopTerminationReason.OBJECTIVE_SATISFIED
    print(f"Attempt 1 failed with {failed.observation.error_message}; retried.")
    print(verified_text(result.result, "1.0842"))


if __name__ == "__main__":
    main()
