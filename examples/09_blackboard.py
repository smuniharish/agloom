"""Blackboard: specialists contribute to a shared workspace in parallel.

Every contributor reads the request and writes its findings to a shared
blackboard; a synthesis step combines them into one answer. Contributors run
concurrently (bounded by `ExecutionLimits.max_workers`). Use it for
multi-perspective analysis, such as cost, latency, and risk reviews.

Run:
    uv run python examples/09_blackboard.py
"""

from _support import real_model, verified_text

from agloom import WorkerSpec, create_agent


def main() -> None:
    agent = create_agent(
        model=real_model(),
        pattern="blackboard",
        workers=[
            WorkerSpec(
                name="latency_analyst",
                description="Assesses only the latency figures provided.",
            ),
            WorkerSpec(
                name="cost_analyst",
                description="Assesses only the per-request costs provided.",
            ),
        ],
    )

    answer = agent.invoke(
        "Compare two API options using only these facts: option A has 120 ms "
        "latency and costs $0.02 per request; option B has 200 ms latency and "
        "costs $0.01 per request. Summarize the trade-off and quote every figure."
    )

    print(verified_text(answer, "120", "200", "0.02", "0.01"))


if __name__ == "__main__":
    main()
