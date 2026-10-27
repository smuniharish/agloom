"""Swarm: peers hand control to each other without a central coordinator.

Each peer contributes within its scope and makes a structured handoff decision:
the peer that continues, or STOP with the final answer. The first worker
starts; `ExecutionLimits.max_handoffs` bounds the conversation.

Run:
    uv run python examples/08_swarm.py
"""

from _support import real_model, verified_text

from agloom import ExecutionLimits, WorkerSpec, create_agent


def main() -> None:
    agent = create_agent(
        model=real_model(),
        pattern="swarm",
        workers=[
            WorkerSpec(
                name="converter",
                description="Converts temperatures and shows the formula used.",
                instructions="After converting, hand off to the reviewer.",
            ),
            WorkerSpec(
                name="reviewer",
                description="Checks conversions and writes the final answer.",
                instructions="Verify the arithmetic, then STOP with the answer.",
            ),
        ],
        limits=ExecutionLimits(max_handoffs=4),
    )

    answer = agent.invoke(
        "Convert 392 degrees Fahrenheit to Celsius and give the verified result."
    )

    print(verified_text(answer, "392", "200"))


if __name__ == "__main__":
    main()
