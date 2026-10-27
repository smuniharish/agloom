"""Supervisor: a coordinator delegates to specialist workers until the job is done.

The supervisor makes structured routing decisions (which worker acts next, with
what assignment, or FINISH with the final answer). Workers are tool-capable
agents with their own scope and instructions. `ExecutionLimits.max_handoffs`
bounds the delegations, so a confused coordinator cannot loop forever.

Run:
    uv run python examples/05_supervisor.py
"""

from _support import real_model, verified_text

from agloom import ExecutionLimits, WorkerSpec, create_agent


def main() -> None:
    agent = create_agent(
        model=real_model(),
        pattern="supervisor",
        workers=[
            WorkerSpec(
                name="analyst",
                description="Computes budget figures from the numbers provided.",
                instructions="Show the arithmetic and state the result.",
            ),
            WorkerSpec(
                name="writer",
                description="Writes short, plain-language status updates.",
            ),
        ],
        limits=ExecutionLimits(max_handoffs=4),
    )

    answer = agent.invoke(
        "A project budget is $1200 and $450 has been spent. Have the analyst "
        "compute the remaining budget, then have the writer turn it into a "
        "one-sentence status update."
    )

    print(verified_text(answer, ("$750", "750")))


if __name__ == "__main__":
    main()
