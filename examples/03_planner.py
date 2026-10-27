"""Planner: independent steps run in parallel waves, then one synthesis.

The Planner topology asks the model for a dependency-aware plan (validated as a
DAG), runs every step whose dependencies are met in parallel, and synthesizes
the step results. `ExecutionLimits` bounds the plan length and the fan-out.
Use it when a request splits into self-contained parts that can run at once.

Run:
    uv run python examples/03_planner.py
"""

from _support import real_model, verified_text

from agloom import ExecutionLimits, create_agent


def main() -> None:
    agent = create_agent(
        model=real_model(),
        pattern="planner",
        limits=ExecutionLimits(max_plan_steps=4, max_workers=3),
    )

    answer = agent.invoke(
        "Run three independent unit conversions as separate steps: 5 kilometers "
        "to meters, 2 hours to minutes, and 3 kilograms to grams. Report all "
        "three results as plain numbers with units."
    )

    print(verified_text(answer, ("5000", "5,000"), "120", ("3000", "3,000")))


if __name__ == "__main__":
    main()
