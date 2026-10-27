"""Reflection: draft, review against the request, revise until approved.

A drafter writes an answer, a reviewer returns a structured verdict (approved,
or concrete issues), and a reviser fixes the issues. The run stops as soon as
a review approves the answer, or after `max_reflections` rounds. Use it when
correctness matters more than latency.

Run:
    uv run python examples/07_reflection.py
"""

from _support import real_model, verified_text

from agloom import create_agent


def main() -> None:
    agent = create_agent(model=real_model(), pattern="reflection", max_reflections=2)

    answer = agent.invoke(
        "Write two accurate sentences for a school newsletter explaining that "
        "Earth takes about 365.25 days to orbit the Sun, and why that is the "
        "reason for leap years."
    )

    print(verified_text(answer, "365.25", "Sun", ("leap", "February 29", "extra day")))


if __name__ == "__main__":
    main()
