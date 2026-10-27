"""Hybrid: compose two to four topologies in sequence.

`pattern="hybrid"` with `composition=[...]` runs the child topologies in order
and passes each result forward. Here ReAct gathers a figure with a tool and
Reflection reviews and polishes the answer: tool use and quality control in
one run.

Run:
    uv run python examples/10_hybrid.py
"""

from _support import real_model, verified_text

from agloom import create_agent

sums: list[tuple[int, int]] = []


def add(left: int, right: int) -> int:
    """Add two integers exactly."""
    sums.append((left, right))
    return left + right


def main() -> None:
    agent = create_agent(
        model=real_model(),
        tools=[add],
        pattern="hybrid",
        composition=["react", "reflection"],
    )

    answer = agent.invoke(
        "Use the add tool to total 1250 visits from search and 830 visits from "
        "email, then write one sentence reporting the total visits."
    )

    assert (1250, 830) in sums, sums
    print(verified_text(answer, ("2080", "2,080")))


if __name__ == "__main__":
    main()
