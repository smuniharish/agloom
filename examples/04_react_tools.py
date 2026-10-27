"""ReAct: a tool-calling agent that reasons, acts, and observes.

`pattern="react"` selects the ReAct topology explicitly, skipping task
analysis. It is LangChain's `create_agent` wrapped with Agloom's capability
routing, policy, cancellation, and loop budgets. Tools are plain LangChain
tools or annotated functions.

Run:
    uv run python examples/04_react_tools.py
"""

from _support import real_model, verified_text

from agloom import create_agent

products: list[tuple[int, int]] = []


def multiply(left: int, right: int) -> int:
    """Multiply two integers exactly."""
    products.append((left, right))
    return left * right


def main() -> None:
    agent = create_agent(model=real_model(), tools=[multiply], pattern="react")

    answer = agent.invoke("Use the multiply tool to calculate 17 times 23.")

    assert products == [(17, 23)], products
    print(verified_text(answer, "391"))


if __name__ == "__main__":
    main()
