from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import create_agent

tool_calls: list[tuple[int, int]] = []


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""

    tool_calls.append((left, right))
    return left + right


agent = create_agent(
    model=real_model(),
    tools=[add],
    pattern="hybrid",
    composition=["react", "reflection"],
)
result = agent.invoke(
    "Use the add tool to calculate 2 + 3, then review the result. The final "
    "answer must preserve the number 5."
)
assert tool_calls == [(2, 3)]
print(verified_text(result, "5"))
