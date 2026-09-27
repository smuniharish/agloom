from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import create_agent

tool_calls: list[tuple[int, int]] = []


@tool
def multiply(left: int, right: int) -> int:
    """Multiply two integers."""

    tool_calls.append((left, right))
    return left * right


agent = create_agent(model=real_model(), tools=[multiply], pattern="react")
result = agent.invoke("Use the multiply tool to calculate 17 times 23.")
assert tool_calls == [(17, 23)]
print(verified_text(result, "391"))
