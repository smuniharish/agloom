from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import create_agent
from agloom.capabilities.integrations import context_middleware

tool_calls: list[str] = []


@tool
def country_capital(country: str) -> str:
    """Return the capital of a supported country."""

    tool_calls.append(country)
    capitals = {"France": "Paris", "Japan": "Tokyo"}
    return capitals[country]


model = real_model()
middleware = context_middleware(
    model,
    trigger=("tokens", 100_000),
    keep=("messages", 5),
)
agent = create_agent(
    model=model,
    tools=[country_capital],
    pattern="react",
    middleware=[middleware],
)
result = agent.invoke(
    "Use the country_capital tool for France and reply in one sentence."
)
assert tool_calls == ["France"]
print(verified_text(result, "Paris"))
