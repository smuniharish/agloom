from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import create_agent

tool_calls = 0


@tool
def lookup_weather(city: str) -> str:
    """Return sample weather for a city."""

    global tool_calls
    tool_calls += 1
    return f"{city}: sunny"


agent = create_agent(
    model=real_model(),
    tools=[lookup_weather],
)
parts = []
for chunk in agent.stream("Reply with exactly: Streaming works."):
    parts.append(str(chunk.content))
    print(chunk.content, end="", flush=True)
print()
assert tool_calls == 0
verified_text("".join(parts), "Streaming works")
