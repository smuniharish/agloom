from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import create_agent

tool_calls = 0


@tool
def lookup_exchange_rate(currency: str) -> str:
    """Return a sample exchange rate for a currency."""

    global tool_calls
    tool_calls += 1
    return f"1 {currency} = 1.08 USD"


agent = create_agent(
    model=real_model(),
    tools=[lookup_exchange_rate],
)
result = agent.invoke(
    "Compute 6 multiplied by 7. Reply in one sentence and include the number 42."
)
assert tool_calls == 0
print(verified_text(result, "42"))
