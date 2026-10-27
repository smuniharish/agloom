"""DIRECT execution: simple requests cost exactly one model call.

Agloom analyzes every request and runs it DIRECT when no orchestration is
needed: one model call, no graph, no tools. The agent below owns a tool, but
a self-contained arithmetic question does not need it, so the analyzer selects
DIRECT and the tool is never called. Simple traffic stays as cheap and fast as
a plain model call while complex requests still get a topology.

Run:
    uv run python examples/01_direct.py
"""

from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import AgentEvent, EventName, create_agent

lookups: list[str] = []


@tool
def exchange_rate(currency: str) -> str:
    """Return today's exchange rate of a currency against the US dollar."""
    lookups.append(currency)
    return f"1 {currency} = 1.08 USD"


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(), tools=[exchange_rate], observers=[events.append]
    )

    answer = agent.invoke("What is 6 multiplied by 7? Answer in one short sentence.")

    selected = next(e for e in events if e.name == EventName.TOPOLOGY_SELECTED)
    assert selected.payload["mode"] == "direct", selected.payload
    assert EventName.DIRECT_EXECUTION_STARTED in [e.name for e in events]
    assert not lookups, "DIRECT execution must not call tools"
    print(verified_text(answer, "42"))
    print(f"Execution: {selected.payload['mode']} ({selected.payload['reason']})")


if __name__ == "__main__":
    main()
