"""Automatic execution selection: one agent, the right execution for each request.

Without `pattern`, Agloom analyzes every request with a structured task
analysis, and a deterministic strategy maps it to DIRECT or one of eight
topologies. Here the same agent runs a request that needs its tool on a
topology (ReAct) and a greeting DIRECT. Pass `pattern=[...]` to restrict the
choice to an allow-list, or `pattern="react"` to skip analysis entirely.

Run:
    uv run python examples/02_automatic.py
"""

from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import AgentEvent, EventName, create_agent

measurements: list[tuple[float, float]] = []


@tool
def average_speed(distance_km: float, duration_hours: float) -> float:
    """Calculate an average speed in kilometers per hour."""
    measurements.append((distance_km, duration_hours))
    return distance_km / duration_hours


def selection(events: list[AgentEvent]) -> AgentEvent:
    """Return the latest execution decision recorded by the observer."""
    return [e for e in events if e.name == EventName.TOPOLOGY_SELECTED][-1]


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(), tools=[average_speed], observers=[events.append]
    )

    answer = agent.invoke(
        "Use the average_speed tool for a train that travels 120 kilometers in "
        "2 hours, then state the result in km/h in one sentence."
    )
    tool_run = selection(events)
    assert tool_run.payload["mode"] == "topology", tool_run
    assert measurements == [(120.0, 2.0)], measurements
    print(verified_text(answer, "60"))
    print(f"-> {tool_run.topology}: {tool_run.payload['reason']}")

    greeting = agent.invoke("Say hello to the operations team in five words.")
    chat_run = selection(events)
    assert chat_run.payload["mode"] == "direct", chat_run
    assert len(measurements) == 1, "the greeting must not call the tool"
    print(verified_text(greeting))
    print(f"-> direct: {chat_run.payload['reason']}")


if __name__ == "__main__":
    main()
