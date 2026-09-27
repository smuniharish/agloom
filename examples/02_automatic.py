from _support import real_model, verified_text
from langchain_core.tools import tool

from agloom import create_agent

tool_calls: list[tuple[float, float]] = []


@tool
def average_speed(distance_km: float, duration_hours: float) -> float:
    """Calculate average speed in kilometers per hour."""

    tool_calls.append((distance_km, duration_hours))
    return distance_km / duration_hours


agent = create_agent(
    model=real_model(),
    tools=[average_speed],
)
result = agent.invoke(
    "Use the average_speed tool for a train that travels 120 kilometers in "
    "2 hours. Return one sentence containing the result in km/h."
)
assert tool_calls == [(120.0, 2.0)]
print(verified_text(result, "60", "km/h"))
