"""Capability routing: show each request only the capabilities it needs.

`DefaultCapabilityRouter` filters the catalog with a `CapabilityFilter`
(admin tools are never routed here), ranks what remains with any LangChain
`Embeddings`, and keeps the top `max_results`; capabilities named in the
request rank first. The model only sees the routed tools, which keeps prompts
small and choices sharp. Application capabilities (services the app owns)
join the catalog and run directly, under the policy, with
`execute_capability`.

Run:
    uv run python examples/45_capability_routing.py
"""

import math
import re
import zlib

from _support import real_model, verified_text
from langchain_core.embeddings import Embeddings

from agloom import (
    AgentEvent,
    CapabilityDescriptor,
    CapabilityFilter,
    DefaultCapabilityRouter,
    EventName,
    create_agent,
)

STOPWORDS = {"a", "an", "and", "for", "is", "of", "or", "the", "to", "what", "with"}


class HashingEmbeddings(Embeddings):
    """Dependency-free bag-of-words embeddings for this example.

    Use any LangChain `Embeddings` in production, such as `OpenAIEmbeddings`.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        vector[0] = 0.1
        for word in re.findall(r"[a-z]+", text.casefold()):
            if word not in STOPWORDS:
                vector[1 + zlib.crc32(word.encode()) % (self.dimensions - 1)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


class CustomerFacing(CapabilityFilter):
    """Customer requests never see administrative capabilities."""

    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        return not capability.name.startswith("admin_")


forecasts: list[str] = []


def weather_forecast(city: str) -> str:
    """Return tomorrow's weather forecast for a city."""
    forecasts.append(city)
    return f"{city} tomorrow: light rain, 14 degrees Celsius."


def currency_converter(amount: float, source: str, target: str) -> str:
    """Convert an amount of money between currencies."""
    return f"{amount} {source} = {amount * 1.08:.2f} {target}"


def flight_status(flight: str) -> str:
    """Return the departure status of a flight."""
    return f"Flight {flight} departs on time."


def hotel_search(city: str) -> str:
    """Find available hotel rooms in a city."""
    return f"Three hotels have rooms in {city}."


def translate_text(text: str, language: str) -> str:
    """Translate a short text into another language."""
    return f"[{language}] {text}"


def admin_reset_password(user: str) -> str:
    """Reset a user's password (administrators only)."""
    return f"Password reset for {user}."


class LoyaltyProgram:
    """An application service exposed as a capability."""

    description = "Look up a member's loyalty points balance."

    def invoke(self, request: dict[str, str]) -> str:
        return f"Member {request['member_id']} has 12,400 points."


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(),
        tools=[
            weather_forecast,
            currency_converter,
            flight_status,
            hotel_search,
            translate_text,
            admin_reset_password,
        ],
        capabilities={"loyalty_points": LoyaltyProgram()},
        capability_router=DefaultCapabilityRouter(
            capability_filter=CustomerFacing(),
            embeddings=HashingEmbeddings(),
            max_results=3,
        ),
        pattern="react",
        observers=[events.append],
    )
    request = "What is the weather forecast for Paris tomorrow?"

    selection = agent.route_capabilities(request)
    print("Routed:", ", ".join(selection.names))
    assert selection.names[0] == "weather_forecast", selection.names
    assert len(selection.names) == 3
    assert not any(name.startswith("admin_") for name in selection.names)

    answer = agent.invoke(request)
    routed = next(e for e in events if e.name == EventName.CAPABILITIES_ROUTED)
    assert forecasts == ["Paris"], forecasts
    assert routed.payload["selected"] == 3
    print(verified_text(answer, "rain", "14"))

    points = agent.execute_capability("loyalty_points", {"member_id": "M-7"})
    print("Application capability:", verified_text(points, "12,400"))


if __name__ == "__main__":
    main()
