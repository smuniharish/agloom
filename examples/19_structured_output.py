"""Structured output: validated Pydantic objects from any Agloom agent.

`agent.with_structured_output(Model)` wraps the agent with xstructured: the
agent runs as usual (any topology, tools, policy), its answer carries the
object in an envelope, and xstructured parses, validates, and, if needed,
repairs it. It accepts the agent's normal inputs (text, a conversation, or a
mapping); the schema instructions join the latest user message. The result
keeps the free-text content next to the validated `structured` object.

Run:
    uv run python examples/19_structured_output.py
"""

from typing import Literal

from _support import real_model
from pydantic import BaseModel, Field

from agloom import create_agent


class Incident(BaseModel):
    """A triaged production incident."""

    service: str = Field(description="The affected service.")
    severity: Literal["low", "medium", "high", "critical"]
    customer_impact: bool = Field(description="Whether customers are affected.")
    error_rate_percent: float = Field(ge=0, le=100)


def main() -> None:
    agent = create_agent(model=real_model(), pattern="react")
    triage = agent.with_structured_output(Incident)

    report = triage.invoke(
        "Triage this report: since 09:10 UTC the payments API returns HTTP 500 "
        "for 40% of checkout requests, and customers cannot pay."
    ).structured
    assert isinstance(report, Incident), report
    assert "payment" in report.service.lower(), report
    assert report.customer_impact
    assert report.error_rate_percent == 40
    print("From a report:", report.model_dump_json())

    chat = triage.invoke(
        [
            ("user", "Search results stopped loading for some users."),
            ("assistant", "How many requests fail, and since when?"),
            ("user", "About 15% of search requests time out since 14:00."),
            ("user", "Triage it."),
        ]
    ).structured
    assert isinstance(chat, Incident), chat
    assert "search" in chat.service.lower(), chat
    assert chat.error_rate_percent == 15
    print("From a conversation:", chat.model_dump_json())


if __name__ == "__main__":
    main()
