"""Structlog-backed runtime event observer."""

from __future__ import annotations

from typing import Any, Protocol

import structlog

from agloom.observability.base import Observer
from agloom.runtime.events import AgentEvent


class StructuredLogger(Protocol):
    def info(self, event: str, **event_kw: Any) -> Any: ...


class LoggingObserver(Observer):
    def __init__(
        self,
        logger: StructuredLogger | None = None,
    ) -> None:
        self._logger = logger or structlog.get_logger("agloom.events")

    def on_event(self, event: AgentEvent) -> None:
        self._logger.info(
            "agloom_event",
            agloom_event=event.name,
            agent_id=event.agent_id,
            run_id=event.run_id,
            timestamp=event.timestamp.isoformat(),
            topology=event.topology,
            payload=event.payload,
        )
