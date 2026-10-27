"""Structured logging of runtime events with structlog."""

from __future__ import annotations

from typing import Any, Protocol

import structlog

from agloom.observability.base import Observer
from agloom.observability.events import AgentEvent, EventName

_FAILURE_EVENTS = frozenset(
    {EventName.EXECUTION_FAILED, EventName.ACTION_FAILED, EventName.TOOL_CALL_DENIED}
)


class StructuredLogger(Protocol):
    """The subset of a structlog logger used by `LoggingObserver`."""

    def info(self, event: str, **event_kw: Any) -> Any:
        """Log an informational event."""
        ...

    def warning(self, event: str, **event_kw: Any) -> Any:
        """Log a warning event."""
        ...


class LoggingObserver(Observer):
    """Writes every runtime event as one structured log record.

    Failures and policy denials are logged at warning level; every other event
    is logged at info level.

    Args:
        logger: A structlog-compatible logger. Defaults to the
            `agloom.events` structlog logger.
    """

    def __init__(self, logger: StructuredLogger | None = None) -> None:
        self._logger: StructuredLogger = logger or structlog.get_logger("agloom.events")

    def on_event(self, event: AgentEvent) -> None:
        """Log the event."""
        log = (
            self._logger.warning if event.name in _FAILURE_EVENTS else self._logger.info
        )
        log(
            "agloom_event",
            agloom_event=event.name,
            agent_id=event.agent_id,
            run_id=event.run_id,
            timestamp=event.timestamp.isoformat(),
            topology=event.topology,
            payload=dict(event.payload),
        )
