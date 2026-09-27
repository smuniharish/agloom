"""Normalized, per-agent runtime events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    name: str
    agent_id: str
    run_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    topology: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


EventSink = Callable[[AgentEvent], None]
