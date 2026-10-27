"""Normalized runtime events emitted by Agloom agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class EventName(StrEnum):
    """Stable names of the events emitted by an `Agent`."""

    AGENT_CREATED = "AgentCreated"
    EXECUTION_STARTED = "ExecutionStarted"
    CAPABILITIES_ROUTED = "CapabilitiesRouted"
    TOPOLOGY_SELECTION_STARTED = "TopologySelectionStarted"
    TOPOLOGY_SELECTED = "TopologySelected"
    DIRECT_EXECUTION_STARTED = "DirectExecutionStarted"
    EXECUTION_COMPLETED = "ExecutionCompleted"
    EXECUTION_PAUSED = "ExecutionPaused"
    EXECUTION_FAILED = "ExecutionFailed"
    EXECUTION_CANCELLED = "ExecutionCancelled"
    TOOL_CALL_DENIED = "ToolCallDenied"
    APPROVAL_REQUIRED = "ApprovalRequired"
    ACTION_EXECUTED = "ActionExecuted"
    ACTION_REPLAYED = "ActionReplayed"
    ACTION_FAILED = "ActionFailed"
    LOOP_STARTED = "LoopStarted"
    LOOP_ITERATION_STARTED = "LoopIterationStarted"
    LOOP_ITERATION_COMPLETED = "LoopIterationCompleted"
    LOOP_REFRESHED = "LoopRefreshed"
    LOOP_PAUSED = "LoopPaused"
    LOOP_COMPLETED = "LoopCompleted"
    LOOP_TERMINATED = "LoopTerminated"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One normalized runtime event.

    Attributes:
        name: Stable event name; see [`EventName`][agloom.EventName].
        agent_id: Identifier of the emitting agent instance.
        run_id: Identifier of the invocation the event belongs to.
        timestamp: UTC time at which the event was created.
        topology: The topology involved, when known.
        payload: Event-specific, JSON-compatible details. Never contains
            model prompts, tool arguments, or credentials.
    """

    name: str
    agent_id: str
    run_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    topology: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
