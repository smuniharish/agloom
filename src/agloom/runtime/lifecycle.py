"""Validated lifecycle transitions."""

from __future__ import annotations

from enum import StrEnum

from agloom.errors import LifecycleError


class LifecycleState(StrEnum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.CREATED: frozenset({LifecycleState.INITIALIZING}),
    LifecycleState.INITIALIZING: frozenset(
        {LifecycleState.READY, LifecycleState.FAILED}
    ),
    LifecycleState.READY: frozenset(
        {LifecycleState.RUNNING, LifecycleState.STOPPED, LifecycleState.CANCELLED}
    ),
    LifecycleState.RUNNING: frozenset(
        {
            LifecycleState.WAITING,
            LifecycleState.PAUSED,
            LifecycleState.COMPLETED,
            LifecycleState.FAILED,
            LifecycleState.CANCELLED,
            LifecycleState.STOPPED,
        }
    ),
    LifecycleState.WAITING: frozenset(
        {LifecycleState.RUNNING, LifecycleState.PAUSED, LifecycleState.FAILED}
    ),
    LifecycleState.PAUSED: frozenset(
        {LifecycleState.RESUMING, LifecycleState.CANCELLED, LifecycleState.STOPPED}
    ),
    LifecycleState.RESUMING: frozenset({LifecycleState.RUNNING, LifecycleState.FAILED}),
    LifecycleState.COMPLETED: frozenset(),
    LifecycleState.FAILED: frozenset(),
    LifecycleState.CANCELLED: frozenset(),
    LifecycleState.STOPPED: frozenset(),
}


class Lifecycle:
    def __init__(self) -> None:
        self.state = LifecycleState.CREATED

    def transition(self, target: LifecycleState) -> None:
        if target not in _TRANSITIONS[self.state]:
            raise LifecycleError(
                f"invalid lifecycle transition {self.state} -> {target}"
            )
        self.state = target
