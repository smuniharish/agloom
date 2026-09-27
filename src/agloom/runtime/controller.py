"""Explicit controller for cooperative stop across registered agent handles."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class RuntimeController:
    """Application-owned stop-all boundary; agents are never registered globally."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stoppers: set[Callable[[], Any]] = set()

    def register(self, agent: Any) -> Callable[[], None]:
        stopper = agent.stop
        with self._lock:
            self._stoppers.add(stopper)

        def unregister() -> None:
            with self._lock:
                self._stoppers.discard(stopper)

        return unregister

    def stop_all(self) -> None:
        with self._lock:
            stoppers = tuple(self._stoppers)
        for stop in stoppers:
            stop()
