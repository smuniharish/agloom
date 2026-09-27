"""Per-invocation bounded recursive execution budget."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

from agloom.errors import RecursionError, RecursionLimitError
from agloom.models import RecursionPolicy


class RecursionSession:
    def __init__(self, policy: RecursionPolicy) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._children = 0
        self._active_workers = 0

    @contextmanager
    def child(self, depth: int) -> Generator[None, None, None]:
        if depth > self.policy.max_depth:
            raise RecursionLimitError(
                f"recursion depth limit {self.policy.max_depth} reached"
            )
        with self._lock:
            self._check_deadline()
            if self._children >= self.policy.max_subtasks:
                raise RecursionLimitError(
                    f"subtask limit {self.policy.max_subtasks} reached"
                )
            if self._active_workers >= self.policy.max_workers:
                raise RecursionLimitError(
                    f"worker limit {self.policy.max_workers} reached"
                )
            self._children += 1
            self._active_workers += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_workers -= 1

    def check(self) -> None:
        with self._lock:
            self._check_deadline()

    def _check_deadline(self) -> None:
        elapsed = time.monotonic() - self._started
        if elapsed > self.policy.max_execution_time:
            raise RecursionLimitError(
                f"execution time limit {self.policy.max_execution_time}s exceeded"
            )


class ChildExecutionUnavailable(RecursionError):
    """A topology requested child execution without an enabled runtime."""
