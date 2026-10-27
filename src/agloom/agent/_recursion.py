"""The budget shared by one recursive invocation tree."""

from __future__ import annotations

import threading
import time

from agloom.errors import RecursionLimitError
from agloom.models import ExecutionLimits


class RecursionSession:
    """Enforces depth, child-count, and wall-clock limits for recursive children.

    One session covers a whole invocation, including every Loop Engineering
    iteration. The wall-clock limit starts with the first child. Parallel
    children are bounded by `ExecutionLimits.max_workers`, because the Planner
    dispatches at most that many steps at a time.
    """

    def __init__(self, limits: ExecutionLimits) -> None:
        self.limits = limits
        self._lock = threading.Lock()
        self._started: float | None = None
        self._children = 0

    def admit(self, depth: int) -> None:
        """Admit one child execution at a depth.

        Raises:
            RecursionLimitError: If a depth, child, or time limit is exhausted.
        """
        if depth > self.limits.max_depth:
            raise RecursionLimitError(
                f"recursion depth limit {self.limits.max_depth} reached"
            )
        with self._lock:
            if self._started is None:
                self._started = time.monotonic()
            self._check_deadline()
            if self._children >= self.limits.max_children:
                raise RecursionLimitError(
                    f"recursive child limit {self.limits.max_children} reached"
                )
            self._children += 1

    def check(self) -> None:
        """Raise when the wall-clock limit is exhausted."""
        with self._lock:
            self._check_deadline()

    def _check_deadline(self) -> None:
        if (
            self._started is not None
            and time.monotonic() - self._started > self.limits.max_execution_time
        ):
            raise RecursionLimitError(
                "recursive execution time limit "
                f"{self.limits.max_execution_time}s exceeded"
            )
