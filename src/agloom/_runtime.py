"""Per-invocation runtime context shared by Agloom graphs and middleware.

The loop-level context travels in the LangGraph run configuration under a
private key; each topology execution additionally runs inside a context
variable scope, which LangGraph propagates to its nodes, threads, and tasks.
The context is never persisted: checkpointers store only JSON-compatible
values.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_config, get_stream_writer

from agloom.errors import CancellationError, LoopBudgetExceededError

if TYPE_CHECKING:
    from agloom.agent._recursion import RecursionSession

RUN_KEY = "__agloom_run__"

Emit = Callable[[str, Mapping[str, Any]], None]


class ToolBudget:
    """A thread-safe tool-call budget shared by every iteration of one loop."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        """The number of tool calls admitted so far."""
        with self._lock:
            return self._used

    def consume(self) -> None:
        """Admit one tool call.

        Raises:
            LoopBudgetExceededError: If the budget is exhausted.
        """
        with self._lock:
            if self._used >= self.limit:
                raise LoopBudgetExceededError(
                    f"tool-call budget of {self.limit} exhausted before tool execution"
                )
            self._used += 1


class ChildRunner(Protocol):
    """Runs one recursive child task through the owning agent."""

    def run(self, task: str, config: RunnableConfig) -> str:
        """Run the child task synchronously and return its text result."""
        ...

    def arun(self, task: str, config: RunnableConfig) -> Awaitable[str]:
        """Run the child task asynchronously and return its text result."""
        ...


@dataclass(frozen=True)
class RunContext:
    """Runtime services available to one harness execution."""

    agent_id: str
    run_id: str
    cancel_event: threading.Event
    emit: Emit
    tool_names: frozenset[str] = frozenset()
    dynamic_tools: Mapping[str, BaseTool] = field(default_factory=dict)
    tool_budget: ToolBudget | None = None
    pausable: bool = False
    depth: int = 0
    recursion: RecursionSession | None = None
    child: ChildRunner | None = None
    errors: dict[int, BaseException] = field(default_factory=dict, compare=False)
    scratch: dict[str, Any] = field(default_factory=dict, compare=False)

    def check_cancelled(self) -> None:
        """Raise when the owning agent requested cooperative cancellation."""
        if self.cancel_event.is_set():
            raise CancellationError("execution was cooperatively cancelled")

    def derive(self, **changes: Any) -> RunContext:
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)


_SCOPE: ContextVar[RunContext | None] = ContextVar("agloom_scope", default=None)


def current_run(config: RunnableConfig | None = None) -> RunContext | None:
    """Return the run context of the current topology execution, if any.

    The scope set with `scoped` takes precedence over the run configuration,
    so the context survives runnable wrappers that rebuild configurations.
    """
    scope = _SCOPE.get()
    if scope is not None:
        return scope
    return configured_run(config)


def configured_run(config: RunnableConfig | None = None) -> RunContext | None:
    """Return the run context carried by a run configuration, if any."""
    if config is None:
        try:
            config = get_config()
        except RuntimeError:
            return None
    configurable = config.get("configurable") or {}
    run = configurable.get(RUN_KEY)
    return run if isinstance(run, RunContext) else None


@contextmanager
def scoped(run: RunContext | None) -> Generator[None]:
    """Make `run` the current run for the code executed inside the block."""
    token = _SCOPE.set(run)
    try:
        yield
    finally:
        _SCOPE.reset(token)


@contextmanager
def tools_withheld() -> Generator[None]:
    """Offer no tools to the model inside the block."""
    run = current_run()
    with scoped(
        run.derive(tool_names=frozenset(), dynamic_tools={})
        if run is not None
        else None
    ):
        yield


def with_run(config: RunnableConfig | None, run: RunContext) -> RunnableConfig:
    """Return a copy of `config` that carries `run`."""
    resolved: dict[str, Any] = dict(config or {})
    configurable = dict(resolved.get("configurable") or {})
    configurable[RUN_KEY] = run
    resolved["configurable"] = configurable
    return cast(RunnableConfig, resolved)


def write_stream(payload: Mapping[str, Any]) -> None:
    """Write to the LangGraph `custom` stream when called inside a graph run.

    Outside a LangGraph run, including inside plain LangChain runnables, the
    payload is dropped.
    """
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):
        return
    writer(dict(payload))
