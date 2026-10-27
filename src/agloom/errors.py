"""Exceptions raised by Agloom.

Every exception defined here derives from [`AgloomError`][agloom.AgloomError].
LangGraph interrupts and agentbarrier runtime exceptions (for example
`agentbarrier.errors.ApprovalRequired`) are never wrapped, so applications
handle them with their upstream types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class AgloomError(Exception):
    """Base class for every Agloom exception."""


class ConfigurationError(AgloomError, ValueError):
    """The agent configuration or an invocation input is invalid."""


class PolicyConfigurationError(ConfigurationError):
    """A policy decision cannot be enforced with the current configuration.

    Raised, for example, when a policy rule requires approval for a tool that
    was not declared consequential and therefore has no business idempotency
    key.
    """


class TopologyError(AgloomError):
    """A topology could not be selected, compiled, or executed."""


class TopologySelectionError(TopologyError):
    """No valid execution mode or topology could be selected."""


class CompilationError(TopologyError):
    """A topology specification could not be compiled to a graph."""


class CapabilityError(AgloomError):
    """A capability operation failed."""


class CapabilityResolutionError(CapabilityError):
    """A capability is unavailable or cannot be executed with the given input."""


class RecursionLimitError(AgloomError):
    """A recursive execution budget was exhausted."""


class HarnessError(AgloomError):
    """The execution harness failed."""


class ExecutionError(HarnessError):
    """An execution failed."""


class CancellationError(HarnessError):
    """An execution was cooperatively cancelled with `Agent.stop()`."""


class ExecutionPaused(AgloomError):
    """An invocation paused and is waiting to be resumed.

    Raised by `Agent.invoke` and `Agent.ainvoke` when the run stopped at a
    LangGraph interrupt, such as a pending policy approval. Use
    `Agent.invoke_loop` to receive pauses as data instead.

    Attributes:
        interrupts: The pending LangGraph interrupts.
        thread_id: The checkpoint thread that can be resumed.
    """

    def __init__(self, interrupts: Sequence[Any], thread_id: str | None) -> None:
        super().__init__(
            f"execution paused at {len(interrupts)} interrupt(s); "
            "resume the thread with Agent.resume()"
        )
        self.interrupts = tuple(interrupts)
        self.thread_id = thread_id


class PolicyViolationError(AgloomError):
    """A tool call was denied by the configured policy before it ran.

    Attributes:
        tool_name: The denied tool.
        rule: The matching policy rule, or `None` when the default effect applied.
    """

    def __init__(self, tool_name: str, rule: str | None) -> None:
        source = f"rule {rule!r}" if rule else "the default policy effect"
        super().__init__(f"tool {tool_name!r} was denied by {source}")
        self.tool_name = tool_name
        self.rule = rule


class LoopError(AgloomError):
    """Base class for Loop Engineering failures."""


class LoopBudgetExceededError(LoopError):
    """A loop operation exceeded a configured resource budget."""


class LoopDecisionError(LoopError):
    """A loop controller produced an invalid or unusable decision."""


class LoopEvaluationError(LoopError):
    """A loop observation could not be evaluated."""


class LoopStateError(LoopError):
    """Loop state is invalid or cannot be resumed."""
