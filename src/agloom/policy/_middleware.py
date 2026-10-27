"""Internal middleware enforcing a PolicyManager inside agent graphs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agloom._runtime import RunContext, current_run
from agloom.errors import ExecutionError
from agloom.policy.manager import PolicyManager


class PolicyMiddleware(AgentMiddleware):
    """Applies a `PolicyManager` to every tool call; always the innermost middleware.

    Being innermost means retry middleware placed before it re-enters the
    policy boundary, where agentbarrier replays completed actions instead of
    executing them twice.
    """

    def __init__(self, manager: PolicyManager) -> None:
        super().__init__()
        self._manager = manager

    @property
    def name(self) -> str:
        """The middleware name."""
        return "agloom_policy"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Enforce the policy around the tool call."""
        return self._manager._handle_tool_call(request, handler, _run_of(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Enforce the policy around the tool call."""
        return await self._manager._ahandle_tool_call(
            request, handler, _run_of(request)
        )


def _run_of(request: ToolCallRequest) -> RunContext:
    """Return the Agloom run the tool call belongs to; tools never run without one."""
    run = current_run(getattr(request.runtime, "config", None))
    if run is None:
        raise ExecutionError("policy-governed tools run only inside Agloom agents")
    return run
