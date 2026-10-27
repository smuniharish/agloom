"""Internal middleware applying Agloom capability routing inside agent graphs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from agloom._runtime import RunContext, current_run
from agloom.observability.events import EventName


class ToolRoutingMiddleware(AgentMiddleware):
    """Expose only routed tools, add dynamic MCP tools, and enforce run limits.

    For every model call, the model sees the tools selected by capability
    routing plus the MCP tools resolved for the current invocation. For every
    tool call, cooperative cancellation and the loop tool-call budget are
    checked before the tool runs, and calls to agent tools that were not
    offered for the request are answered with an error instead of running.
    """

    @property
    def name(self) -> str:
        """The middleware name."""
        return "agloom_tool_routing"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Route the tools offered to the model."""
        run = current_run()
        if run is None:
            return handler(request)
        run.check_cancelled()
        return handler(request.override(tools=_routed_tools(request, run)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        """Route the tools offered to the model."""
        run = current_run()
        if run is None:
            return await handler(request)
        run.check_cancelled()
        return await handler(request.override(tools=_routed_tools(request, run)))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Resolve dynamic tools and enforce run limits before execution."""
        prepared = _prepare_tool_call(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return handler(prepared)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Resolve dynamic tools and enforce run limits before execution."""
        prepared = _prepare_tool_call(request)
        if isinstance(prepared, ToolMessage):
            return prepared
        return await handler(prepared)


def _routed_tools(
    request: ModelRequest, run: RunContext
) -> list[BaseTool | dict[str, Any]]:
    # Provider built-in tools (dicts) are not Agloom capabilities and pass through.
    tools = [
        tool
        for tool in request.tools
        if not isinstance(tool, BaseTool) or tool.name in run.tool_names
    ]
    present = {tool.name for tool in tools if isinstance(tool, BaseTool)}
    tools.extend(
        tool for name, tool in run.dynamic_tools.items() if name not in present
    )
    return tools


def _prepare_tool_call(request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
    run = current_run(getattr(request.runtime, "config", None))
    if run is None:
        return request
    run.check_cancelled()
    name = request.tool_call["name"]
    if request.tool is not None and name not in run.tool_names:
        # The agent has the tool, but it was not offered for this request.
        run.emit(EventName.TOOL_CALL_DENIED, {"tool": name, "reason": "ToolNotOffered"})
        return ToolMessage(
            content=f"Tool {name!r} is not available for this request.",
            tool_call_id=request.tool_call.get("id") or "",
            name=name,
            status="error",
        )
    if run.tool_budget is not None:
        run.tool_budget.consume()
    dynamic = run.dynamic_tools.get(name)
    if dynamic is not None and request.tool is None:
        return request.override(tool=dynamic)
    return request
