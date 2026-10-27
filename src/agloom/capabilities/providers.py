"""Capability providers for local tools, application services, and MCP servers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as as_tool

from agloom.capabilities.registry import CapabilityDescriptor, CapabilityKind
from agloom.errors import CapabilityResolutionError, ConfigurationError

if TYPE_CHECKING:
    from mcp_capability_router import MCPRuntime

_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_TOOL_NAME = 64


def to_tool(value: BaseTool | Callable[..., Any]) -> BaseTool:
    """Return a LangChain tool for a tool object or a plain annotated function."""
    if isinstance(value, BaseTool):
        return value
    if callable(value):
        return as_tool(value)
    raise ConfigurationError(
        f"tools must be LangChain tools or functions, not {value!r}"
    )


def tool_descriptors(
    tools: Sequence[BaseTool], *, provider: str = "langchain"
) -> tuple[CapabilityDescriptor, ...]:
    """Describe LangChain tools supplied directly or by agent middleware."""
    return tuple(
        CapabilityDescriptor(
            name=tool.name,
            description=tool.description or "",
            kind=CapabilityKind.TOOL,
            value=tool,
            provider=provider,
        )
        for tool in tools
    )


def application_descriptors(
    capabilities: Mapping[str, Any],
) -> tuple[CapabilityDescriptor, ...]:
    """Describe application-owned capability objects."""
    return tuple(
        CapabilityDescriptor(
            name=name,
            description=str(getattr(capability, "description", "") or ""),
            kind=CapabilityKind.APPLICATION,
            value=capability,
            provider="application",
            metadata={"type": type(capability).__name__},
        )
        for name, capability in capabilities.items()
    )


def mcp_tool_name(server_id: str, name: str) -> str:
    """Return a collision-free, function-calling-safe name for one MCP tool.

    The name is `<server_id>__<tool>` with unsupported characters replaced by
    `_`. Names longer than 64 characters are shortened with a stable digest.
    """
    candidate = _TOOL_NAME.sub("_", f"{server_id}__{name}").strip("_") or "mcp_tool"
    if len(candidate) <= _MAX_TOOL_NAME:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[: _MAX_TOOL_NAME - 9]}_{digest}"


@dataclass(frozen=True)
class MCPCapability:
    """Executable handle for one capability discovered by an `MCPRuntime`."""

    runtime: MCPRuntime
    capability_id: str
    capability_type: str
    tool_name: str | None = None

    def invoke(self, arguments: Any = None) -> Any:
        """Reject synchronous execution; MCP capabilities are asynchronous."""
        raise CapabilityResolutionError(
            f"MCP capability {self.capability_id!r} is asynchronous; "
            "use Agent.aexecute_capability"
        )

    async def ainvoke(self, arguments: Any = None) -> Any:
        """Call the tool, read the resource, or render the prompt."""
        if self.capability_type == "tool":
            return await self.runtime.execute(self.capability_id, arguments or {})
        if self.capability_type == "resource":
            if arguments is not None:
                raise CapabilityResolutionError("MCP resources do not accept arguments")
            return await self.runtime.read_resource(self.capability_id)
        return await self.runtime.get_prompt(self.capability_id, arguments)

    async def as_tool(self) -> BaseTool:
        """Return the LangChain tool that calls this MCP tool."""
        return await self.runtime.as_tool(self.capability_id, name=self.tool_name)


async def mcp_descriptors(runtime: MCPRuntime) -> tuple[CapabilityDescriptor, ...]:
    """Describe the capabilities currently registered in an `MCPRuntime`."""
    descriptors: list[CapabilityDescriptor] = []
    for capability in await runtime.registry.list():
        capability_type = str(capability.type)
        tool_name = (
            mcp_tool_name(capability.server_id, capability.name)
            if capability_type == "tool"
            else None
        )
        aliases = tuple(alias for alias in (capability.name, capability.title) if alias)
        descriptors.append(
            CapabilityDescriptor(
                name=tool_name or f"mcp:{capability.capability_id}",
                description=capability.description or capability.title or "",
                kind=CapabilityKind.MCP,
                value=MCPCapability(
                    runtime, capability.capability_id, capability_type, tool_name
                ),
                provider="mcp-capability-router",
                metadata={
                    "aliases": aliases,
                    "mcp_type": capability_type,
                    "mcp_capability_id": capability.capability_id,
                    "mcp_server_id": capability.server_id,
                },
            )
        )
    return tuple(descriptors)
