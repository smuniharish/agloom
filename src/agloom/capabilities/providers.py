"""Capability providers for local tools, application services, and MCP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from langchain_core.tools import BaseTool

from agloom.capabilities.registry import CapabilityDescriptor, CapabilityKind
from agloom.errors import CapabilityResolutionError

_MCP_RUNTIME_CAPABILITIES = {"mcp_runtime", "mcp_adapter"}
_INTERNAL_RUNTIME_CAPABILITIES = {*_MCP_RUNTIME_CAPABILITIES, "xai_runtime"}


class CapabilityProvider(ABC):
    """Loads capability descriptors from one concrete source."""

    @abstractmethod
    def load(self) -> tuple[CapabilityDescriptor, ...]:
        """Return the provider's currently available capabilities."""

    async def aload(self) -> tuple[CapabilityDescriptor, ...]:
        return self.load()


class LocalLangChainToolProvider(CapabilityProvider):
    def __init__(self, tools: Sequence[BaseTool | Any]) -> None:
        self._tools = tuple(tools)

    def load(self) -> tuple[CapabilityDescriptor, ...]:
        descriptors = []
        for tool in self._tools:
            fallback_name = getattr(tool, "__name__", type(tool).__name__)
            name = str(getattr(tool, "name", fallback_name))
            description = str(
                getattr(tool, "description", getattr(tool, "__doc__", "")) or ""
            )
            descriptors.append(
                CapabilityDescriptor(
                    name=name,
                    description=description,
                    kind=CapabilityKind.TOOL,
                    value=tool,
                    provider="langchain",
                )
            )
        return tuple(descriptors)


class ApplicationCapabilityProvider(CapabilityProvider):
    def __init__(self, capabilities: Mapping[str, Any]) -> None:
        self._capabilities = dict(capabilities)

    def load(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            CapabilityDescriptor(
                name=name,
                description=str(getattr(capability, "description", "") or ""),
                kind=CapabilityKind.APPLICATION,
                value=capability,
                provider="application",
                metadata={
                    "type": type(capability).__name__,
                    "internal": name in _INTERNAL_RUNTIME_CAPABILITIES,
                },
            )
            for name, capability in self._capabilities.items()
            if name not in _MCP_RUNTIME_CAPABILITIES
        )


@dataclass(frozen=True)
class MCPRemoteCapability:
    runtime: Any
    capability_id: str
    capability_type: str

    def invoke(self, arguments: dict[str, Any] | None = None) -> Any:
        raise CapabilityResolutionError(
            f"MCP capability {self.capability_id!r} requires async execution"
        )

    async def ainvoke(self, arguments: dict[str, Any] | None = None) -> Any:
        if self.capability_type == "tool":
            result = self.runtime.execute(self.capability_id, arguments)
        elif self.capability_type == "resource":
            if arguments is not None:
                raise CapabilityResolutionError(
                    "MCP resources do not accept invocation arguments"
                )
            result = self.runtime.read_resource(self.capability_id)
        elif self.capability_type == "prompt":
            result = self.runtime.get_prompt(self.capability_id, arguments)
        else:
            raise CapabilityResolutionError(
                f"unsupported MCP capability type {self.capability_type!r}"
            )
        return await result if isawaitable(result) else result


class MCPCapabilityProvider(CapabilityProvider):
    """Discovers MCP tools through the configured package adapter."""

    def __init__(
        self,
        capabilities: Mapping[str, Any],
        *,
        server_name: str | None = None,
    ) -> None:
        self._capabilities = dict(capabilities)
        self._server_name = server_name

    def load(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            CapabilityDescriptor(
                name=name,
                description=str(getattr(capability, "description", "") or ""),
                kind=CapabilityKind.MCP,
                value=capability,
                provider="mcp-capability-router",
                metadata={
                    "type": type(capability).__name__,
                    "internal": True,
                },
            )
            for name, capability in self._capabilities.items()
            if name in _MCP_RUNTIME_CAPABILITIES
        )

    async def adiscover(self) -> tuple[CapabilityDescriptor, ...]:
        descriptors = []
        runtime = self._capabilities.get("mcp_runtime")
        if runtime is None:
            return ()
        capabilities = runtime.registry.list(server_id=self._server_name)
        if isawaitable(capabilities):
            capabilities = await capabilities
        for capability in capabilities:
            capability_id = str(capability.capability_id)
            capability_type = str(capability.type)
            metadata = dict(capability.metadata)
            metadata.update(
                {
                    "mcp_capability_id": capability_id,
                    "mcp_type": capability_type,
                    "mcp_server_id": str(capability.server_id),
                }
            )
            descriptors.append(
                CapabilityDescriptor(
                    name=f"mcp:{capability_id}",
                    description=str(capability.description or ""),
                    kind=CapabilityKind.MCP,
                    value=MCPRemoteCapability(
                        runtime,
                        capability_id,
                        capability_type,
                    ),
                    provider="mcp-capability-router",
                    metadata=metadata,
                )
            )
        return tuple(descriptors)
