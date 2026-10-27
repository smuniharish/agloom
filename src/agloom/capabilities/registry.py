"""Capability descriptors and the agent-local capability catalog."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from agloom.errors import CapabilityResolutionError, ConfigurationError


class CapabilityKind(StrEnum):
    """Where a capability comes from."""

    TOOL = "tool"
    APPLICATION = "application"
    MCP = "mcp"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """One discoverable capability and the object that implements it.

    Attributes:
        name: Stable, unique capability name.
        description: What the capability does; used for routing and analysis.
        kind: The capability source.
        value: The implementing object (a LangChain tool, an application
            service, or an MCP capability handle).
        provider: The component that supplied the capability.
        metadata: Read-only routing metadata, such as `aliases`.
    """

    name: str
    description: str
    kind: CapabilityKind
    value: Any = field(repr=False, compare=False)
    provider: str = "application"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        name = self.name.strip()
        provider = self.provider.strip()
        if not name:
            raise ConfigurationError("capability name must not be blank")
        if not provider:
            raise ConfigurationError("capability provider must not be blank")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def model_callable(self) -> bool:
        """Whether the model can call this capability as a tool."""
        return self.kind is CapabilityKind.TOOL or (
            self.kind is CapabilityKind.MCP and self.metadata.get("mcp_type") == "tool"
        )


class CapabilityRegistry(ABC):
    """A catalog of capabilities shared by routing, analysis, and execution."""

    @abstractmethod
    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Register one descriptor.

        Raises:
            ConfigurationError: If the name is already registered.
        """

    @abstractmethod
    def descriptor(self, name: str) -> CapabilityDescriptor:
        """Return one descriptor by its exact name.

        Raises:
            CapabilityResolutionError: If no capability has the name.
        """

    @abstractmethod
    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Return a snapshot of every registered descriptor."""

    def resolve(self, name: str) -> Any:
        """Return the object that implements a capability."""
        return self.descriptor(name).value

    @property
    def names(self) -> tuple[str, ...]:
        """The registered capability names, in registration order."""
        return tuple(descriptor.name for descriptor in self.descriptors())


class InMemoryCapabilityRegistry(CapabilityRegistry):
    """A thread-safe, process-local capability catalog."""

    def __init__(self, descriptors: Iterable[CapabilityDescriptor] = ()) -> None:
        self._lock = threading.Lock()
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Register one descriptor, rejecting duplicate names."""
        with self._lock:
            if descriptor.name in self._descriptors:
                raise ConfigurationError(
                    f"capability {descriptor.name!r} is already registered"
                )
            self._descriptors[descriptor.name] = descriptor

    def descriptor(self, name: str) -> CapabilityDescriptor:
        """Return one descriptor by its exact name."""
        with self._lock:
            descriptor = self._descriptors.get(name)
        if descriptor is None:
            raise CapabilityResolutionError(f"capability {name!r} is not registered")
        return descriptor

    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Return a snapshot of every registered descriptor."""
        with self._lock:
            return tuple(self._descriptors.values())
