"""Abstract capability catalog and its agent-local implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from agloom.errors import CapabilityResolutionError


class CapabilityKind(StrEnum):
    TOOL = "tool"
    APPLICATION = "application"
    MCP = "mcp"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """One discoverable capability and the object that implements it."""

    name: str
    description: str
    kind: CapabilityKind
    value: Any = field(repr=False)
    provider: str = "application"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("capability name must not be empty")
        provider = self.provider.strip()
        if not provider:
            raise ValueError("capability provider must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


class CapabilityRegistry(ABC):
    """Abstract catalog used by routing, analysis, and execution."""

    @abstractmethod
    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Register one descriptor, rejecting duplicate stable names."""

    @abstractmethod
    def descriptor(self, name: str) -> CapabilityDescriptor:
        """Return one descriptor by exact stable name."""

    @abstractmethod
    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """Return a stable snapshot of all registered descriptors."""

    def resolve(self, name: str) -> Any:
        return self.descriptor(name).value

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(descriptor.name for descriptor in self.descriptors())


class InMemoryCapabilityRegistry(CapabilityRegistry):
    """Per-agent in-memory implementation of the abstract catalog."""

    def __init__(
        self,
        descriptors: Iterable[CapabilityDescriptor] = (),
    ) -> None:
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: CapabilityDescriptor) -> None:
        if descriptor.name in self._descriptors:
            raise ValueError(f"capability {descriptor.name!r} is already registered")
        self._descriptors[descriptor.name] = descriptor

    def descriptor(self, name: str) -> CapabilityDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise CapabilityResolutionError(
                f"capability {name!r} is not registered"
            ) from error

    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(self._descriptors.values())
