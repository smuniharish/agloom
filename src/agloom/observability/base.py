"""Pluggable observability contracts and composition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from langchain_core.callbacks.base import BaseCallbackHandler

from agloom.runtime.events import AgentEvent


class Observer(ABC):
    """Receives runtime events and optionally contributes model callbacks."""

    @abstractmethod
    def on_event(self, event: AgentEvent) -> None:
        """Record one normalized Agloom runtime event."""

    @property
    def callback_handlers(self) -> tuple[BaseCallbackHandler, ...]:
        return ()

    def flush(self) -> None:
        """Flush buffered telemetry."""
        return None

    def shutdown(self) -> None:
        """Release observer resources."""
        return None


class CallableObserver(Observer):
    """Adapt the legacy event sink callable to the observer contract."""

    def __init__(self, sink: Callable[[AgentEvent], None]) -> None:
        self._sink = sink

    def on_event(self, event: AgentEvent) -> None:
        self._sink(event)


class CompositeObserver(Observer):
    """Fan out telemetry without hiding observer failures."""

    def __init__(self, observers: Iterable[Observer] = ()) -> None:
        self._observers = tuple(observers)

    def on_event(self, event: AgentEvent) -> None:
        for observer in self._observers:
            observer.on_event(event)

    @property
    def callback_handlers(self) -> tuple[BaseCallbackHandler, ...]:
        return tuple(
            handler
            for observer in self._observers
            for handler in observer.callback_handlers
        )

    def flush(self) -> None:
        for observer in self._observers:
            observer.flush()

    def shutdown(self) -> None:
        for observer in reversed(self._observers):
            observer.shutdown()

    @property
    def observers(self) -> tuple[Observer, ...]:
        return self._observers
