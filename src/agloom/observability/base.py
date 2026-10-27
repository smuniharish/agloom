"""The observer contract and its internal composition."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from langchain_core.callbacks.base import BaseCallbackHandler

from agloom.observability.events import AgentEvent

_LOGGER = logging.getLogger("agloom.observability")


class Observer(ABC):
    """Receives Agloom runtime events and optionally contributes LangChain callbacks.

    Observer failures never change the outcome of an execution: they are logged
    with their traceback on the `agloom.observability` logger and the remaining
    observers still receive the event.
    """

    @abstractmethod
    def on_event(self, event: AgentEvent) -> None:
        """Record one runtime event."""

    @property
    def callback_handlers(self) -> tuple[BaseCallbackHandler, ...]:
        """LangChain callback handlers attached to every model, tool, and graph run."""
        return ()

    def flush(self) -> None:
        """Flush buffered telemetry."""
        return

    def shutdown(self) -> None:
        """Flush and release observer resources."""
        return


class CallableObserver(Observer):
    """Adapts a plain `(event) -> None` callable to the observer contract."""

    def __init__(self, handler: Callable[[AgentEvent], object]) -> None:
        self._handler = handler

    def on_event(self, event: AgentEvent) -> None:
        """Forward the event to the wrapped callable."""
        self._handler(event)


class CompositeObserver(Observer):
    """Fans events out to several observers with failure isolation."""

    def __init__(self, observers: Iterable[Observer] = ()) -> None:
        self._observers = tuple(observers)

    @property
    def observers(self) -> tuple[Observer, ...]:
        """The composed observers, in registration order."""
        return self._observers

    def on_event(self, event: AgentEvent) -> None:
        """Deliver the event to every observer."""
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception:
                _LOGGER.exception(
                    "observer %s failed to record %s",
                    type(observer).__name__,
                    event.name,
                )

    @property
    def callback_handlers(self) -> tuple[BaseCallbackHandler, ...]:
        """The callback handlers of every observer."""
        return tuple(
            handler
            for observer in self._observers
            for handler in observer.callback_handlers
        )

    def flush(self) -> None:
        """Flush every observer."""
        for observer in self._observers:
            try:
                observer.flush()
            except Exception:
                _LOGGER.exception(
                    "observer %s failed to flush", type(observer).__name__
                )

    def shutdown(self) -> None:
        """Shut every observer down in reverse registration order."""
        for observer in reversed(self._observers):
            try:
                observer.shutdown()
            except Exception:
                _LOGGER.exception(
                    "observer %s failed to shut down", type(observer).__name__
                )


def compose_observers(
    observers: Iterable[Observer | Callable[[AgentEvent], object]],
) -> CompositeObserver:
    """Normalize observers and plain callables into one composite observer."""
    resolved: list[Observer] = []
    for observer in observers:
        if isinstance(observer, Observer):
            resolved.append(observer)
        elif callable(observer):
            resolved.append(CallableObserver(observer))
        else:
            raise TypeError(
                "observers must be Observer instances or callables, "
                f"not {type(observer).__name__}"
            )
    return CompositeObserver(resolved)
