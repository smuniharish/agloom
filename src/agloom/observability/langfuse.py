"""Langfuse tracing of runtime events and LangChain model calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agloom.observability.base import Observer
from agloom.observability.events import AgentEvent, EventName

_ERROR_EVENTS = frozenset({EventName.EXECUTION_FAILED, EventName.ACTION_FAILED})
_WARNING_EVENTS = frozenset({EventName.TOOL_CALL_DENIED, EventName.EXECUTION_CANCELLED})


class LangfuseObserver(Observer):
    """Records runtime events in Langfuse and traces model, tool, and graph runs.

    Events of one invocation share a deterministic trace derived from its run
    identifier. The LangChain callback handler traces every model, tool, and
    graph call made by the agent.

    Args:
        client: An existing Langfuse client. When omitted, a client is created
            from `client_options` (or from `LANGFUSE_*` environment variables).
        callback_handler: An existing LangChain callback handler. By default a
            handler bound to the default Langfuse client is created; when you
            pass a client for another project, pass a matching handler, for
            example `CallbackHandler(public_key=...)`.
        event_metadata: Metadata added to every recorded event.
        **client_options: Keyword arguments for `langfuse.Langfuse`.

    Raises:
        ValueError: If both `client` and `client_options` are supplied.
    """

    def __init__(
        self,
        *,
        client: Langfuse | None = None,
        callback_handler: CallbackHandler | None = None,
        event_metadata: Mapping[str, Any] | None = None,
        **client_options: Any,
    ) -> None:
        if client is not None and client_options:
            raise ValueError("client options cannot be combined with a Langfuse client")
        self.client = client or Langfuse(**client_options)
        self._callback_handler = callback_handler or CallbackHandler(
            public_key=client_options.get("public_key")
        )
        self._event_metadata = dict(event_metadata or {})

    def on_event(self, event: AgentEvent) -> None:
        """Record the event as a Langfuse event observation."""
        if event.name in _ERROR_EVENTS:
            level = "ERROR"
        elif event.name in _WARNING_EVENTS:
            level = "WARNING"
        else:
            level = "DEFAULT"
        error = event.payload.get("error")
        self.client.create_event(
            trace_context={"trace_id": self.client.create_trace_id(seed=event.run_id)},
            name=event.name,
            metadata={
                **self._event_metadata,
                "agent_id": event.agent_id,
                "run_id": event.run_id,
                "topology": event.topology,
                **event.payload,
            },
            level=level,
            status_message=str(error) if error is not None else None,
        )

    @property
    def callback_handlers(self) -> tuple[CallbackHandler, ...]:
        """The Langfuse LangChain callback handler."""
        return (self._callback_handler,)

    def flush(self) -> None:
        """Flush buffered Langfuse data."""
        self.client.flush()

    def shutdown(self) -> None:
        """Flush and shut the Langfuse client down."""
        self.client.shutdown()
