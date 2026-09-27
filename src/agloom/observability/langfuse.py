"""Langfuse runtime events and LangChain model-call tracing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agloom.observability.base import Observer
from agloom.runtime.events import AgentEvent


class LangfuseObserver(Observer):
    def __init__(
        self,
        *,
        client: Langfuse | None = None,
        callback_handler: CallbackHandler | None = None,
        event_metadata: Mapping[str, Any] | None = None,
        **client_options: Any,
    ) -> None:
        if client is not None and client_options:
            raise ValueError("client options cannot be supplied with a Langfuse client")
        self.client = client or Langfuse(**client_options)
        public_key = client_options.get("public_key")
        self._callback_handler = callback_handler or CallbackHandler(
            public_key=public_key
        )
        self._event_metadata = dict(event_metadata or {})

    def on_event(self, event: AgentEvent) -> None:
        trace_id = self.client.create_trace_id(seed=event.run_id)
        level = "ERROR" if event.name == "ExecutionFailed" else "DEFAULT"
        self.client.create_event(
            trace_context={"trace_id": trace_id},
            name=event.name,
            metadata={
                **self._event_metadata,
                "agent_id": event.agent_id,
                "run_id": event.run_id,
                "topology": event.topology,
                **event.payload,
            },
            level=level,
            status_message=event.payload.get("error"),
        )

    @property
    def callback_handlers(self) -> tuple[CallbackHandler, ...]:
        return (self._callback_handler,)

    def xai_observability(self) -> Any:
        from langgraph_xai import LangfuseObservability

        return LangfuseObservability(self.client)

    def flush(self) -> None:
        self.client.flush()

    def shutdown(self) -> None:
        self.client.shutdown()
