"""Lazy constructors and event adapter for the supported ecosystem packages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agloom.runtime.events import AgentEvent


def feedback_manager(**kwargs: Any) -> Any:
    """Construct the actual `feedback_manager.FeedbackManager`."""

    from feedback_manager import FeedbackManager

    return FeedbackManager(**kwargs)


def behavior_engine(
    *,
    policies: Iterable[Any] = (),
    patterns: Iterable[Any] | None = None,
    state_store: Any = None,
) -> Any:
    """Construct a BehaviorWeave engine using its public constructor."""

    from behaviorweave import BehaviorEngine

    return BehaviorEngine(
        policies=policies,
        patterns=patterns,
        state_store=state_store,
    )


def context_middleware(model: Any, **kwargs: Any) -> Any:
    """Construct ContextSage's LangChain agent middleware."""

    from contextsage import IntelligentSummarizationMiddleware

    return IntelligentSummarizationMiddleware(model=model, **kwargs)


def mcp_runtime(**kwargs: Any) -> Any:
    """Construct MCP capability-router's isolated runtime."""

    from mcp_capability_router import MCPRuntime

    return MCPRuntime(**kwargs)


def mcp_refresh_policy(**kwargs: Any) -> Any:
    """Construct the router policy used for MCP capability discovery."""

    from mcp_capability_router import RefreshPolicy

    return RefreshPolicy(**kwargs)


def refresh_engine(source: Any, operation: Any, **kwargs: Any) -> Any:
    """Construct RefreshEngine from an application-owned source and operation."""

    from refresh_engine import RefreshEngine

    return RefreshEngine(source, operation, **kwargs)


def xai_runtime(**kwargs: Any) -> Any:
    """Construct langgraph-xai's graph instrumentation runtime."""

    from langgraph_xai import XAIRuntime

    return XAIRuntime(**kwargs)


def with_structured_output(runnable: Any, schema: Any, **kwargs: Any) -> Any:
    """Wrap a LangChain Runnable with the actual xstructured API."""

    from xstructured import with_xstructured_output

    return with_xstructured_output(runnable, schema, **kwargs)


class BehaviorWeaveAdapter:
    """Translate Agloom's normalized events into BehaviorWeave observations."""

    def __init__(self, engine: Any, *, scope: str) -> None:
        if not scope.strip():
            raise ValueError("BehaviorWeave scope must not be empty")
        self.engine = engine
        self.scope = scope

    def __call__(self, event: AgentEvent) -> Any:
        from behaviorweave import BehaviorEvent, EventType, Outcome

        if event.name == "ExecutionFailed":
            normalized = BehaviorEvent.outcome_event(
                Outcome.FAILURE,
                scope=self.scope,
                event_id=f"{event.run_id}:{event.name}",
            )
        elif event.name == "ExecutionCompleted":
            normalized = BehaviorEvent.outcome_event(
                Outcome.SUCCESS,
                scope=self.scope,
                event_id=f"{event.run_id}:{event.name}",
            )
        else:
            normalized = BehaviorEvent(
                event_type=EventType.NODE_EXECUTION,
                scope=self.scope,
                event_id=f"{event.run_id}:{event.name}",
                run_id=event.run_id,
                node_name=event.name,
                agent_name=event.agent_id,
                metadata={"topology": event.topology, **event.payload},
            )
        return self.engine.process(normalized)
