from __future__ import annotations

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from prometheus_client import CollectorRegistry, generate_latest

from agloom import (
    AgentEvent,
    LangfuseObserver,
    LoggingObserver,
    Observer,
    PrometheusObserver,
    create_agent,
)
from agloom.observability.application import (
    application_observers,
    observability_status,
)

DIRECT_ANALYSIS = '{"complexity":0,"estimated_steps":1}'


class RecordingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.starts += 1

    def on_llm_end(self, response, **kwargs):
        self.ends += 1


class RecordingObserver(Observer):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.handler = RecordingHandler()
        self.flushed = False
        self.stopped = False

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)

    @property
    def callback_handlers(self):
        return (self.handler,)

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.stopped = True


def test_observer_receives_runtime_events_and_model_callbacks() -> None:
    observer = RecordingObserver()
    agent = create_agent(
        model=FakeListChatModel(responses=[DIRECT_ANALYSIS, "Hello"]),
        tools=[],
        observers=[observer],
    )

    result = agent.invoke("Say hello")
    agent.close_observability()

    assert result.content == "Hello"
    assert observer.handler.starts == 2
    assert observer.handler.ends == 2
    assert [event.name for event in observer.events] == [
        "AgentCreated",
        "HarnessStarted",
        "TopologySelectionStarted",
        "DirectExecutionStarted",
        "ExecutionCompleted",
    ]
    assert observer.flushed
    assert observer.stopped


def test_observer_model_callbacks_reach_topology_children() -> None:
    observer = RecordingObserver()
    agent = create_agent(
        model=FakeListChatModel(responses=["Hello"]),
        tools=[],
        pattern="react",
        observers=[observer],
    )

    result = agent.invoke("Say hello")

    assert result.content == "Hello"
    assert observer.handler.starts == 1
    assert observer.handler.ends == 1


def test_legacy_event_sink_remains_supported() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=FakeListChatModel(responses=["Hello"]),
        tools=[],
        pattern="react",
        event_sink=events.append,
    )

    agent.invoke("Say hello")

    assert any(event.name == "ExecutionCompleted" for event in events)


def test_prometheus_observer_exports_lifecycle_metrics() -> None:
    registry = CollectorRegistry()
    observer = PrometheusObserver(registry=registry)
    agent = create_agent(
        model=FakeListChatModel(responses=["Hello"]),
        tools=[],
        pattern="react",
        observers=[observer],
    )

    agent.invoke("Say hello")
    metrics = generate_latest(registry).decode()

    assert 'agloom_events_total{event="ExecutionCompleted"' in metrics
    assert 'agloom_executions_total{status="completed",topology="react"} 1.0' in metrics
    assert "agloom_execution_duration_seconds_count" in metrics


def test_prometheus_http_server_lifecycle() -> None:
    observer = PrometheusObserver(registry=CollectorRegistry())

    observer.start_http_server(port=0)
    observer.shutdown()


def test_application_observers_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGLOOM_OBSERVABILITY_ENABLED", raising=False)
    assert application_observers("example", default_metrics_port=9464) == ()
    assert observability_status(9464) == {
        "enabled": False,
        "prometheus": False,
        "metrics_port": 9464,
        "langfuse": False,
        "xai": True,
    }


def test_application_observers_require_complete_langfuse_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="must be configured together"):
        application_observers("example", default_metrics_port=0)


def test_logging_observer_emits_structured_event() -> None:
    class RecordingLogger:
        def __init__(self) -> None:
            self.events = []

        def info(self, event, **event_kw):
            self.events.append((event, event_kw))

    logger = RecordingLogger()
    observer = LoggingObserver(logger)
    event = AgentEvent(
        name="ExecutionCompleted",
        agent_id="agent-1",
        run_id="run-1",
        topology="react",
    )

    observer.on_event(event)

    assert logger.events[0][0] == "agloom_event"
    assert logger.events[0][1]["agloom_event"] == "ExecutionCompleted"
    assert logger.events[0][1]["run_id"] == "run-1"


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.events = []
        self.flushed = False
        self.stopped = False

    def create_trace_id(self, *, seed=None):
        return f"trace:{seed}"

    def create_event(self, **kwargs):
        self.events.append(kwargs)

    def flush(self):
        self.flushed = True

    def shutdown(self):
        self.stopped = True


def test_langfuse_observer_records_events_and_flushes() -> None:
    client = FakeLangfuseClient()
    handler = RecordingHandler()
    observer = LangfuseObserver(
        client=client,
        callback_handler=handler,
        event_metadata={"application_id": "test-application"},
    )
    event = AgentEvent(
        name="ExecutionFailed",
        agent_id="agent-1",
        run_id="run-1",
        payload={"error": "ValueError"},
    )

    observer.on_event(event)
    observer.flush()
    observer.shutdown()

    assert client.events[0]["trace_context"] == {"trace_id": "trace:run-1"}
    assert client.events[0]["level"] == "ERROR"
    assert client.events[0]["metadata"]["application_id"] == "test-application"
    assert observer.callback_handlers == (handler,)
    assert client.flushed
    assert client.stopped


def test_langfuse_observer_backs_automatic_xai_observability() -> None:
    from langgraph_xai import LangfuseObservability, ObservabilityProvider

    client = FakeLangfuseClient()
    observer = LangfuseObserver(
        client=client,
        callback_handler=RecordingHandler(),
    )
    agent = create_agent(
        model=FakeListChatModel(responses=["Hello"]),
        tools=[],
        pattern="react",
        observers=[observer],
    )

    provider = agent.xai.registry.require(ObservabilityProvider)

    assert isinstance(provider, LangfuseObservability)
    assert provider.client is client


def test_observer_failures_are_not_silenced() -> None:
    class BrokenObserver(Observer):
        def on_event(self, event: AgentEvent) -> None:
            raise RuntimeError("telemetry unavailable")

    with pytest.raises(RuntimeError, match="telemetry unavailable"):
        create_agent(
            model=FakeListChatModel(responses=["unused"]),
            tools=[],
            observers=[BrokenObserver()],
        )
