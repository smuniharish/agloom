from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from backend.app import create_app
from backend.knowledge import KnowledgeBase
from backend.observability import (
    Telemetry,
    langfuse_configured,
    observability_status,
    start_telemetry,
)
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agloom import (
    AgentEvent,
    EventName,
    LoggingObserver,
    Observer,
    PolicyManager,
    PrometheusObserver,
)

METADATA = {"application_id": "enterprise-knowledge-assistant", "tenant_id": "t"}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AGLOOM_OBSERVABILITY_ENABLED",
        "AGLOOM_METRICS_PORT",
        "AGLOOM_METRICS_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeLangfuseObserver(Observer):
    def __init__(self, *, event_metadata: dict[str, str]) -> None:
        self.event_metadata = event_metadata

    def on_event(self, event: AgentEvent) -> None:
        return


def test_telemetry_is_disabled_by_default() -> None:
    telemetry = start_telemetry("eka", default_metrics_port=9464, metadata=METADATA)
    assert telemetry.observers == ()
    assert observability_status(9464) == {
        "enabled": False,
        "prometheus": False,
        "metrics_port": 9464,
        "langfuse": False,
    }


def test_enabled_telemetry_serves_prometheus_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    telemetry = start_telemetry(
        "enterprise_knowledge_assistant", default_metrics_port=9464, metadata=METADATA
    )
    logging_observer, prometheus = telemetry.observers
    assert isinstance(logging_observer, LoggingObserver)
    assert isinstance(prometheus, PrometheusObserver)
    port = prometheus.server_port
    assert port
    try:
        for name in (EventName.EXECUTION_STARTED, EventName.EXECUTION_COMPLETED):
            prometheus.on_event(AgentEvent(name=name, agent_id="a", run_id="r"))
        metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=10).text
    finally:
        telemetry.shutdown()
    assert (
        'agloom_enterprise_knowledge_assistant_executions_total{status="completed"'
        in metrics
    )
    assert prometheus.server_port is None
    assert observability_status(9464)["metrics_port"] == 0


def test_langfuse_keys_must_be_configured_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    with pytest.raises(RuntimeError, match="configured together"):
        langfuse_configured()


def test_langfuse_observer_is_added_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "yes")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr("backend.observability.LangfuseObserver", FakeLangfuseObserver)
    telemetry = start_telemetry("eka", default_metrics_port=9464, metadata=METADATA)
    try:
        langfuse = telemetry.observers[-1]
        assert isinstance(langfuse, FakeLangfuseObserver)
        assert langfuse.event_metadata == METADATA
        assert observability_status(9464)["langfuse"] is True
    finally:
        telemetry.shutdown()


def test_application_passes_its_observers_to_the_agent(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    received: list[tuple[Observer, ...]] = []

    class Agent:
        async def ainvoke(self, input: Sequence[Any]) -> AIMessage:
            return AIMessage(content="Report it now. [IT-101-2]")

        def close(self) -> None:
            for observer in received[0]:
                observer.shutdown()

    def factory(
        _: KnowledgeBase, __: PolicyManager, observers: Sequence[Observer]
    ) -> Agent:
        received.append(tuple(observers))
        return Agent()

    with TestClient(create_app(kb, agent_factory=factory)) as client:
        assert client.get("/api/health").json()["observability"]["enabled"] is True
        assert client.post("/api/chat", json={"question": "lost device"}).is_success
    observers = received[0]
    assert [type(observer) for observer in observers] == [
        LoggingObserver,
        PrometheusObserver,
    ]
    prometheus = observers[1]
    assert isinstance(prometheus, PrometheusObserver)
    assert prometheus.server_port is None


def test_metrics_are_labeled_with_the_explicit_topology(
    kb: KnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    scripted_model: Callable[..., Any],
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    started: list[Telemetry] = []

    def recording_start(*args: Any, **kwargs: Any) -> Telemetry:
        telemetry = start_telemetry(*args, **kwargs)
        started.append(telemetry)
        return telemetry

    model = scripted_model(AIMessage(content="Report it now. [IT-101-2]"))
    monkeypatch.setattr("backend.app.chat_model", lambda: model)
    monkeypatch.setattr("backend.app.start_telemetry", recording_start)
    with TestClient(create_app(kb)) as client:
        assert client.post("/api/chat", json={"question": "lost device"}).is_success
        prometheus = started[0].observers[1]
        assert isinstance(prometheus, PrometheusObserver)
        url = f"http://127.0.0.1:{prometheus.server_port}/metrics"
        metrics = httpx.get(url, timeout=10).text
    expected = (
        "agloom_enterprise_knowledge_assistant_executions_total"
        '{status="completed",topology="react"} 1.0'
    )
    assert expected in metrics
