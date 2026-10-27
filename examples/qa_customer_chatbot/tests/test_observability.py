from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from backend.app import Draft, create_app
from backend.observability import (
    Telemetry,
    langfuse_configured,
    observability_status,
    start_telemetry,
)
from backend.store import Store
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage

from agloom import (
    AgentEvent,
    EventName,
    LoggingObserver,
    Observer,
    PolicyManager,
    PrometheusObserver,
)

METADATA = {"application_id": "qa-customer-chatbot", "tenant_id": "t"}


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
    telemetry = start_telemetry("qa", default_metrics_port=9465, metadata=METADATA)
    assert telemetry.observers == ()
    assert observability_status(9465) == {
        "enabled": False,
        "prometheus": False,
        "metrics_port": 9465,
        "langfuse": False,
    }


def test_enabled_telemetry_serves_prometheus_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    telemetry = start_telemetry(
        "qa_customer_chatbot", default_metrics_port=9465, metadata=METADATA
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
    assert 'agloom_qa_customer_chatbot_executions_total{status="completed"' in metrics
    assert prometheus.server_port is None


def test_langfuse_keys_must_be_configured_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    with pytest.raises(RuntimeError, match="configured together"):
        langfuse_configured()


def test_langfuse_observer_is_added_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "1")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr("backend.observability.LangfuseObserver", FakeLangfuseObserver)
    telemetry = start_telemetry("qa", default_metrics_port=9465, metadata=METADATA)
    try:
        langfuse = telemetry.observers[-1]
        assert isinstance(langfuse, FakeLangfuseObserver)
        assert langfuse.event_metadata == METADATA
        assert observability_status(9465)["langfuse"] is True
    finally:
        telemetry.shutdown()


def test_application_passes_its_observers_to_the_agent(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    received: list[tuple[Observer, ...]] = []

    class Drafter:
        async def draft(
            self, messages: Sequence[BaseMessage], *, conversation_id: str
        ) -> Draft:
            return Draft(
                answer="Unused items can be returned. [POL-RETURNS]",
                policy_ids=["POL-RETURNS"],
            )

        def close(self) -> None:
            for observer in received[0]:
                observer.shutdown()

    def factory(_: Store, __: PolicyManager, observers: Sequence[Observer]) -> Drafter:
        received.append(tuple(observers))
        return Drafter()

    with TestClient(create_app(store, drafter_factory=factory)) as client:
        assert client.get("/api/health").json()["observability"]["enabled"] is True
        assert client.post("/api/chat", json={"message": "return policy"}).is_success
    observers = received[0]
    assert [type(observer) for observer in observers] == [
        LoggingObserver,
        PrometheusObserver,
    ]
    prometheus = observers[1]
    assert isinstance(prometheus, PrometheusObserver)
    assert prometheus.server_port is None


def test_metrics_are_labeled_with_the_explicit_topology(
    store: Store,
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

    draft = {
        "answer": "Unused items can be returned within 30 days. [POL-RETURNS]",
        "policy_ids": ["POL-RETURNS"],
    }
    model = scripted_model(
        AIMessage(content=f"<xstructured>{json.dumps(draft)}</xstructured>")
    )
    monkeypatch.setattr("backend.app.chat_model", lambda: model)
    monkeypatch.setattr("backend.app.start_telemetry", recording_start)
    with TestClient(create_app(store)) as client:
        reply = client.post("/api/chat", json={"message": "What is the return policy?"})
        assert reply.is_success, reply.text
        prometheus = started[0].observers[1]
        assert isinstance(prometheus, PrometheusObserver)
        url = f"http://127.0.0.1:{prometheus.server_port}/metrics"
        metrics = httpx.get(url, timeout=10).text
    expected = (
        "agloom_qa_customer_chatbot_executions_total"
        '{status="completed",topology="react"} 1.0'
    )
    assert expected in metrics
