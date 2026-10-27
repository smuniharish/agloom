"""Application telemetry built from Agloom observers."""

from __future__ import annotations

import dataclasses

import httpx
import pytest
from backend.observability import METRICS_NAMESPACE, build_telemetry
from backend.settings import Settings

from agloom import AgentEvent, LangfuseObserver, LoggingObserver, PrometheusObserver

SETTINGS = Settings.from_environment({"DATABASE_URL": "postgresql://db.invalid/x"})


def test_logs_and_metrics_are_always_observed() -> None:
    telemetry = build_telemetry(SETTINGS, environ={})

    assert [type(observer) for observer in telemetry.observers] == [
        LoggingObserver,
        PrometheusObserver,
    ]
    assert telemetry.langfuse is None
    assert telemetry.status() == {
        "logging": True,
        "prometheus": False,
        "metrics_port": None,
        "langfuse": False,
    }


def test_metric_names_match_the_shared_dashboard() -> None:
    telemetry = build_telemetry(SETTINGS, environ={})
    for name in ("ExecutionStarted", "TopologySelected", "ExecutionCompleted"):
        telemetry.prometheus.on_event(
            AgentEvent(name=name, agent_id="a", run_id="r", topology="react")
        )

    def sample(metric: str, **labels: str) -> float | None:
        return telemetry.registry.get_sample_value(
            f"{METRICS_NAMESPACE}_{metric}", labels
        )

    assert sample("executions_total", status="completed", topology="react") == 1
    assert sample("events_total", event="ExecutionStarted", topology="react") == 1
    assert sample("active_executions") == 0


def test_metrics_endpoint_serves_the_application_registry() -> None:
    telemetry = dataclasses.replace(
        build_telemetry(SETTINGS, environ={}), metrics_enabled=True, metrics_port=0
    )
    telemetry.start()
    try:
        port = telemetry.prometheus.server_port
        assert port is not None
        body = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=10).text
    finally:
        telemetry.shutdown()

    assert f"{METRICS_NAMESPACE}_active_executions" in body
    assert telemetry.prometheus.server_port is None


def test_langfuse_is_enabled_by_a_complete_key_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    telemetry = build_telemetry(
        SETTINGS,
        environ={
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
            "LANGFUSE_BASE_URL": "http://127.0.0.1:9",
        },
    )
    try:
        assert isinstance(telemetry.langfuse, LangfuseObserver)
        assert telemetry.status()["langfuse"] is True
    finally:
        telemetry.shutdown()


def test_a_single_langfuse_key_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="configured together"):
        build_telemetry(SETTINGS, environ={"LANGFUSE_PUBLIC_KEY": "pk-lf-test"})
