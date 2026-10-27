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
    assert telemetry.status()["langfuse"] is False


def test_metric_names_match_the_shared_dashboard() -> None:
    telemetry = build_telemetry(SETTINGS, environ={})
    for name in ("ExecutionStarted", "TopologySelected", "ExecutionFailed"):
        telemetry.prometheus.on_event(
            AgentEvent(name=name, agent_id="a", run_id="r", topology="react")
        )
    telemetry.audits.labels("fail").inc()
    registry = telemetry.registry

    assert (
        registry.get_sample_value(
            f"{METRICS_NAMESPACE}_executions_total",
            {"status": "failed", "topology": "react"},
        )
        == 1
    )
    assert registry.get_sample_value(f"{METRICS_NAMESPACE}_active_executions") == 0
    assert (
        registry.get_sample_value(
            f"{METRICS_NAMESPACE}_audits_total", {"verdict": "fail"}
        )
        == 1
    )


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
    assert f"{METRICS_NAMESPACE}_audits_total" in body


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
    finally:
        telemetry.shutdown()


def test_a_single_langfuse_key_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="configured together"):
        build_telemetry(SETTINGS, environ={"LANGFUSE_SECRET_KEY": "sk-lf-test"})
