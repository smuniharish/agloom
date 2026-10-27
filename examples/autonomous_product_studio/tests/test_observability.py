from __future__ import annotations

import urllib.request

import pytest
from agloom import AgentEvent, EventName, Observer, PrometheusObserver

from backend import observability


def test_observability_is_off_unless_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGLOOM_OBSERVABILITY_ENABLED", raising=False)
    assert observability.build_observers() == ()
    assert observability.observability_status() == {
        "enabled": False,
        "prometheus": False,
        "metrics_port": 9469,
        "langfuse": False,
    }


def test_langfuse_keys_must_be_configured_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-only")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="configured together"):
        observability.langfuse_configured()


def test_langfuse_server_comes_from_langfuse_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class RecordingLangfuse(Observer):
        def __init__(self, **options: object) -> None:
            created.append(options)

        def on_event(self, event: AgentEvent) -> None:
            return None

    class UnservedPrometheus(Observer):
        """Keeps the process-wide Prometheus registry free for the metrics test."""

        def __init__(self, **options: object) -> None:
            return None

        def on_event(self, event: AgentEvent) -> None:
            return None

        def start_http_server(self, port: int, address: str) -> None:
            return None

    monkeypatch.setattr(observability, "LangfuseObserver", RecordingLangfuse)
    monkeypatch.setattr(observability, "PrometheusObserver", UnservedPrometheus)
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    monkeypatch.delenv("AGLOOM_LANGFUSE_ENABLED", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.test:3000")
    observability.shutdown_observers(observability.build_observers())

    [options] = created
    assert options["base_url"] == "http://langfuse.test:3000"
    assert options["event_metadata"] == {"application_id": observability.APPLICATION_ID}


def test_enabled_observability_serves_application_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGLOOM_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("AGLOOM_METRICS_PORT", "0")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    observers = observability.build_observers()
    try:
        [prometheus] = [
            item for item in observers if isinstance(item, PrometheusObserver)
        ]
        for observer in observers:
            observer.on_event(
                AgentEvent(
                    name=EventName.APPROVAL_REQUIRED,
                    agent_id="a",
                    run_id="r",
                    payload={"tool": "publish_release"},
                )
            )
        with urllib.request.urlopen(
            f"http://127.0.0.1:{prometheus.server_port}/metrics", timeout=5
        ) as response:
            metrics = response.read().decode()
    finally:
        observability.shutdown_observers(observers)

    assert (
        'agloom_autonomous_product_studio_policy_actions_total{outcome="approval_required",tool="publish_release"} 1.0'
        in metrics
    )
    assert observability.observability_status()["prometheus"] is True


def test_shutdown_isolates_failing_observers() -> None:
    calls: list[str] = []

    class Broken(Observer):
        def on_event(self, event: AgentEvent) -> None:
            return None

        def flush(self) -> None:
            raise RuntimeError("telemetry backend unavailable")

    class Healthy(Observer):
        def on_event(self, event: AgentEvent) -> None:
            return None

        def shutdown(self) -> None:
            calls.append("shutdown")

    observability.shutdown_observers([Broken(), Healthy()])
    assert calls == ["shutdown"]
