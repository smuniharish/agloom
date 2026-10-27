"""Runtime events and observers."""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from agloom import (
    AgentEvent,
    EventName,
    LangfuseObserver,
    LoggingObserver,
    Observer,
    PrometheusObserver,
    create_agent,
)
from agloom.observability.base import CompositeObserver, compose_observers
from helpers import Call, ScriptedChatModel, analysis


def event(name: str, run_id: str = "run", **payload: Any) -> AgentEvent:
    topology = payload.pop("topology", None)
    return AgentEvent(
        name=name, agent_id="agent", run_id=run_id, topology=topology, payload=payload
    )


def test_events_are_immutable_and_named_uniquely() -> None:
    recorded = event(EventName.LOOP_COMPLETED, iterations=2)
    with pytest.raises(TypeError):
        recorded.payload["iterations"] = 3  # type: ignore[index]
    assert recorded.timestamp.tzinfo is not None
    assert len({member.value for member in EventName}) == len(EventName)


class Failing(Observer):
    def on_event(self, event: AgentEvent) -> None:
        raise RuntimeError("down")

    def flush(self) -> None:
        raise RuntimeError("down")

    def shutdown(self) -> None:
        raise RuntimeError("down")


def test_composition_isolates_failures(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[str] = []
    composite = compose_observers([Failing(), lambda item: seen.append(item.name)])
    assert isinstance(composite, CompositeObserver)
    assert len(composite.observers) == 2
    composite.on_event(event(EventName.AGENT_CREATED))
    composite.flush()
    composite.shutdown()
    assert seen == [EventName.AGENT_CREATED]
    assert "failed to record" in caplog.text
    assert "failed to flush" in caplog.text
    assert "failed to shut down" in caplog.text
    assert composite.callback_handlers == ()
    with pytest.raises(TypeError, match="Observer instances or callables"):
        compose_observers([42])  # type: ignore[list-item]


def test_logging_observer_levels() -> None:
    class Logger:
        def __init__(self) -> None:
            self.records: list[tuple[str, dict[str, Any]]] = []

        def info(self, message: str, **fields: Any) -> None:
            self.records.append(("info", fields))

        def warning(self, message: str, **fields: Any) -> None:
            self.records.append(("warning", fields))

    logger = Logger()
    observer = LoggingObserver(logger)
    observer.on_event(event(EventName.EXECUTION_STARTED))
    observer.on_event(event(EventName.TOOL_CALL_DENIED, tool="delete"))
    assert [level for level, _ in logger.records] == ["info", "warning"]
    assert logger.records[1][1]["payload"] == {"tool": "delete"}
    LoggingObserver().on_event(event(EventName.AGENT_CREATED))


class TestPrometheus:
    def observer(self) -> tuple[PrometheusObserver, CollectorRegistry]:
        registry = CollectorRegistry()
        return PrometheusObserver(registry=registry, namespace="test"), registry

    def test_execution_policy_and_loop_metrics(self) -> None:
        observer, registry = self.observer()
        for item in (
            event(EventName.EXECUTION_STARTED, run_id="r1"),
            event(EventName.TOPOLOGY_SELECTED, run_id="r1", topology="planner"),
            event(EventName.ACTION_EXECUTED, run_id="r1", tool="refund"),
            event(
                EventName.LOOP_ITERATION_COMPLETED,
                run_id="r1",
                progress_status="progress",
                action="continue",
            ),
            event(
                EventName.LOOP_COMPLETED,
                run_id="r1",
                termination_reason="objective_satisfied",
            ),
            event(EventName.EXECUTION_COMPLETED, run_id="r1"),
            event(EventName.EXECUTION_STARTED, run_id="r2"),
            event(EventName.DIRECT_EXECUTION_STARTED, run_id="r2"),
            event(EventName.EXECUTION_FAILED, run_id="r2", error="ValueError"),
            event(EventName.EXECUTION_COMPLETED, run_id="unknown"),
        ):
            observer.on_event(item)
        value = registry.get_sample_value
        assert (
            value(
                "test_executions_total", {"status": "completed", "topology": "planner"}
            )
            == 1
        )
        assert (
            value("test_executions_total", {"status": "failed", "topology": "direct"})
            == 1
        )
        assert value("test_active_executions") == 0
        assert (
            value(
                "test_policy_actions_total", {"tool": "refund", "outcome": "executed"}
            )
            == 1
        )
        assert (
            value(
                "test_loop_iterations_total",
                {"progress": "progress", "topology": "pending"},
            )
            == 1
        )
        assert value("test_loop_decisions_total", {"action": "continue"}) == 1
        assert (
            value("test_loop_terminations_total", {"reason": "objective_satisfied"})
            == 1
        )

    def test_metrics_endpoint(self) -> None:
        observer, _ = self.observer()
        assert observer.server_port is None
        observer.shutdown()
        observer.start_http_server(0)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                observer.start_http_server(0)
            port = observer.server_port
            assert port
            observer.on_event(event(EventName.AGENT_CREATED))
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=5
            ) as response:
                body = response.read().decode()
            assert "test_events_total" in body
        finally:
            observer.shutdown()
        assert observer.server_port is None

    def test_agent_metrics_are_labeled_with_the_topology(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        observer, registry = self.observer()
        agent = create_agent(
            model=model(None, "ok"), pattern="react", observers=[observer]
        )
        agent.invoke("hi")
        assert (
            registry.get_sample_value(
                "test_executions_total", {"status": "completed", "topology": "react"}
            )
            == 1
        )


class FakeLangfuse:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.flushed = 0
        self.closed = 0

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace-{seed}"

    def create_event(self, **fields: Any) -> None:
        self.events.append(fields)

    def flush(self) -> None:
        self.flushed += 1

    def shutdown(self) -> None:
        self.closed += 1


class TestLangfuse:
    def test_records_events_with_levels(self) -> None:
        client = FakeLangfuse()
        handler = object()
        observer = LangfuseObserver(
            client=client,  # type: ignore[arg-type]
            callback_handler=handler,  # type: ignore[arg-type]
            event_metadata={"service": "tests"},
        )
        observer.on_event(event(EventName.EXECUTION_STARTED, run_id="r"))
        observer.on_event(event(EventName.EXECUTION_FAILED, run_id="r", error="Boom"))
        observer.on_event(event(EventName.TOOL_CALL_DENIED, run_id="r", tool="t"))
        levels = [item["level"] for item in client.events]
        assert levels == ["DEFAULT", "ERROR", "WARNING"]
        assert client.events[1]["status_message"] == "Boom"
        assert client.events[0]["trace_context"] == {"trace_id": "trace-r"}
        assert client.events[0]["metadata"]["service"] == "tests"
        assert observer.callback_handlers == (handler,)
        observer.flush()
        observer.shutdown()
        assert (client.flushed, client.closed) == (1, 1)

    def test_rejects_client_with_options(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            LangfuseObserver(client=FakeLangfuse(), public_key="pk")  # type: ignore[arg-type]

    def test_builds_its_own_client(self) -> None:
        observer = LangfuseObserver(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            base_url="http://127.0.0.1:9",
            tracing_enabled=False,
        )
        assert observer.callback_handlers
        observer.shutdown()


def test_agent_events_reach_custom_observers(
    model: Callable[..., ScriptedChatModel],
) -> None:
    received: list[AgentEvent] = []

    class Recorder(Observer):
        def on_event(self, event: AgentEvent) -> None:
            received.append(event)

    def respond(call: Call) -> Any:
        return analysis() if call.wants("TaskAnalysis") else "hello"

    create_agent(model=model(respond), observers=[Recorder()]).invoke("hi")
    names = [item.name for item in received]
    assert names[0] == EventName.AGENT_CREATED
    assert names[-1] == EventName.EXECUTION_COMPLETED
    run_ids = {item.run_id for item in received[1:]}
    assert len(run_ids) == 1


def test_prometheus_tracking_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import agloom.observability.prometheus as prometheus

    monkeypatch.setattr(prometheus, "_MAX_TRACKED_RUNS", 1)
    registry = CollectorRegistry()
    observer = PrometheusObserver(registry=registry, namespace="bounded")
    observer.on_event(event(EventName.EXECUTION_STARTED, run_id="first"))
    observer.on_event(event(EventName.EXECUTION_STARTED, run_id="second"))
    observer.on_event(event(EventName.EXECUTION_COMPLETED, run_id="first"))
    observer.on_event(event(EventName.EXECUTION_COMPLETED, run_id="second"))
    completed = registry.get_sample_value(
        "bounded_executions_total", {"status": "completed", "topology": "pending"}
    )
    assert completed == 1
