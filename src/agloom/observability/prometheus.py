"""Prometheus metrics suitable for live Grafana dashboards."""

from __future__ import annotations

import threading
import time
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from agloom.observability.base import Observer
from agloom.runtime.events import AgentEvent

_TERMINAL_STATUS = {
    "ExecutionCompleted": "completed",
    "ExecutionFailed": "failed",
    "ExecutionCancelled": "cancelled",
    "ExecutionPaused": "paused",
}


class PrometheusObserver(Observer):
    def __init__(
        self,
        *,
        registry: CollectorRegistry = REGISTRY,
        namespace: str = "agloom",
    ) -> None:
        self.registry = registry
        self._events = Counter(
            "events",
            "Agloom runtime events.",
            ("event", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._executions = Counter(
            "executions",
            "Completed Agloom execution lifecycles.",
            ("status", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._duration = Histogram(
            "execution_duration_seconds",
            "Agloom execution duration.",
            ("status", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._active = Gauge(
            "active_executions",
            "Currently active Agloom executions.",
            namespace=namespace,
            registry=registry,
        )
        self._started: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._server: Any = None
        self._server_thread: threading.Thread | None = None

    def on_event(self, event: AgentEvent) -> None:
        topology = event.topology or "direct_or_pending"
        self._events.labels(event.name, topology).inc()
        with self._lock:
            if event.name == "HarnessStarted":
                self._started[event.run_id] = (time.monotonic(), topology)
                self._active.inc()
                return
            if event.name == "TopologySelected" and event.run_id in self._started:
                started_at, _ = self._started[event.run_id]
                self._started[event.run_id] = (started_at, topology)
                return
            if event.name == "DirectExecutionStarted" and event.run_id in self._started:
                started_at, _ = self._started[event.run_id]
                self._started[event.run_id] = (started_at, "direct")
                return
            status = _TERMINAL_STATUS.get(event.name)
            if status is None:
                return
            started = self._started.pop(event.run_id, None)
            if started is None:
                return
            started_at, started_topology = started
            resolved_topology = (
                topology if topology != "direct_or_pending" else started_topology
            )
            self._active.dec()
            self._executions.labels(status, resolved_topology).inc()
            self._duration.labels(status, resolved_topology).observe(
                time.monotonic() - started_at
            )

    def start_http_server(
        self,
        port: int = 9464,
        *,
        address: str = "127.0.0.1",
    ) -> None:
        if self._server is not None:
            raise RuntimeError("Prometheus metrics server is already running")
        self._server, self._server_thread = start_http_server(
            port,
            addr=address,
            registry=self.registry,
        )

    def shutdown(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._server = None
        self._server_thread = None
