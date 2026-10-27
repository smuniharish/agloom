"""Prometheus metrics for runtime, policy, and Loop Engineering events."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
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
from agloom.observability.events import AgentEvent, EventName

_TERMINAL_STATUS: dict[str, str] = {
    EventName.EXECUTION_COMPLETED: "completed",
    EventName.EXECUTION_FAILED: "failed",
    EventName.EXECUTION_CANCELLED: "cancelled",
    EventName.EXECUTION_PAUSED: "paused",
}
_POLICY_OUTCOMES: dict[str, str] = {
    EventName.TOOL_CALL_DENIED: "denied",
    EventName.APPROVAL_REQUIRED: "approval_required",
    EventName.ACTION_EXECUTED: "executed",
    EventName.ACTION_REPLAYED: "replayed",
    EventName.ACTION_FAILED: "failed",
}
_LOOP_TERMINAL = frozenset(
    {EventName.LOOP_COMPLETED, EventName.LOOP_TERMINATED, EventName.LOOP_PAUSED}
)
_MAX_TRACKED_RUNS = 10_000
_PENDING = "pending"


class PrometheusObserver(Observer):
    """Exports runtime, policy, and Loop Engineering metrics to Prometheus.

    Metrics (with the configured namespace prefix):

    - `events_total{event,topology}`
    - `executions_total{status,topology}` and
      `execution_duration_seconds{status,topology}`
    - `active_executions`
    - `policy_actions_total{tool,outcome}`
    - `loop_iterations_total{progress,topology}`,
      `loop_decisions_total{action}`, and `loop_terminations_total{reason}`

    Args:
        registry: The collector registry that receives the metrics.
        namespace: The metric name prefix.
    """

    def __init__(
        self,
        *,
        registry: CollectorRegistry = REGISTRY,
        namespace: str = "agloom",
    ) -> None:
        self.registry = registry
        self._events = Counter(
            "events",
            "Runtime events.",
            ("event", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._executions = Counter(
            "executions",
            "Finished executions.",
            ("status", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._duration = Histogram(
            "execution_duration_seconds",
            "Execution duration.",
            ("status", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._active = Gauge(
            "active_executions",
            "Executions in progress.",
            namespace=namespace,
            registry=registry,
        )
        self._policy_actions = Counter(
            "policy_actions",
            "Policy-governed tool calls.",
            ("tool", "outcome"),
            namespace=namespace,
            registry=registry,
        )
        self._loop_iterations = Counter(
            "loop_iterations",
            "Completed loop iterations.",
            ("progress", "topology"),
            namespace=namespace,
            registry=registry,
        )
        self._loop_decisions = Counter(
            "loop_decisions",
            "Loop decisions.",
            ("action",),
            namespace=namespace,
            registry=registry,
        )
        self._loop_terminations = Counter(
            "loop_terminations",
            "Loop outcomes.",
            ("reason",),
            namespace=namespace,
            registry=registry,
        )
        self._started: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._server: tuple[Any, threading.Thread] | None = None

    def on_event(self, event: AgentEvent) -> None:
        """Update the metrics affected by the event."""
        topology = event.topology or _PENDING
        self._events.labels(event.name, topology).inc()
        if (outcome := _POLICY_OUTCOMES.get(event.name)) is not None:
            self._policy_actions.labels(
                str(event.payload.get("tool", "unknown")), outcome
            ).inc()
        if event.name == EventName.LOOP_ITERATION_COMPLETED:
            progress = str(event.payload.get("progress_status", "unknown"))
            self._loop_iterations.labels(progress, topology).inc()
            self._loop_decisions.labels(
                str(event.payload.get("action", "unknown"))
            ).inc()
        if event.name in _LOOP_TERMINAL:
            reason = str(event.payload.get("termination_reason", "unknown"))
            self._loop_terminations.labels(reason).inc()
        self._track_execution(event, topology)

    def _track_execution(self, event: AgentEvent, topology: str) -> None:
        with self._lock:
            if event.name == EventName.EXECUTION_STARTED:
                self._started[event.run_id] = (time.monotonic(), _PENDING)
                if len(self._started) > _MAX_TRACKED_RUNS:
                    self._started.popitem(last=False)
                self._active.inc()
                return
            started = self._started.get(event.run_id)
            if started is None:
                return
            if event.name == EventName.TOPOLOGY_SELECTED:
                self._started[event.run_id] = (started[0], topology)
                return
            if event.name == EventName.DIRECT_EXECUTION_STARTED:
                self._started[event.run_id] = (started[0], "direct")
                return
            status = _TERMINAL_STATUS.get(event.name)
            if status is None:
                return
            del self._started[event.run_id]
            started_at, tracked_topology = started
            resolved = topology if topology != _PENDING else tracked_topology
            self._active.dec()
            self._executions.labels(status, resolved).inc()
            self._duration.labels(status, resolved).observe(
                time.monotonic() - started_at
            )

    def start_http_server(
        self, port: int = 9464, *, address: str = "127.0.0.1"
    ) -> None:
        """Serve this observer's registry over HTTP for Prometheus scraping.

        Args:
            port: The TCP port. Use `0` to bind an ephemeral port.
            address: The bind address; loopback by default.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._server is not None:
            raise RuntimeError("the Prometheus metrics server is already running")
        self._server = start_http_server(port, addr=address, registry=self.registry)

    @property
    def server_port(self) -> int | None:
        """The bound port of the metrics server, or `None` when it is not running."""
        if self._server is None:
            return None
        return int(self._server[0].server_port)

    def shutdown(self) -> None:
        """Stop the metrics server when it is running."""
        if self._server is None:
            return
        server, thread = self._server
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        self._server = None
