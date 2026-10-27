"""Application telemetry: structured logs, Prometheus metrics, and Langfuse traces.

Agloom ships the observers; this module decides which ones the application
uses. Metric names keep the `agloom_web_compliance_auditor_` prefix that the
shared Grafana dashboard queries, next to the application's own audit counter.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter

from agloom import (
    AgentEvent,
    LangfuseObserver,
    LoggingObserver,
    Observer,
    PrometheusObserver,
)

from .settings import APPLICATION_ID, TENANT_ID, Settings

METRICS_NAMESPACE = "agloom_web_compliance_auditor"

EventObserver = Observer | Callable[[AgentEvent], object]


@dataclass
class Telemetry:
    """The observers of one application instance and their metrics endpoint."""

    observers: tuple[EventObserver, ...]
    registry: CollectorRegistry
    prometheus: PrometheusObserver
    audits: Counter
    langfuse: LangfuseObserver | None
    metrics_enabled: bool
    metrics_host: str
    metrics_port: int

    def start(self) -> None:
        """Serve the metrics registry when metrics are enabled."""
        if self.metrics_enabled:
            self.prometheus.start_http_server(
                self.metrics_port, address=self.metrics_host
            )

    def status(self) -> dict[str, object]:
        """Return the non-sensitive telemetry configuration."""
        return {
            "logging": True,
            "prometheus": self.metrics_enabled,
            "metrics_port": self.prometheus.server_port,
            "langfuse": self.langfuse is not None,
        }

    def shutdown(self) -> None:
        """Flush and stop every observer."""
        for observer in reversed(self.observers):
            if isinstance(observer, Observer):
                observer.shutdown()


def build_telemetry(
    settings: Settings, environ: Mapping[str, str] | None = None
) -> Telemetry:
    """Create the observers configured by `settings` and the environment.

    Langfuse tracing is enabled when `LANGFUSE_PUBLIC_KEY` and
    `LANGFUSE_SECRET_KEY` are both set; `LANGFUSE_BASE_URL` selects the server.

    Raises:
        ValueError: If only one Langfuse key is configured.
    """
    env = os.environ if environ is None else environ
    registry = CollectorRegistry()
    prometheus = PrometheusObserver(registry=registry, namespace=METRICS_NAMESPACE)
    audits = Counter(
        "audits",
        "Completed compliance audits.",
        ("verdict",),
        namespace=METRICS_NAMESPACE,
        registry=registry,
    )
    public_key = env.get("LANGFUSE_PUBLIC_KEY") or None
    secret_key = env.get("LANGFUSE_SECRET_KEY") or None
    if (public_key is None) != (secret_key is None):
        raise ValueError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together"
        )
    langfuse = (
        LangfuseObserver(
            public_key=public_key,
            secret_key=secret_key,
            base_url=env.get("LANGFUSE_BASE_URL", "http://127.0.0.1:3000"),
            event_metadata={"application_id": APPLICATION_ID, "tenant_id": TENANT_ID},
        )
        if public_key and secret_key
        else None
    )
    observers: tuple[EventObserver, ...] = (
        LoggingObserver(),
        prometheus,
        *((langfuse,) if langfuse is not None else ()),
    )
    return Telemetry(
        observers=observers,
        registry=registry,
        prometheus=prometheus,
        audits=audits,
        langfuse=langfuse,
        metrics_enabled=settings.metrics_enabled,
        metrics_host=settings.metrics_host,
        metrics_port=settings.metrics_port,
    )
