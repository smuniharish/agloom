"""Environment-driven telemetry for the RCA generator.

Telemetry is off unless `AGLOOM_OBSERVABILITY_ENABLED` is true. When it is on,
every Agloom event is logged with structlog and exported as Prometheus metrics
on `AGLOOM_METRICS_HOST:AGLOOM_METRICS_PORT`. When both `LANGFUSE_PUBLIC_KEY`
and `LANGFUSE_SECRET_KEY` are set, runs are also traced in Langfuse; the
Langfuse SDK reads its keys and `LANGFUSE_BASE_URL` from the environment, so
no credential passes through this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from prometheus_client import CollectorRegistry

from agloom import LangfuseObserver, LoggingObserver, Observer, PrometheusObserver

_ENABLED = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class Telemetry:
    """The observers passed to the agent; empty when telemetry is disabled."""

    observers: tuple[Observer, ...] = ()

    def shutdown(self) -> None:
        """Flush and release every observer, in reverse order of creation."""
        for observer in reversed(self.observers):
            observer.shutdown()


def observability_enabled() -> bool:
    return os.getenv("AGLOOM_OBSERVABILITY_ENABLED", "").strip().casefold() in _ENABLED


def langfuse_configured() -> bool:
    public_key = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    if public_key != bool(os.getenv("LANGFUSE_SECRET_KEY")):
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together"
        )
    return public_key


def metrics_port(default: int) -> int:
    return int(os.getenv("AGLOOM_METRICS_PORT", str(default)))


def start_telemetry(
    namespace: str, *, default_metrics_port: int, metadata: Mapping[str, str]
) -> Telemetry:
    """Create the configured observers and start the Prometheus endpoint."""
    if not observability_enabled():
        return Telemetry()
    langfuse = langfuse_configured()
    prometheus = PrometheusObserver(
        registry=CollectorRegistry(), namespace=f"agloom_{namespace}"
    )
    prometheus.start_http_server(
        metrics_port(default_metrics_port),
        address=os.getenv("AGLOOM_METRICS_HOST", "127.0.0.1"),
    )
    observers: list[Observer] = [LoggingObserver(), prometheus]
    if langfuse:
        observers.append(LangfuseObserver(event_metadata=metadata))
    return Telemetry(tuple(observers))


def observability_status(default_metrics_port: int) -> dict[str, object]:
    """Describe the telemetry configuration without exposing credentials."""
    enabled = observability_enabled()
    return {
        "enabled": enabled,
        "prometheus": enabled,
        "metrics_port": metrics_port(default_metrics_port),
        "langfuse": enabled
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY")),
    }
