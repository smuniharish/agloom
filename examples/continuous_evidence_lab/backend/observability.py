"""Environment-driven telemetry shared by every agent of one process."""

from __future__ import annotations

import logging
import os

from prometheus_client import Counter, Gauge

from agloom import LangfuseObserver, LoggingObserver, Observer, PrometheusObserver

from .config import env_flag

APPLICATION_ID = "continuous-evidence-lab"
METRICS_NAMESPACE = "agloom_continuous_evidence_lab"
DEFAULT_METRICS_PORT = 9470

RUNS = Counter("continuous_evidence_runs_total", "Evidence runs by outcome", ["status"])
QUALITY = Gauge("continuous_evidence_quality", "Latest validated evidence quality")
DECISIONS = Counter(
    "continuous_evidence_decisions_total", "Reviewer decisions", ["decision"]
)

_LOGGER = logging.getLogger(__name__)


def observability_enabled() -> bool:
    return env_flag("OBSERVABILITY_ENABLED")


def metrics_port() -> int:
    return int(os.getenv("METRICS_PORT", str(DEFAULT_METRICS_PORT)))


def _langfuse_keys() -> tuple[str | None, str | None]:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or None
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") or None
    if bool(public_key) != bool(secret_key):
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together"
        )
    return public_key, secret_key


class Telemetry:
    """Logging, Prometheus, and optional Langfuse observers of one process."""

    def __init__(self, observers: tuple[Observer, ...] = ()) -> None:
        self.observers = observers

    def flush(self) -> None:
        for observer in self.observers:
            try:
                observer.flush()
            except Exception:
                _LOGGER.exception("telemetry flush failed")

    def shutdown(self) -> None:
        for observer in reversed(self.observers):
            try:
                observer.shutdown()
            except Exception:
                _LOGGER.exception("telemetry shutdown failed")


_PROCESS_TELEMETRY: Telemetry | None = None


def process_telemetry(component: str) -> Telemetry:
    """Return this process's telemetry, starting the metrics server once."""
    global _PROCESS_TELEMETRY
    if _PROCESS_TELEMETRY is not None:
        return _PROCESS_TELEMETRY
    if not observability_enabled():
        _PROCESS_TELEMETRY = Telemetry()
        return _PROCESS_TELEMETRY
    public_key, secret_key = _langfuse_keys()
    prometheus = PrometheusObserver(namespace=METRICS_NAMESPACE)
    prometheus.start_http_server(
        metrics_port(), address=os.getenv("METRICS_HOST", "127.0.0.1")
    )
    observers: list[Observer] = [LoggingObserver(), prometheus]
    if public_key and secret_key:
        observers.append(
            LangfuseObserver(
                public_key=public_key,
                secret_key=secret_key,
                base_url=os.getenv("LANGFUSE_BASE_URL", "http://127.0.0.1:3000"),
                event_metadata={
                    "application_id": APPLICATION_ID,
                    "component": component,
                },
            )
        )
    _PROCESS_TELEMETRY = Telemetry(tuple(observers))
    return _PROCESS_TELEMETRY


def observability_status() -> dict[str, object]:
    """Return the non-sensitive telemetry configuration."""
    enabled = observability_enabled()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    return {
        "enabled": enabled,
        "prometheus": enabled,
        "metrics_port": metrics_port(),
        "langfuse": enabled and bool(public_key) and bool(secret_key),
    }
