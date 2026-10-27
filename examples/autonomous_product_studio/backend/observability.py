"""Environment-driven observability for the Product Studio API and worker.

Agloom ships the observers; this module decides which ones the application
runs. Prometheus metrics use the `agloom_autonomous_product_studio` prefix and
include the procrastinate job queue, so the shared Grafana dashboard can chart
runs, policy outcomes, loop decisions, and queue health.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from typing import Any

import psycopg
from agloom import LangfuseObserver, LoggingObserver, Observer, PrometheusObserver
from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

APPLICATION_ID = "autonomous-product-studio"
METRICS_NAMESPACE = "agloom_autonomous_product_studio"
DEFAULT_METRICS_PORT = 9469
_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOGGER = logging.getLogger(__name__)


def observability_enabled() -> bool:
    return os.getenv("AGLOOM_OBSERVABILITY_ENABLED", "").casefold() in _TRUE_VALUES


def metrics_port() -> int:
    return int(os.getenv("AGLOOM_METRICS_PORT", str(DEFAULT_METRICS_PORT)))


def langfuse_configured() -> bool:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if bool(public_key) != bool(secret_key):
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together"
        )
    return (
        bool(public_key)
        and os.getenv("AGLOOM_LANGFUSE_ENABLED", "true").casefold() in _TRUE_VALUES
    )


def build_observers(*, database_url: str | None = None) -> tuple[Observer, ...]:
    """Create the configured observers and start the metrics endpoint.

    Returns no observers unless `AGLOOM_OBSERVABILITY_ENABLED` is true. Langfuse
    is added only when both Langfuse keys are configured.
    """
    if not observability_enabled():
        return ()
    prometheus = PrometheusObserver(namespace=METRICS_NAMESPACE)
    if database_url:
        register_queue_metrics(database_url)
    prometheus.start_http_server(
        metrics_port(), address=os.getenv("AGLOOM_METRICS_HOST", "127.0.0.1")
    )
    observers: list[Observer] = [LoggingObserver(), prometheus]
    if langfuse_configured():
        observers.append(
            LangfuseObserver(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                base_url=os.getenv("LANGFUSE_BASE_URL", "http://127.0.0.1:3000"),
                event_metadata={"application_id": APPLICATION_ID},
            )
        )
    return tuple(observers)


def shutdown_observers(observers: Iterable[Observer]) -> None:
    """Flush and stop observers; one failing observer never blocks the others."""
    for observer in observers:
        try:
            observer.flush()
            observer.shutdown()
        except Exception:
            _LOGGER.exception(
                "observer %s failed to shut down", type(observer).__name__
            )


def observability_status() -> dict[str, object]:
    """Non-sensitive telemetry configuration for the health endpoint."""
    enabled = observability_enabled()
    return {
        "enabled": enabled,
        "prometheus": enabled,
        "metrics_port": metrics_port(),
        "langfuse": enabled and langfuse_configured(),
    }


class JobQueueCollector(Collector):
    """Reports procrastinate queue depth and age at scrape time."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def collect(self) -> Iterator[Any]:
        jobs = GaugeMetricFamily(
            f"{METRICS_NAMESPACE}_jobs",
            "Procrastinate jobs by status.",
            labels=["status"],
        )
        oldest = GaugeMetricFamily(
            f"{METRICS_NAMESPACE}_oldest_queued_job_seconds",
            "Age of the oldest job waiting in the queue.",
        )
        try:
            with psycopg.connect(self._database_url, connect_timeout=2) as db:
                for status, count in db.execute(
                    "SELECT status::text, count(*) FROM procrastinate_jobs GROUP BY status"
                ).fetchall():
                    jobs.add_metric([status], float(count))
                age = db.execute(
                    """SELECT COALESCE(EXTRACT(EPOCH FROM
                              now() - min(COALESCE(j.scheduled_at, e.at))), 0)
                       FROM procrastinate_jobs j
                       JOIN procrastinate_events e
                         ON e.job_id = j.id AND e.type = 'deferred'
                       WHERE j.status = 'todo'
                         AND (j.scheduled_at IS NULL OR j.scheduled_at <= now())"""
                ).fetchone()
        except psycopg.Error:
            _LOGGER.warning("job queue metrics are unavailable")
            return
        oldest.add_metric([], float(age[0]) if age else 0.0)
        yield jobs
        yield oldest


_QUEUE_COLLECTOR: JobQueueCollector | None = None


def register_queue_metrics(database_url: str) -> None:
    global _QUEUE_COLLECTOR
    if _QUEUE_COLLECTOR is None:
        _QUEUE_COLLECTOR = JobQueueCollector(database_url)
        REGISTRY.register(_QUEUE_COLLECTOR)
