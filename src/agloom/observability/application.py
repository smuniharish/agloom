"""Environment-driven observability for deployable Agloom applications."""

from __future__ import annotations

import os
import re

from agloom.observability.base import Observer
from agloom.observability.langfuse import LangfuseObserver
from agloom.observability.logging import LoggingObserver
from agloom.observability.prometheus import PrometheusObserver

_TRUE_VALUES = {"1", "true", "yes", "on"}


def application_observers(
    application_id: str,
    *,
    default_metrics_port: int,
) -> tuple[Observer, ...]:
    """Build configured observers without exposing credentials to the caller."""

    if not observability_enabled():
        return ()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if bool(public_key) != bool(secret_key):
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together"
        )
    namespace = re.sub(r"[^a-zA-Z0-9_:]", "_", application_id)
    prometheus = PrometheusObserver(namespace=f"agloom_{namespace}")
    prometheus.start_http_server(
        port=int(os.getenv("AGLOOM_METRICS_PORT", str(default_metrics_port))),
        address=os.getenv("AGLOOM_METRICS_HOST", "127.0.0.1"),
    )
    observers: list[Observer] = [LoggingObserver(), prometheus]
    if public_key and secret_key:
        base_url = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
        observers.append(
            LangfuseObserver(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url,
                event_metadata={"application_id": application_id},
            )
        )
    return tuple(observers)


def observability_enabled() -> bool:
    return os.getenv("AGLOOM_OBSERVABILITY_ENABLED", "").casefold() in _TRUE_VALUES


def observability_status(default_metrics_port: int) -> dict[str, object]:
    """Return non-sensitive runtime telemetry configuration."""

    enabled = observability_enabled()
    return {
        "enabled": enabled,
        "prometheus": enabled,
        "metrics_port": int(
            os.getenv("AGLOOM_METRICS_PORT", str(default_metrics_port))
        ),
        "langfuse": enabled
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY")),
        "xai": True,
    }
