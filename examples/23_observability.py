"""Observability: events to logs, metrics, plain callables, and Langfuse.

Every agent emits normalized events (`AgentEvent`, see `EventName`) to its
observers. `LoggingObserver` writes structured logs with structlog,
`PrometheusObserver` exports execution, policy, and loop metrics, and any
plain callable receives events too. `LangfuseObserver` also traces model and
tool calls; it is attached here when `LANGFUSE_BASE_URL`,
`LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` point at a reachable Langfuse
(see examples/observability for a local stack).

Run:
    uv run python examples/23_observability.py
"""

import os
import urllib.request
from collections.abc import Callable

from _support import real_model, verified_text
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


def langfuse() -> LangfuseObserver | None:
    """Return a Langfuse observer when Langfuse is configured and reachable."""
    names = ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    if not all(os.environ.get(name) for name in names):
        return None
    observer = LangfuseObserver()
    try:
        reachable = observer.client.auth_check()
    except Exception as error:
        reachable = False
        print(f"Langfuse is configured but unreachable ({type(error).__name__}).")
    if not reachable:
        observer.shutdown()
        return None
    return observer


STOCK = {"SKU-778": 42, "SKU-901": 7}


def stock_level(sku: str) -> int:
    """Return the units in stock for a product."""
    return STOCK.get(sku, 0)


def main() -> None:
    prometheus = PrometheusObserver(registry=CollectorRegistry())
    prometheus.start_http_server(0)
    events: list[AgentEvent] = []
    observers: list[Observer | Callable[[AgentEvent], object]] = [
        LoggingObserver(),
        prometheus,
        events.append,
    ]
    if (tracing := langfuse()) is not None:
        observers.append(tracing)

    agent = create_agent(
        model=real_model(), tools=[stock_level], pattern="react", observers=observers
    )
    answer = agent.invoke("How many units of SKU-778 are in stock?")
    print(verified_text(answer, "42"))

    url = f"http://127.0.0.1:{prometheus.server_port}/metrics"
    with urllib.request.urlopen(url, timeout=10) as response:
        metrics = response.read().decode()
    agent.close()

    names = [event.name for event in events]
    assert names[0] == EventName.AGENT_CREATED, names
    assert names[-1] == EventName.EXECUTION_COMPLETED, names
    assert 'agloom_executions_total{status="completed"' in metrics
    print("Events:", ", ".join(names))
    print(
        "Metrics:",
        [line for line in metrics.splitlines() if "executions_total{" in line],
    )
    print(f"Langfuse: {'traced' if tracing else 'off'}")


if __name__ == "__main__":
    main()
