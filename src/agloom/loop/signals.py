"""Bridges that feed ecosystem decisions into Loop Engineering."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_COLLECTOR: ContextVar[list[str] | None] = ContextVar(
    "agloom_behavior_signals", default=None
)


def behavior_signals(decision: Any, event: Any) -> None:
    """Record BehaviorWeave decisions as Loop Engineering signals.

    Pass this function as `on_decision` of
    `behaviorweave.integrations.langchain.BehaviorWeaveMiddleware`. Every
    actionable decision made during an iteration becomes a behavior signal of
    that iteration's observation, which the default controller maps to loop
    actions within the `LoopPolicy`: `stop` stops the loop, `pause` waits,
    `human_review` and `escalate` request human intervention, `redirect`
    replans, and `retry` retries.

    Args:
        decision: The BehaviorWeave `InterventionDecision`.
        event: The BehaviorWeave event that produced the decision.
    """
    del event
    collector = _COLLECTOR.get()
    if collector is None or not getattr(decision, "actionable", False):
        return
    kind = getattr(getattr(decision, "intervention", None), "kind", None)
    if kind is not None:
        collector.append(str(getattr(kind, "value", kind)))


@contextmanager
def collect_behavior_signals() -> Generator[list[str]]:
    """Collect the behavior signals recorded while the context is active."""
    signals: list[str] = []
    token = _COLLECTOR.set(signals)
    try:
        yield signals
    finally:
        _COLLECTOR.reset(token)
