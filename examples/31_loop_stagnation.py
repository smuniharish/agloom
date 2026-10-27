"""Stagnation: stop early when repeated attempts make no progress.

Loops must not burn their whole budget on an objective that cannot move.
Here the carrier has not assigned a tracking number yet, so every attempt
reports the same state. An observer measures objective progress (is there a
tracking number?), and after `stagnation_threshold` iterations without
progress the controller stops with `STAGNATION`, long before `max_iterations`.
Allow replan, refresh, or human intervention to try those instead.

Run:
    uv run python examples/31_loop_stagnation.py
"""

import re
from typing import Any

from _support import real_model

from agloom import (
    DefaultLoopObserver,
    LoopObservation,
    LoopPolicy,
    LoopTerminationReason,
    ProgressStatus,
    create_agent,
)

TRACKING_NUMBER = re.compile(r"\b1Z[0-9A-Z]{10,}\b")
checks: list[str] = []


def shipment_status(order_id: str) -> str:
    """Return the carrier status of an order's shipment."""
    checks.append(order_id)
    return f"Order {order_id}: label created, no tracking number assigned yet."


class TrackingProgress(DefaultLoopObserver):
    """Progress is 1.0 once the answer contains a carrier tracking number."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        found = TRACKING_NUMBER.search(kwargs["result"].text) is not None
        return observation.model_copy(
            update={"objective_progress": 1.0 if found else 0.0}
        )


def main() -> None:
    policy = LoopPolicy(max_iterations=6, stagnation_threshold=2)
    agent = create_agent(
        model=real_model(),
        tools=[shipment_status],
        pattern="react",
        loop_policy=policy,
        loop_observer=TrackingProgress(),
    )

    result = agent.invoke_loop("Get the carrier tracking number for order A-2040.")

    state = result.state
    assert state.termination_reason is LoopTerminationReason.STAGNATION
    assert state.iteration == policy.stagnation_threshold < policy.max_iterations
    assert state.progress_status is ProgressStatus.STAGNATION
    print(
        f"Stopped after {state.iteration} of {policy.max_iterations} iterations: "
        f"{state.decision.reason} ({len(checks)} carrier checks)."
    )


if __name__ == "__main__":
    main()
