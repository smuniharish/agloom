"""Replan: change the approach when a validator rejects the result.

A `LoopObserver` can validate each result against rules the model never saw.
Here a notification gateway accepts at most 120 characters. A rejected message
sets `replan_required` with feedback, so the next iteration takes a different
approach (with the feedback as context) within `max_replans`; an accepted
message sets `validation_passed`, which completes the loop.

Run:
    uv run python examples/29_loop_replan.py
"""

from typing import Any

from _support import real_model

from agloom import (
    DefaultLoopObserver,
    LoopAction,
    LoopObservation,
    LoopPolicy,
    LoopTerminationReason,
    create_agent,
)

LIMIT = 120


class NotificationGateway(DefaultLoopObserver):
    """Validates the push notification the way the delivery gateway will."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        length = len(kwargs["result"].text.strip())
        if length <= LIMIT:
            return observation.model_copy(update={"validation_passed": True})
        return observation.model_copy(
            update={
                "replan_required": True,
                "feedback": (
                    f"The gateway rejected the notification: {length} characters, "
                    f"the limit is {LIMIT}. Rewrite it as a single short line.",
                ),
            }
        )


def main() -> None:
    agent = create_agent(
        model=real_model(),
        loop_policy=LoopPolicy(max_iterations=4, allow_replan=True, max_replans=2),
        loop_observer=NotificationGateway(),
    )

    result = agent.invoke_loop(
        "Write the push notification for order A-1001: apologize for the delay, "
        "give the new delivery date (Thursday, October 9), include the discount "
        "code SORRY10, and the tracking link https://acme.example/t/A-1001. "
        "Reply with the notification text only."
    )

    state = result.state
    message = result.result.text.strip()
    assert state.history[0].decision.action is LoopAction.REPLAN
    assert state.termination_reason is LoopTerminationReason.VALIDATED
    assert len(message) <= LIMIT, message
    print(f"Rejected: {state.history[0].evaluation.feedback[0]}")
    print(f"Accepted after {state.budget.replans} replan(s): {message}")


if __name__ == "__main__":
    main()
