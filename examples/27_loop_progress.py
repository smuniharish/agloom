"""Progress: measure objective progress from the system, not from the model.

A `LoopObserver` turns each iteration's outcome into signals. Here it reads
the real progress of a data backfill (batches done / total) and reports it as
`objective_progress`. The deterministic `DefaultLoopEvaluator` classifies
each iteration (progress, no progress, regression, stagnation) without extra
model calls and completes the loop at 100%. The backfill service accepts two
batches per maintenance window; a window opens for each attempt.

Run:
    uv run python examples/27_loop_progress.py
"""

from typing import Any

from _support import real_model

from agloom import (
    AgentEvent,
    DefaultLoopEvaluator,
    DefaultLoopObserver,
    EventName,
    LoopObservation,
    LoopPolicy,
    LoopTerminationReason,
    ProgressStatus,
    create_agent,
)

BATCHES = [f"2026-09-{day:02d}" for day in range(1, 7)]
done: list[str] = []
window = {"remaining": 0}


def open_window(event: AgentEvent) -> None:
    """A new maintenance window (two batches) opens before each attempt."""
    if event.name == EventName.LOOP_ITERATION_STARTED:
        window["remaining"] = 2


def backfill_next_batch() -> str:
    """Backfill the next pending day of order data."""
    pending = [batch for batch in BATCHES if batch not in done]
    if not pending:
        return "All batches are already backfilled."
    if window["remaining"] == 0:
        return "The maintenance window is used up; continue in the next attempt."
    window["remaining"] -= 1
    done.append(pending[0])
    return f"Backfilled {pending[0]} ({len(done)}/{len(BATCHES)} batches done)."


class BackfillProgress(DefaultLoopObserver):
    """Reports the share of batches done as the objective progress."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        return observation.model_copy(
            update={"objective_progress": len(done) / len(BATCHES)}
        )


def main() -> None:
    agent = create_agent(
        model=real_model(),
        tools=[backfill_next_batch],
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=8),
        loop_observer=BackfillProgress(),
        loop_evaluator=DefaultLoopEvaluator(),
        observers=[open_window],
    )

    result = agent.invoke_loop(
        "Backfill all pending days of order data. Call the backfill tool until "
        "it says the window is used up, then report how many batches are done."
    )

    state = result.state
    for entry in state.history:
        print(
            f"iteration {entry.observation.iteration}: "
            f"{entry.evaluation.objective_progress:.0%} "
            f"({entry.evaluation.progress_status})"
        )
    statuses = {entry.evaluation.progress_status for entry in state.history}
    assert done == BATCHES, done
    assert state.progress == 1.0
    assert state.iteration == 3, state.iteration
    assert statuses == {ProgressStatus.PROGRESS}, statuses
    assert state.termination_reason is LoopTerminationReason.OBJECTIVE_SATISFIED


if __name__ == "__main__":
    main()
