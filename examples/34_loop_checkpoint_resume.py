"""Checkpoint resume: a stopped loop continues in a new process where it left off.

Every loop iteration is a checkpointed task. When a run stops mid-loop (a
deploy, a crash, or `agent.stop()` as here), a new agent instance with the
same durable checkpointer continues the thread with `agent.resume(config)`:
completed iterations and their tool calls are replayed from the checkpoint,
not executed again.

Run:
    uv run python examples/34_loop_checkpoint_resume.py
"""

import tempfile
from pathlib import Path
from typing import Any

from _support import real_model, verified_text
from langgraph.checkpoint.sqlite import SqliteSaver

from agloom import (
    Agent,
    AgentEvent,
    CancellationError,
    DefaultLoopObserver,
    EventName,
    LoopObservation,
    LoopPolicy,
    Observer,
    create_agent,
)

REGIONS = ("EU", "US", "APAC")
closed: list[str] = []
session = {"open": False}


def close_next_region() -> str:
    """Close the month-end books of the next open region (one per session)."""
    if not session["open"]:
        return "The ledger session is used up; continue in the next attempt."
    session["open"] = False
    region = REGIONS[len(closed)]
    closed.append(region)
    return f"Books closed for {region} ({len(closed)}/{len(REGIONS)})."


def ledger_session(event: AgentEvent) -> None:
    """The ledger grants one close per session; a session opens per attempt."""
    if event.name == EventName.LOOP_ITERATION_STARTED:
        session["open"] = True


class CloseProgress(DefaultLoopObserver):
    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        return observation.model_copy(
            update={"objective_progress": len(closed) / len(REGIONS)}
        )


class Operator(Observer):
    """Stops the agent when a given iteration starts, like a deploy would."""

    def __init__(self, iteration: int) -> None:
        self.iteration = iteration
        self.agent: Agent | None = None

    def on_event(self, event: AgentEvent) -> None:
        started = event.name == EventName.LOOP_ITERATION_STARTED
        if started and event.payload["iteration"] == self.iteration and self.agent:
            self.agent.stop()


def month_end_agent(saver: SqliteSaver, *observers: Any) -> Agent:
    return create_agent(
        model=real_model(),
        tools=[close_next_region],
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=5),
        loop_observer=CloseProgress(),
        checkpointer=saver,
        checkpoint_authorizer=lambda thread_id: thread_id.startswith("close-"),
        observers=[ledger_session, *observers],
    )


def main() -> None:
    config = {"configurable": {"thread_id": "close-2026-09"}}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        database = str(Path(folder) / "loops.sqlite")

        with SqliteSaver.from_conn_string(database) as saver:
            operator = Operator(iteration=2)
            agent = month_end_agent(saver, operator)
            operator.agent = agent
            try:
                agent.invoke("Close the month-end books of every region.", config)
            except CancellationError:
                print(f"Stopped during iteration 2; closed so far: {closed}")
            else:
                raise AssertionError("the operator should stop the run")

        started: list[int] = []

        def record_start(event: AgentEvent) -> None:
            if event.name == EventName.LOOP_ITERATION_STARTED:
                started.append(event.payload["iteration"])

        with SqliteSaver.from_conn_string(database) as saver:
            restarted = month_end_agent(saver, record_start)
            answer = restarted.resume(config)

    assert closed == list(REGIONS), closed
    assert started == [2, 3], started
    print(f"Resumed at iteration {started[0]}; every region closed exactly once.")
    print(verified_text(answer, "APAC"))


if __name__ == "__main__":
    main()
