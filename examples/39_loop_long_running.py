"""Long-running loops: stream progress while a multi-iteration job runs.

Long objectives run as many bounded iterations. Watch them live with
`astream(stream_mode="custom")`, which yields Agloom's loop events as they
happen. A checkpointer makes every iteration durable, `max_duration` caps
active time, `history_limit` keeps the loop state small, and `agent.stop()`
cancels cooperatively. The migration service accepts two tables per
maintenance window; a window opens for each attempt.

Run:
    uv run python examples/39_loop_long_running.py
"""

import asyncio
from typing import Any

from _support import real_model
from langgraph.checkpoint.memory import InMemorySaver

from agloom import (
    AgentEvent,
    DefaultLoopObserver,
    EventName,
    LoopObservation,
    LoopPolicy,
    create_agent,
)

TABLES = ("customers", "orders", "invoices", "payments", "shipments")
migrated: list[str] = []
window = {"remaining": 0}


def maintenance_window(event: AgentEvent) -> None:
    """A maintenance window for two tables opens before each attempt."""
    if event.name == EventName.LOOP_ITERATION_STARTED:
        window["remaining"] = 2


def migrate_next_table() -> str:
    """Migrate the next pending table to the new database."""
    pending = [table for table in TABLES if table not in migrated]
    if not pending:
        return "Every table is already migrated."
    if window["remaining"] == 0:
        return "The maintenance window is used up; continue in the next attempt."
    window["remaining"] -= 1
    migrated.append(pending[0])
    return f"Migrated {pending[0]} ({len(migrated)}/{len(TABLES)})."


class MigrationProgress(DefaultLoopObserver):
    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        return observation.model_copy(
            update={"objective_progress": len(migrated) / len(TABLES)}
        )


async def main() -> None:
    agent = create_agent(
        model=real_model(),
        tools=[migrate_next_table],
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=10, max_duration=900, history_limit=5),
        loop_observer=MigrationProgress(),
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=lambda thread_id: thread_id.startswith("migration-"),
        observers=[maintenance_window],
    )
    config = {"configurable": {"thread_id": "migration-2026-10"}}

    summary: dict[str, Any] = {}
    async for event in agent.astream(
        "Migrate every table. Call the migration tool until the window is used "
        "up, then report progress.",
        config,
        stream_mode="custom",
    ):
        if event["agloom_event"] == EventName.LOOP_ITERATION_COMPLETED:
            print(
                f"iteration {event['iteration']}: {len(migrated)}/{len(TABLES)} "
                f"tables, {event['progress_status']} -> {event['action']}"
            )
        elif event["agloom_event"] == EventName.LOOP_COMPLETED:
            summary = event

    assert migrated == list(TABLES), migrated
    assert summary["termination_reason"] == "objective_satisfied", summary
    assert summary["iterations"] == 3, summary
    print(f"Migration finished in {summary['iterations']} iterations.")


if __name__ == "__main__":
    asyncio.run(main())
