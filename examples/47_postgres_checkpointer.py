"""PostgreSQL checkpointer: durable threads that any worker can resume.

`langgraph-checkpoint-postgres` (install `agloom[postgres]`, or
`agloom[postgres-binary]` on hosts without a PostgreSQL client library) stores
every checkpoint in PostgreSQL, so paused runs (approvals, breakpoints, human
intervention) and conversation memory survive restarts and move between
workers. Call `setup()` once to create the tables. Here one process pauses
before a tool call and another resumes the thread and keeps the conversation.

Run (with a PostgreSQL URL in `AGLOOM_POSTGRES_URL`):
    uv run python examples/47_postgres_checkpointer.py
"""

import os
import uuid

from _support import real_model, verified_text
from langgraph.checkpoint.postgres import PostgresSaver

from agloom import Agent, ExecutionPaused, create_agent

escalations: list[str] = []


def escalate_ticket(ticket_id: str, team: str) -> str:
    """Escalate a support ticket to an on-call team."""
    escalations.append(ticket_id)
    return f"Ticket {ticket_id} escalated to the {team} on-call engineer."


def support_agent(saver: PostgresSaver) -> Agent:
    return create_agent(
        model=real_model(),
        tools=[escalate_ticket],
        pattern="react",
        interrupt_before=["tools"],
        checkpointer=saver,
        checkpoint_authorizer=lambda thread_id: thread_id.startswith("ticket-"),
    )


def main() -> None:
    url = os.environ.get("AGLOOM_POSTGRES_URL")
    if not url:
        raise SystemExit(
            "Set AGLOOM_POSTGRES_URL to a PostgreSQL URL, for example "
            "postgresql://agloom@localhost:5432/agloom"
        )
    config = {"configurable": {"thread_id": f"ticket-{uuid.uuid4().hex[:8]}"}}

    with PostgresSaver.from_conn_string(url) as saver:
        saver.setup()
        try:
            support_agent(saver).invoke(
                "Escalate ticket T-88 to the payments team.", config
            )
        except ExecutionPaused as paused:
            print(f"Paused on thread {paused.thread_id}; this worker exits.")
        else:
            raise AssertionError("the run should pause before the tool call")
    assert not escalations

    with PostgresSaver.from_conn_string(url) as saver:
        worker = support_agent(saver)
        answer = worker.resume(config)
        assert escalations == ["T-88"], escalations
        print("Resumed by another worker:", verified_text(answer, "T-88"))

        follow_up = worker.invoke("Which ticket did you escalate?", config)
        print("Memory:", verified_text(follow_up, "T-88"))


if __name__ == "__main__":
    main()
