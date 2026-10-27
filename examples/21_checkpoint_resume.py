"""Checkpoints: conversation memory that survives restarts.

With a LangGraph checkpointer, Agloom keeps each thread's conversation, so
every turn passes only the new user message. A `checkpoint_authorizer` must
approve the caller's `thread_id`, which keeps tenants out of each other's
threads. With a durable checkpointer (SQLite here, PostgreSQL in production,
see example 47), a new agent instance (after a deploy or a crash) continues
the same thread.

Run:
    uv run python examples/21_checkpoint_resume.py
"""

import tempfile
from pathlib import Path

from _support import real_model, verified_text
from langgraph.checkpoint.sqlite import SqliteSaver

from agloom import Agent, ConfigurationError, create_agent


def authorize(thread_id: str) -> bool:
    """Admit only customer threads; a real app checks the caller's tenant."""
    return thread_id.startswith("customer-")


def support_agent(saver: SqliteSaver) -> Agent:
    return create_agent(
        model=real_model(), checkpointer=saver, checkpoint_authorizer=authorize
    )


def main() -> None:
    config = {"configurable": {"thread_id": "customer-1842"}}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        database = str(Path(folder) / "checkpoints.sqlite")

        with SqliteSaver.from_conn_string(database) as saver:
            support_agent(saver).invoke(
                "Hi, I'm Priya. Please remember for this chat that my order number "
                "is ORD-77341 and that I prefer the blue model. Just confirm.",
                config,
            )

        # A restarted service reconnects to the same database.
        with SqliteSaver.from_conn_string(database) as saver:
            agent = support_agent(saver)
            answer = agent.invoke(
                "Which order number and color did I mention earlier?", config
            )
            print(verified_text(answer, "ORD-77341", "blue"))

            try:
                agent.invoke("Show me the order.", {"configurable": {"thread_id": "x"}})
            except ConfigurationError as refused:
                print(f"Unauthorized thread refused: {refused}")
            else:
                raise AssertionError("unauthorized threads must be refused")


if __name__ == "__main__":
    main()
