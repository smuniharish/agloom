"""Create or upgrade every schema the studio uses: `python -m backend.migrate`.

1. Alembic: the application tables and procrastinate's job queue.
2. langgraph-checkpoint-postgres: the checkpoint tables (idempotent and versioned).
3. agentbarrier: the policy runtime schema (idempotent and versioned).
"""

from __future__ import annotations

import os

from agentbarrier.runtime import PostgresRuntimeStore
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres import PostgresSaver

from .settings import APP_ROOT


def alembic_config(database_url: str) -> Config:
    config = Config(str(APP_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(APP_ROOT / "migrations"))
    config.attributes["database_url"] = database_url
    return config


def migrate(database_url: str, *, policy_schema: str = "agentbarrier") -> None:
    command.upgrade(alembic_config(database_url), "head")
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
    PostgresRuntimeStore(
        database_url, schema=policy_schema, create_schema=True, migrate=True
    ).close()


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    migrate(
        database_url, policy_schema=os.getenv("STUDIO_POLICY_SCHEMA", "agentbarrier")
    )


if __name__ == "__main__":
    main()
