"""Create or upgrade every schema the application uses.

Run `python -m backend.setup_database`:

1. Alembic migrations for the application tables.
2. The procrastinate job-queue schema (applied once; later procrastinate
   releases ship SQL migrations that are applied as described in MIGRATIONS.md).
3. The LangGraph PostgreSQL checkpointer tables (versioned, idempotent).
4. The agentbarrier runtime schema (versioned, idempotent).
"""

from __future__ import annotations

import asyncio
import logging
import os

import psycopg
from agentbarrier.runtime import PostgresRuntimeStore
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .config import APP_ROOT, Settings
from .jobs import create_job_app
from .runtime import job_connector, run_async

_LOGGER = logging.getLogger(__name__)


def alembic_config(dsn: str) -> Config:
    config = Config(str(APP_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(APP_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
    return config


async def procrastinate_schema_exists(dsn: str) -> bool:
    async with await psycopg.AsyncConnection.connect(dsn) as db:
        row = await (
            await db.execute(
                """SELECT 1 FROM information_schema.tables
                   WHERE table_schema = current_schema()
                   AND table_name = 'procrastinate_jobs'"""
            )
        ).fetchone()
    return row is not None


async def setup_database(settings: Settings) -> None:
    dsn = settings.require_database()
    await asyncio.to_thread(command.upgrade, alembic_config(dsn), "head")
    if not await procrastinate_schema_exists(dsn):
        jobs = create_job_app(job_connector(dsn, max_size=2), settings)
        async with jobs.open_async():
            await jobs.schema_manager.apply_schema_async()
        _LOGGER.info("applied the procrastinate schema")
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
    store = await asyncio.to_thread(
        PostgresRuntimeStore,
        dsn,
        schema=settings.barrier_schema,
        create_schema=True,
        migrate=True,
    )
    store.close()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_async(setup_database(Settings.from_env()))
