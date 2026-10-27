"""Alembic environment.

The database URL comes from `-x db_url=...`, the `database_url` attribute set
by `backend.migrate`, the `sqlalchemy.url` option, or the `DATABASE_URL`
environment variable, in that order; credentials never live in alembic.ini.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = None


def _database_url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    url = (
        x_args.get("db_url")
        or config.attributes.get("database_url")
        or config.get_main_option("sqlalchemy.url")
        or os.environ.get("DATABASE_URL")
    )
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Export it, or pass -x db_url=... to alembic."
        )
    if url.startswith("postgresql://"):
        # Use the psycopg3 dialect already required by the application
        # instead of SQLAlchemy's psycopg2 default.
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        version_table_schema=config.attributes.get("version_table_schema"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        postgres = connection.dialect.name == "postgresql"
        if postgres:
            connection.execute(
                text(
                    "SELECT pg_advisory_lock("
                    "hashtext('agloom-autonomous-product-studio-migrations'))"
                )
            )
            connection.commit()
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                version_table_schema=config.attributes.get("version_table_schema"),
            )
            with context.begin_transaction():
                context.run_migrations()
        finally:
            if postgres:
                connection.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtext('agloom-autonomous-product-studio-migrations'))"
                    )
                )
                connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
