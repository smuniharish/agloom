from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from agentbarrier.runtime import SQLiteRuntimeStore
from backend.auth import principal_scope
from backend.config import Settings
from backend.jobs import create_job_app
from procrastinate import App
from procrastinate.testing import InMemoryConnector
from psycopg import sql

from tests.support import TEST_PRINCIPAL

PGVECTOR_IMAGE = os.getenv("TEST_POSTGRES_IMAGE", "pgvector/pgvector:pg18")


def pytest_asyncio_loop_factories(config: Any, item: Any) -> dict[str, Any]:
    # psycopg's async driver needs a selector event loop on Windows.
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return {"default": asyncio.new_event_loop}


@pytest.fixture(autouse=True)
def authenticated_principal() -> Iterator[None]:
    with principal_scope(TEST_PRINCIPAL):
        yield


@pytest.fixture
def settings() -> Settings:
    return Settings(database_url=None, max_attempts=2, retry_backoff_seconds=1)


@pytest.fixture
def barrier(tmp_path: Path) -> Iterator[SQLiteRuntimeStore]:
    store = SQLiteRuntimeStore(str(tmp_path / "agentbarrier.db"))
    yield store
    store.close()


@pytest.fixture
async def memory_jobs(settings: Settings) -> AsyncIterator[App]:
    app = create_job_app(InMemoryConnector(), settings)
    async with app.open_async():
        yield app


def _docker() -> str | None:
    return shutil.which("docker")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _private_database(server_url: str) -> Iterator[str]:
    name = f"evidence_test_{uuid4().hex[:12]}"
    with psycopg.connect(server_url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        dsn = urlunsplit(urlsplit(server_url)._replace(path=f"/{name}"))
        with psycopg.connect(dsn, autocommit=True) as db:
            db.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        yield dsn
    finally:
        with psycopg.connect(server_url, autocommit=True) as db:
            db.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)
                )
            )


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    """A throwaway PostgreSQL database with pgvector.

    With `TEST_DATABASE_URL`, the tests create a private database on that
    server (its role needs CREATEDB) and drop it afterwards, so the configured
    database itself is never touched. Otherwise they start a container.
    """
    configured = os.getenv("TEST_DATABASE_URL")
    if configured:
        yield from _private_database(configured)
        return
    docker = _docker()
    if docker is None:
        pytest.skip("TEST_DATABASE_URL is unset and Docker is unavailable")
    name = f"evidence-lab-test-{uuid4().hex[:12]}"
    password = secrets.token_urlsafe(18)
    port = _free_port()
    # Host networking with a random port works with Docker Engine and with
    # Podman machines, whose published ports are not forwarded to the host.
    started = subprocess.run(
        [
            docker,
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "host",
            "-e",
            "POSTGRES_USER=evidence",
            "-e",
            "POSTGRES_DB=evidence",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            PGVECTOR_IMAGE,
            "-c",
            f"port={port}",
            "-c",
            "listen_addresses=127.0.0.1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if started.returncode != 0:
        pytest.skip(f"cannot start {PGVECTOR_IMAGE}: {started.stderr.strip()[:200]}")
    try:
        dsn = f"postgresql://evidence:{password}@127.0.0.1:{port}/evidence"
        deadline = time.monotonic() + 60
        while True:
            try:
                with psycopg.connect(dsn, connect_timeout=2) as db:
                    db.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
                    db.commit()
                break
            except psycopg.OperationalError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        yield dsn
    finally:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, check=False)


@pytest.fixture
def schema_dsn(postgres_dsn: str) -> Iterator[str]:
    """A private schema on the test database, first on the search path."""
    schema = f"t_{uuid4().hex[:16]}"
    with psycopg.connect(postgres_dsn, autocommit=True) as db:
        db.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    separator = "&" if "?" in postgres_dsn else "?"
    yield f"{postgres_dsn}{separator}options=-csearch_path%3D{schema},public"
    with psycopg.connect(postgres_dsn, autocommit=True) as db:
        db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
