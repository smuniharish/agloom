from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import subprocess
import time
from collections.abc import Iterator
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend.auth import Principal, StaticAuthenticator, principal_scope

TEST_PRINCIPAL = Principal("test-user", "test-tenant")
AUTHORIZATION = {"Authorization": "Bearer test-token"}
POSTGRES_IMAGE = os.getenv("TEST_POSTGRES_IMAGE", "pgvector/pgvector:pg18")


@pytest.fixture(autouse=True)
def principal() -> Iterator[Principal]:
    with principal_scope(TEST_PRINCIPAL) as value:
        yield value


@pytest.fixture
def authenticator() -> StaticAuthenticator:
    return StaticAuthenticator(TEST_PRINCIPAL)


@pytest.fixture
def anyio_backend() -> tuple[str, dict[str, object]]:
    # psycopg's asynchronous driver needs a selector event loop on Windows.
    options = {"loop_factory": asyncio.SelectorEventLoop} if os.name == "nt" else {}
    return "asyncio", options


def with_database(url: str, name: str) -> str:
    return urlsplit(url)._replace(path=f"/{name}").geturl()


def _docker(*arguments: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    ).stdout.strip()


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    """A PostgreSQL server with pgvector: TEST_DATABASE_URL or a throwaway container.

    Container engines that run in a VM without localhost port forwarding can set
    TEST_DOCKER_PUBLISH_IP (for example 0.0.0.0) and TEST_DOCKER_HOST (the VM address).
    """
    configured = os.getenv("TEST_DATABASE_URL")
    if configured:
        yield configured
        return
    if shutil.which("docker") is None:
        pytest.skip("PostgreSQL tests need TEST_DATABASE_URL or Docker")
    name = f"studio-test-{uuid4().hex[:12]}"
    password = secrets.token_urlsafe(18)
    publish_ip = os.getenv("TEST_DOCKER_PUBLISH_IP", "127.0.0.1")
    host = os.getenv("TEST_DOCKER_HOST", "127.0.0.1")
    try:
        _docker(
            "run",
            "--detach",
            "--name",
            name,
            "--env",
            "POSTGRES_PASSWORD",
            "--env",
            "POSTGRES_USER=studio",
            "--publish",
            f"{publish_ip}::5432",
            POSTGRES_IMAGE,
            env={**os.environ, "POSTGRES_PASSWORD": password},
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"cannot start a PostgreSQL container: {error}")
    try:
        host_port = _docker("port", name, "5432/tcp").splitlines()[0].rsplit(":", 1)[1]
        url = f"postgresql://studio:{password}@{host}:{host_port}/studio"
        deadline = time.monotonic() + 60
        while True:
            try:
                with psycopg.connect(url, connect_timeout=3) as db:
                    db.execute("SELECT 1")
                break
            except psycopg.OperationalError as error:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"the PostgreSQL test container is unreachable at {host}:{host_port}; "
                        "set TEST_DATABASE_URL, or TEST_DOCKER_HOST and TEST_DOCKER_PUBLISH_IP"
                    ) from error
                time.sleep(0.5)
        yield url
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, check=False
        )


@pytest.fixture(scope="session")
def migrated_template(postgres_url: str) -> Iterator[str]:
    """One fully migrated database that every PostgreSQL test clones."""
    from backend.migrate import migrate

    name = f"studio_template_{uuid4().hex[:10]}"
    with psycopg.connect(postgres_url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        migrate(with_database(postgres_url, name))
        yield name
    finally:
        with psycopg.connect(postgres_url, autocommit=True) as db:
            db.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)
                )
            )


@pytest.fixture
def database_url(postgres_url: str, migrated_template: str) -> Iterator[str]:
    name = f"studio_test_{uuid4().hex[:10]}"
    with psycopg.connect(postgres_url, autocommit=True) as db:
        db.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                sql.Identifier(name), sql.Identifier(migrated_template)
            )
        )
    try:
        yield with_database(postgres_url, name)
    finally:
        with psycopg.connect(postgres_url, autocommit=True) as db:
            db.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)
                )
            )
