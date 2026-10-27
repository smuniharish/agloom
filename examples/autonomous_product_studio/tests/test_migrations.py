"""Alembic migrations, including procrastinate's job queue schema."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from psycopg import sql

from backend.migrate import alembic_config
from backend.settings import APP_ROOT
from tests.conftest import with_database

STUDIO_TABLES = {
    "studio_projects",
    "studio_tasks",
    "studio_loop_runs",
    "studio_evidence",
    "studio_artifacts",
    "studio_run_events",
}
QUEUE_TABLES = {
    "procrastinate_jobs",
    "procrastinate_events",
    "procrastinate_periodic_defers",
    "procrastinate_workers",
}


def test_migration_graph_is_well_formed() -> None:
    config = Config(str(APP_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(APP_ROOT / "migrations"))
    script = ScriptDirectory.from_config(config)

    assert script.get_heads() == ["0005"]
    revisions = {rev.revision: rev for rev in script.walk_revisions()}
    assert set(revisions) == {"0001", "0002", "0003", "0004", "0005"}
    assert revisions["0001"].down_revision is None
    assert revisions["0002"].down_revision == "0001"
    assert revisions["0003"].down_revision == "0002"
    assert revisions["0004"].down_revision == "0003"
    assert revisions["0005"].down_revision == "0004"


@pytest.fixture
def empty_database(postgres_url: str) -> Iterator[str]:
    name = f"studio_migration_{uuid4().hex[:10]}"
    with psycopg.connect(postgres_url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield with_database(postgres_url, name)
    finally:
        with psycopg.connect(postgres_url, autocommit=True) as db:
            db.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)
                )
            )


def _tables(url: str) -> set[str]:
    with psycopg.connect(url) as db:
        return {
            row[0]
            for row in db.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ).fetchall()
        }


def _columns(url: str, table: str) -> set[str]:
    with psycopg.connect(url) as db:
        return {
            row[0]
            for row in db.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s",
                (table,),
            ).fetchall()
        }


@pytest.mark.postgres
def test_upgrade_head_installs_the_application_and_queue_schema(
    empty_database: str,
) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "head")
    tables = _tables(empty_database)

    assert tables >= STUDIO_TABLES | QUEUE_TABLES
    assert "studio_jobs" not in tables
    assert {
        "approvals",
        "job_id",
        "idempotency_key",
        "tenant_id",
        "user_id",
    } <= _columns(empty_database, "studio_loop_runs")
    with psycopg.connect(empty_database) as db:
        functions = db.execute(
            "SELECT count(*) FROM pg_proc WHERE starts_with(proname, 'procrastinate')"
        ).fetchone()[0]
        db.execute(
            """INSERT INTO studio_projects (id,name,objective,acceptance_criteria,embedding,tenant_id,user_id)
               VALUES (gen_random_uuid(),'p','objective','[]',array_fill(0,ARRAY[384])::vector,'t','u')"""
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            db.execute(
                """INSERT INTO studio_tasks (id,project_id,title,objective,status,tenant_id,user_id)
                   SELECT gen_random_uuid(), id, 't', 'objective', 'paused', 't', 'u' FROM studio_projects"""
            )
    assert functions > 0

    command.downgrade(config, "base")
    remaining = _tables(empty_database)
    assert not remaining & (STUDIO_TABLES | QUEUE_TABLES | {"studio_jobs"})
    with psycopg.connect(empty_database) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM pg_proc WHERE starts_with(proname, 'procrastinate')"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.postgres
def test_downgrading_0004_restores_the_previous_schema(empty_database: str) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "head")
    command.downgrade(config, "0003")
    tables = _tables(empty_database)
    assert "studio_jobs" in tables
    assert not tables & QUEUE_TABLES
    assert "approvals" not in _columns(empty_database, "studio_loop_runs")
    command.upgrade(config, "head")
    assert _tables(empty_database) >= QUEUE_TABLES


@pytest.mark.postgres
def test_stamp_baseline_marks_an_existing_schema_as_current(
    empty_database: str,
) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "0001")
    command.stamp(config, "0001", purge=True)
    with psycopg.connect(empty_database) as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0001",
        )
    command.upgrade(config, "head")
    assert "tenant_id" in _columns(empty_database, "studio_projects")
    with psycopg.connect(empty_database) as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0005",
        )


@pytest.mark.postgres
def test_runs_can_wait_for_reconciliation_and_count_as_active(
    empty_database: str,
) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "head")
    with psycopg.connect(empty_database) as db:
        project = db.execute(
            """INSERT INTO studio_projects (id,name,objective,acceptance_criteria,embedding,tenant_id,user_id)
               VALUES (gen_random_uuid(),'p','objective','[]',array_fill(0,ARRAY[384])::vector,'t','u')
               RETURNING id"""
        ).fetchone()[0]
        task = db.execute(
            """INSERT INTO studio_tasks (id,project_id,title,objective,status,tenant_id,user_id)
               VALUES (gen_random_uuid(), %s, 't', 'objective', 'awaiting_reconciliation', 't', 'u')
               RETURNING id""",
            (project,),
        ).fetchone()[0]
        insert_run = """INSERT INTO studio_loop_runs (id,task_id,status,proof,result,tenant_id,user_id)
            VALUES (gen_random_uuid(), %s, %s, '{}', '{}', 't', 'u')"""
        db.execute(insert_run, (task, "awaiting_reconciliation"))
        db.commit()
        with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
            db.execute(insert_run, (task, "queued"))

    command.downgrade(config, "0004")
    with psycopg.connect(empty_database) as db:
        statuses = db.execute("SELECT status FROM studio_loop_runs").fetchall()
        task_status = db.execute("SELECT status FROM studio_tasks").fetchone()
    assert statuses == [("failed",)]
    assert task_status == ("failed",)
    command.upgrade(config, "head")


@pytest.mark.postgres
def test_runs_in_flight_in_the_removed_job_table_are_failed(
    empty_database: str,
) -> None:
    config = alembic_config(empty_database)
    command.upgrade(config, "0003")
    with psycopg.connect(empty_database) as db:
        project = db.execute(
            """INSERT INTO studio_projects (id,name,objective,acceptance_criteria,embedding,tenant_id,user_id)
               VALUES (gen_random_uuid(),'p','objective','[]',array_fill(0,ARRAY[384])::vector,'t','u')
               RETURNING id"""
        ).fetchone()[0]
        task = db.execute(
            """INSERT INTO studio_tasks (id,project_id,title,objective,status,tenant_id,user_id)
               VALUES (gen_random_uuid(),%s,'t','objective','running','t','u') RETURNING id""",
            (project,),
        ).fetchone()[0]
        run = db.execute(
            """INSERT INTO studio_loop_runs (id,task_id,status,proof,tenant_id,user_id)
               VALUES (gen_random_uuid(),%s,'running','{}','t','u') RETURNING id""",
            (task,),
        ).fetchone()[0]
        db.execute(
            """INSERT INTO studio_jobs (id,run_id,idempotency_key,ownership,payload,max_attempts)
               VALUES (gen_random_uuid(),%s,'key-1','{"tenant_id":"t","user_id":"u"}','{}',3)""",
            (run,),
        )
    command.upgrade(config, "head")
    with psycopg.connect(empty_database) as db:
        status, result, key = db.execute(
            "SELECT status, result, idempotency_key FROM studio_loop_runs WHERE id=%s",
            (run,),
        ).fetchone()
        task_status = db.execute(
            "SELECT status FROM studio_tasks WHERE id=%s", (task,)
        ).fetchone()[0]

    assert status == "failed"
    assert result["error"] == "interrupted by the job queue migration"
    assert key == "key-1"
    assert task_status == "failed"
