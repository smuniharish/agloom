"""Schema migrations. The graph check is offline; the rest needs PostgreSQL."""

from __future__ import annotations

import json
from uuid import uuid4

import psycopg
import pytest
from agentbarrier.runtime import PostgresRuntimeStore
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from backend.config import APP_ROOT, Settings
from backend.runtime import missing_platform_schema
from backend.setup_database import alembic_config, setup_database
from backend.store import EvidenceStore
from psycopg import sql

APP_TABLES = {
    "sources",
    "loop_runs",
    "claims",
    "evidence",
    "loop_iterations",
    "decisions",
    "run_events",
}


def _tables(dsn: str) -> set[str]:
    with psycopg.connect(dsn) as db:
        rows = db.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
    return {row[0] for row in rows}


def _columns(dsn: str, table: str) -> set[str]:
    with psycopg.connect(dsn) as db:
        rows = db.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s",
            (table,),
        ).fetchall()
    return {row[0] for row in rows}


def test_migration_graph_is_well_formed() -> None:
    config = Config(str(APP_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(APP_ROOT / "migrations"))
    script = ScriptDirectory.from_config(config)
    assert script.get_heads() == ["0004"]
    revisions = {rev.revision: rev for rev in script.walk_revisions()}
    assert set(revisions) == {"0001", "0002", "0003", "0004"}
    assert revisions["0001"].down_revision is None
    assert revisions["0002"].down_revision == "0001"
    assert revisions["0003"].down_revision == "0002"
    assert revisions["0004"].down_revision == "0003"


@pytest.mark.postgres
async def test_upgrade_head_and_downgrade_base(schema_dsn: str) -> None:
    command.upgrade(alembic_config(schema_dsn), "head")
    tables = _tables(schema_dsn)
    assert tables >= APP_TABLES
    assert not tables & {"evidence_jobs", "evidence_run_events"}
    assert {
        "request",
        "idempotency_key",
        "job_id",
        "approval",
        "published_at",
        "updated_at",
    } <= _columns(schema_dsn, "loop_runs")
    assert "action_id" in _columns(schema_dsn, "decisions")
    assert await EvidenceStore(schema_dsn).verify_schema() == set()

    command.downgrade(alembic_config(schema_dsn), "base")
    assert not _tables(schema_dsn) & (APP_TABLES | {"evidence_jobs"})


@pytest.mark.postgres
def test_upgrade_from_0003_carries_runs_requests_and_events(schema_dsn: str) -> None:
    command.upgrade(alembic_config(schema_dsn), "0003")
    run_id = uuid4()
    ownership = json.dumps({"tenant_id": "tenant-x", "user_id": "user-x"})
    with psycopg.connect(schema_dsn) as db:
        db.execute(
            """INSERT INTO loop_runs (id, objective, topics, status, objective_state,
               accuracy, execution_status, tenant_id, user_id)
               VALUES (%s, 'Verify controls', '["access"]', 'running', 'incomplete',
               %s, 'running', 'tenant-x', 'user-x')""",
            (
                run_id,
                json.dumps(
                    {
                        "citation_precision": 0,
                        "claim_support": 0,
                        "coverage": 0,
                        "quality": 0,
                    }
                ),
            ),
        )
        db.execute(
            """INSERT INTO evidence_jobs
               (id, run_id, idempotency_key, ownership, payload, max_attempts, status)
               VALUES (%s, %s, 'key-1', %s, %s, 3, 'running')""",
            (
                uuid4(),
                run_id,
                ownership,
                json.dumps(
                    {
                        "objective": "Verify controls",
                        "topics": ["access"],
                        "mode": "one_shot",
                    }
                ),
            ),
        )
        db.execute(
            """INSERT INTO evidence_run_events (run_id, ownership, event_type, payload)
               VALUES (%s, %s, 'started', '{"message": "accepted"}')""",
            (run_id, ownership),
        )

    command.upgrade(alembic_config(schema_dsn), "head")

    with psycopg.connect(schema_dsn) as db:
        row = db.execute(
            """SELECT request->>'mode', idempotency_key, status, execution_status, error
               FROM loop_runs WHERE id=%s""",
            (run_id,),
        ).fetchone()
        assert row[:4] == ("one_shot", "key-1", "failed", "failed")
        assert "submit the run again" in row[4]
        events = db.execute(
            "SELECT tenant_id, event_type FROM run_events WHERE run_id=%s", (run_id,)
        ).fetchall()
        assert events == [("tenant-x", "started")]

    command.downgrade(alembic_config(schema_dsn), "0003")
    with psycopg.connect(schema_dsn) as db:
        restored = db.execute(
            "SELECT ownership->>'tenant_id', event_type FROM evidence_run_events"
        ).fetchall()
    assert restored == [("tenant-x", "started")]


@pytest.mark.postgres
def test_stamped_baseline_upgrades_to_head(schema_dsn: str) -> None:
    config = alembic_config(schema_dsn)
    command.upgrade(config, "0001")
    command.stamp(config, "0001", purge=True)
    with psycopg.connect(schema_dsn) as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0001",
        )
    command.upgrade(config, "head")
    assert "tenant_id" in _columns(schema_dsn, "loop_runs")


@pytest.mark.postgres
async def test_setup_database_prepares_every_schema(schema_dsn: str) -> None:
    barrier_schema = f"ab_{uuid4().hex[:12]}"
    settings = Settings(database_url=schema_dsn, barrier_schema=barrier_schema)
    try:
        await setup_database(settings)
        await setup_database(settings)
        assert await EvidenceStore(schema_dsn).verify_schema() == set()
        assert await missing_platform_schema(schema_dsn) == set()
        PostgresRuntimeStore(schema_dsn, schema=barrier_schema).close()
    finally:
        with psycopg.connect(schema_dsn, autocommit=True) as db:
            db.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(barrier_schema)
                )
            )


@pytest.mark.postgres
async def test_unmigrated_database_is_reported(schema_dsn: str) -> None:
    missing = await EvidenceStore(schema_dsn).verify_schema()
    assert "loop_runs" in missing
    assert "alembic_version.0004" in missing
    assert "procrastinate_jobs" in await missing_platform_schema(schema_dsn)
