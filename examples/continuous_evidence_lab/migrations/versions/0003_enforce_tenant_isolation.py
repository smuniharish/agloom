"""Enforce tenant ownership for evidence data and job execution.

Rows created before tenant isolation are assigned to the owner given by
`BACKFILL_TENANT_ID` and `BACKFILL_USER_ID`; the migration aborts instead of
guessing when such rows exist and no owner is configured.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import os

from alembic import op
from sqlalchemy import text

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _backfill_identity() -> tuple[str, str]:
    connection = op.get_bind()
    has_rows = connection.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM loop_runs) OR EXISTS (SELECT 1 FROM sources)"
        )
    ).scalar()
    tenant = os.getenv("BACKFILL_TENANT_ID")
    user = os.getenv("BACKFILL_USER_ID")
    if has_rows and (not tenant or not user):
        raise RuntimeError(
            "Existing rows require BACKFILL_TENANT_ID and BACKFILL_USER_ID"
        )
    return tenant or "migration-empty-tenant", user or "migration-empty-user"


def upgrade() -> None:
    tenant, user = _backfill_identity()
    op.execute("ALTER TABLE loop_runs DROP COLUMN IF EXISTS checkpoint")
    for table in ("sources", "claims", "evidence", "loop_iterations", "decisions"):
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE loop_runs ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute("ALTER TABLE decisions ADD COLUMN IF NOT EXISTS user_id TEXT")
    connection = op.get_bind()
    connection.execute(
        text("UPDATE sources SET tenant_id=:tenant WHERE tenant_id IS NULL"),
        {"tenant": tenant},
    )
    connection.execute(
        text("""UPDATE loop_runs SET tenant_id=:tenant,user_id=:user
            WHERE tenant_id IS NULL OR user_id IS NULL"""),
        {"tenant": tenant, "user": user},
    )
    for table in ("claims", "evidence", "loop_iterations"):
        op.execute(f"""UPDATE {table} c SET tenant_id=r.tenant_id FROM loop_runs r
            WHERE c.run_id=r.id AND c.tenant_id IS NULL""")
    connection.execute(
        text("""UPDATE decisions d SET tenant_id=r.tenant_id,user_id=:user
            FROM loop_runs r WHERE d.run_id=r.id
            AND (d.tenant_id IS NULL OR d.user_id IS NULL)"""),
        {"user": user},
    )
    for table, columns in {
        "sources": ("tenant_id",),
        "loop_runs": ("tenant_id", "user_id"),
        "claims": ("tenant_id",),
        "evidence": ("tenant_id",),
        "loop_iterations": ("tenant_id",),
        "decisions": ("tenant_id", "user_id"),
    }.items():
        for column in columns:
            op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} SET NOT NULL")
    op.execute("ALTER TABLE sources DROP CONSTRAINT sources_pkey")
    op.execute("ALTER TABLE sources ADD PRIMARY KEY (tenant_id,id)")
    op.execute(
        "ALTER TABLE loop_runs ADD CONSTRAINT "
        "loop_runs_tenant_id_id_key UNIQUE (tenant_id,id)"
    )
    for table, constraint in (
        ("claims", "claims_run_id_fkey"),
        ("evidence", "evidence_run_id_fkey"),
        ("loop_iterations", "loop_iterations_run_id_fkey"),
        ("decisions", "decisions_run_id_fkey"),
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
    op.execute("""ALTER TABLE claims
        ADD CONSTRAINT claims_tenant_run_fkey
        FOREIGN KEY (tenant_id,run_id) REFERENCES loop_runs(tenant_id,id)""")
    op.execute("""ALTER TABLE evidence
        ADD CONSTRAINT evidence_tenant_run_fkey
        FOREIGN KEY (tenant_id,run_id) REFERENCES loop_runs(tenant_id,id)""")
    op.execute("""ALTER TABLE evidence
        ADD CONSTRAINT evidence_tenant_source_fkey
        FOREIGN KEY (tenant_id,source_id) REFERENCES sources(tenant_id,id)""")
    op.execute("""ALTER TABLE loop_iterations
        ADD CONSTRAINT iterations_tenant_run_fkey
        FOREIGN KEY (tenant_id,run_id) REFERENCES loop_runs(tenant_id,id)""")
    op.execute("""ALTER TABLE decisions
        ADD CONSTRAINT decisions_tenant_run_fkey
        FOREIGN KEY (tenant_id,run_id) REFERENCES loop_runs(tenant_id,id)""")
    connection.execute(
        text("""UPDATE evidence_jobs j SET ownership =
        j.ownership || jsonb_build_object('tenant_id',r.tenant_id,'user_id',r.user_id)
        FROM loop_runs r WHERE j.run_id=r.id
        AND NOT (j.ownership ? 'tenant_id' AND j.ownership ? 'user_id')""")
    )
    connection.execute(
        text("""UPDATE evidence_run_events e SET ownership =
        e.ownership || jsonb_build_object('tenant_id',r.tenant_id,'user_id',r.user_id)
        FROM loop_runs r WHERE e.run_id=r.id
        AND NOT (e.ownership ? 'tenant_id' AND e.ownership ? 'user_id')""")
    )
    identity_check = """jsonb_typeof(ownership->'tenant_id')='string'
        AND length(ownership->>'tenant_id') > 0
        AND jsonb_typeof(ownership->'user_id')='string'
        AND length(ownership->>'user_id') > 0"""
    op.execute(f"""ALTER TABLE evidence_jobs ADD CONSTRAINT evidence_jobs_identity_ck
        CHECK ({identity_check})""")
    op.execute(
        f"""ALTER TABLE evidence_run_events ADD CONSTRAINT evidence_events_identity_ck
        CHECK ({identity_check})"""
    )
    op.execute("CREATE INDEX sources_tenant_idx ON sources(tenant_id)")
    op.execute(
        "CREATE INDEX loop_runs_tenant_created_idx ON loop_runs(tenant_id,created_at)"
    )
    op.execute("CREATE INDEX claims_tenant_run_idx ON claims(tenant_id,run_id)")
    op.execute("CREATE INDEX evidence_tenant_run_idx ON evidence(tenant_id,run_id)")
    op.execute(
        "CREATE INDEX iterations_tenant_run_idx ON loop_iterations(tenant_id,run_id)"
    )
    op.execute("CREATE INDEX decisions_tenant_run_idx ON decisions(tenant_id,run_id)")


def downgrade() -> None:
    for index in (
        "decisions_tenant_run_idx",
        "iterations_tenant_run_idx",
        "evidence_tenant_run_idx",
        "claims_tenant_run_idx",
        "loop_runs_tenant_created_idx",
        "sources_tenant_idx",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index}")
    op.execute("""ALTER TABLE evidence_run_events
        DROP CONSTRAINT IF EXISTS evidence_events_identity_ck""")
    op.execute(
        "ALTER TABLE evidence_jobs DROP CONSTRAINT IF EXISTS evidence_jobs_identity_ck"
    )
    for table, constraint in (
        ("decisions", "decisions_tenant_run_fkey"),
        ("loop_iterations", "iterations_tenant_run_fkey"),
        ("evidence", "evidence_tenant_source_fkey"),
        ("evidence", "evidence_tenant_run_fkey"),
        ("claims", "claims_tenant_run_fkey"),
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
    for table, constraint in (
        ("claims", "claims_run_id_fkey"),
        ("evidence", "evidence_run_id_fkey"),
        ("loop_iterations", "loop_iterations_run_id_fkey"),
        ("decisions", "decisions_run_id_fkey"),
    ):
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
            "FOREIGN KEY (run_id) REFERENCES loop_runs(id)"
        )
    op.execute(
        "ALTER TABLE loop_runs DROP CONSTRAINT IF EXISTS loop_runs_tenant_id_id_key"
    )
    op.execute("ALTER TABLE sources DROP CONSTRAINT sources_pkey")
    op.execute("ALTER TABLE sources ADD PRIMARY KEY (id)")
    for table, column in (
        ("decisions", "user_id"),
        ("decisions", "tenant_id"),
        ("loop_iterations", "tenant_id"),
        ("evidence", "tenant_id"),
        ("claims", "tenant_id"),
        ("loop_runs", "user_id"),
        ("sources", "tenant_id"),
    ):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")
    op.execute("ALTER TABLE loop_runs ALTER COLUMN tenant_id DROP NOT NULL")
    op.execute("ALTER TABLE loop_runs ADD COLUMN IF NOT EXISTS checkpoint JSONB")
