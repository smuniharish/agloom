"""Enforce tenant ownership and active-run invariants.

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
    """The owner assigned to rows created before tenant ownership existed."""
    connection = op.get_bind()
    has_rows = connection.execute(
        text("SELECT EXISTS (SELECT 1 FROM studio_projects)")
    ).scalar()
    tenant = os.getenv("STUDIO_BACKFILL_TENANT_ID")
    user = os.getenv("STUDIO_BACKFILL_USER_ID")
    if has_rows and (not tenant or not user):
        raise RuntimeError(
            "Rows without an owner require STUDIO_BACKFILL_TENANT_ID and "
            "STUDIO_BACKFILL_USER_ID"
        )
    return tenant or "migration-empty-tenant", user or "migration-empty-user"


def upgrade() -> None:
    tenant, user = _backfill_identity()
    op.execute("ALTER TABLE studio_loop_runs DROP COLUMN IF EXISTS checkpoint")
    op.execute("ALTER TABLE studio_projects ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute("ALTER TABLE studio_tasks ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE studio_tasks ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute("ALTER TABLE studio_loop_runs ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute("ALTER TABLE studio_evidence ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE studio_artifacts ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    connection = op.get_bind()
    connection.execute(
        text(
            "UPDATE studio_projects SET tenant_id=:tenant, user_id=:user WHERE tenant_id IS NULL OR user_id IS NULL"
        ),
        {"tenant": tenant, "user": user},
    )
    op.execute("""UPDATE studio_tasks t SET tenant_id=p.tenant_id,
        user_id=COALESCE(t.user_id,p.user_id) FROM studio_projects p
        WHERE t.project_id=p.id AND (t.tenant_id IS NULL OR t.user_id IS NULL)""")
    op.execute("""UPDATE studio_loop_runs r SET tenant_id=t.tenant_id,
        user_id=COALESCE(r.user_id,t.user_id) FROM studio_tasks t
        WHERE r.task_id=t.id AND (r.tenant_id IS NULL OR r.user_id IS NULL)""")
    op.execute("""UPDATE studio_evidence e SET tenant_id=r.tenant_id
        FROM studio_loop_runs r WHERE e.run_id=r.id AND e.tenant_id IS NULL""")
    op.execute("""UPDATE studio_artifacts a SET tenant_id=p.tenant_id
        FROM studio_projects p WHERE a.project_id=p.id AND a.tenant_id IS NULL""")
    for table, columns in {
        "studio_projects": ("tenant_id", "user_id"),
        "studio_tasks": ("tenant_id", "user_id"),
        "studio_loop_runs": ("tenant_id", "user_id"),
        "studio_evidence": ("tenant_id",),
        "studio_artifacts": ("tenant_id",),
    }.items():
        for column in columns:
            op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} SET NOT NULL")
    op.execute(
        "ALTER TABLE studio_projects ADD CONSTRAINT "
        "studio_projects_tenant_id_id_key UNIQUE (tenant_id,id)"
    )
    op.execute(
        "ALTER TABLE studio_tasks ADD CONSTRAINT "
        "studio_tasks_tenant_id_id_key UNIQUE (tenant_id,id)"
    )
    op.execute(
        "ALTER TABLE studio_loop_runs ADD CONSTRAINT "
        "studio_loop_runs_tenant_id_id_key UNIQUE (tenant_id,id)"
    )
    for table, constraint in (
        ("studio_tasks", "studio_tasks_project_id_fkey"),
        ("studio_loop_runs", "studio_loop_runs_task_id_fkey"),
        ("studio_evidence", "studio_evidence_run_id_fkey"),
        ("studio_artifacts", "studio_artifacts_project_id_fkey"),
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
    op.execute("""ALTER TABLE studio_tasks
        ADD CONSTRAINT studio_tasks_tenant_project_fkey
        FOREIGN KEY (tenant_id,project_id)
        REFERENCES studio_projects(tenant_id,id)""")
    op.execute("""ALTER TABLE studio_loop_runs
        ADD CONSTRAINT studio_runs_tenant_task_fkey
        FOREIGN KEY (tenant_id,task_id)
        REFERENCES studio_tasks(tenant_id,id)""")
    op.execute("""ALTER TABLE studio_evidence
        ADD CONSTRAINT studio_evidence_tenant_run_fkey
        FOREIGN KEY (tenant_id,run_id)
        REFERENCES studio_loop_runs(tenant_id,id)""")
    op.execute("""ALTER TABLE studio_artifacts
        ADD CONSTRAINT studio_artifacts_tenant_project_fkey
        FOREIGN KEY (tenant_id,project_id)
        REFERENCES studio_projects(tenant_id,id)""")
    connection.execute(
        text("""UPDATE studio_jobs j SET ownership =
            j.ownership || jsonb_build_object('tenant_id',r.tenant_id,'user_id',r.user_id)
            FROM studio_loop_runs r WHERE j.run_id=r.id
            AND NOT (j.ownership ? 'tenant_id' AND j.ownership ? 'user_id')""")
    )
    connection.execute(
        text("""UPDATE studio_run_events e SET ownership =
            e.ownership || jsonb_build_object('tenant_id',r.tenant_id,'user_id',r.user_id)
            FROM studio_loop_runs r WHERE e.run_id=r.id
            AND NOT (e.ownership ? 'tenant_id' AND e.ownership ? 'user_id')""")
    )
    identity_check = """jsonb_typeof(ownership->'tenant_id')='string'
        AND length(ownership->>'tenant_id') > 0
        AND jsonb_typeof(ownership->'user_id')='string'
        AND length(ownership->>'user_id') > 0"""
    op.execute(f"""ALTER TABLE studio_jobs ADD CONSTRAINT studio_jobs_identity_ck
        CHECK ({identity_check})""")
    op.execute(
        f"""ALTER TABLE studio_run_events ADD CONSTRAINT studio_events_identity_ck
        CHECK ({identity_check})"""
    )
    op.execute(
        "CREATE INDEX studio_projects_tenant_created_idx ON studio_projects(tenant_id,created_at)"
    )
    op.execute(
        "CREATE INDEX studio_tasks_tenant_project_idx ON studio_tasks(tenant_id,project_id)"
    )
    op.execute(
        "CREATE INDEX studio_runs_tenant_task_idx ON studio_loop_runs(tenant_id,task_id)"
    )
    op.execute(
        "CREATE INDEX studio_evidence_tenant_run_idx ON studio_evidence(tenant_id,run_id)"
    )
    op.execute(
        "CREATE INDEX studio_artifacts_tenant_project_idx ON studio_artifacts(tenant_id,project_id)"
    )
    op.execute("""CREATE UNIQUE INDEX studio_one_active_run_per_task_idx
        ON studio_loop_runs(tenant_id,task_id) WHERE status='running'""")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS studio_one_active_run_per_task_idx")
    for index in (
        "studio_artifacts_tenant_project_idx",
        "studio_evidence_tenant_run_idx",
        "studio_runs_tenant_task_idx",
        "studio_tasks_tenant_project_idx",
        "studio_projects_tenant_created_idx",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index}")
    op.execute(
        "ALTER TABLE studio_run_events DROP CONSTRAINT IF EXISTS studio_events_identity_ck"
    )
    op.execute(
        "ALTER TABLE studio_jobs DROP CONSTRAINT IF EXISTS studio_jobs_identity_ck"
    )
    for table, constraint in (
        ("studio_artifacts", "studio_artifacts_tenant_project_fkey"),
        ("studio_evidence", "studio_evidence_tenant_run_fkey"),
        ("studio_loop_runs", "studio_runs_tenant_task_fkey"),
        ("studio_tasks", "studio_tasks_tenant_project_fkey"),
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint}")
    op.execute(
        "ALTER TABLE studio_tasks ADD CONSTRAINT studio_tasks_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES studio_projects(id)"
    )
    op.execute(
        "ALTER TABLE studio_loop_runs ADD CONSTRAINT studio_loop_runs_task_id_fkey "
        "FOREIGN KEY (task_id) REFERENCES studio_tasks(id)"
    )
    op.execute(
        "ALTER TABLE studio_evidence ADD CONSTRAINT studio_evidence_run_id_fkey "
        "FOREIGN KEY (run_id) REFERENCES studio_loop_runs(id)"
    )
    op.execute(
        "ALTER TABLE studio_artifacts ADD CONSTRAINT studio_artifacts_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES studio_projects(id)"
    )
    op.execute(
        "ALTER TABLE studio_loop_runs DROP CONSTRAINT "
        "IF EXISTS studio_loop_runs_tenant_id_id_key"
    )
    op.execute(
        "ALTER TABLE studio_tasks DROP CONSTRAINT "
        "IF EXISTS studio_tasks_tenant_id_id_key"
    )
    op.execute(
        "ALTER TABLE studio_projects DROP CONSTRAINT "
        "IF EXISTS studio_projects_tenant_id_id_key"
    )
    for table, column in (
        ("studio_artifacts", "tenant_id"),
        ("studio_evidence", "tenant_id"),
        ("studio_loop_runs", "user_id"),
        ("studio_tasks", "user_id"),
        ("studio_tasks", "tenant_id"),
        ("studio_projects", "user_id"),
    ):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")
    op.execute("ALTER TABLE studio_loop_runs ALTER COLUMN tenant_id DROP NOT NULL")
    op.execute("ALTER TABLE studio_projects ALTER COLUMN tenant_id DROP NOT NULL")
    op.execute("ALTER TABLE studio_loop_runs ADD COLUMN IF NOT EXISTS checkpoint JSONB")
