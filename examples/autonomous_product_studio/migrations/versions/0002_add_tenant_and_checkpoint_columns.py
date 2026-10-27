"""Forward-looking tenant/checkpoint columns.

Adds nullable, additive columns to support future multi-tenant scoping and
mid-run checkpoint/resume of loop executions. Nothing in the current
application code reads or writes these columns yet; they exist so a future
change can start using them without another schema migration, and so
existing rows remain valid (all new columns are nullable with no default
that would require a table rewrite).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-27

"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE studio_projects ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute(
        "CREATE INDEX IF NOT EXISTS studio_projects_tenant_idx "
        "ON studio_projects (tenant_id)"
    )
    op.execute("ALTER TABLE studio_loop_runs ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE studio_loop_runs ADD COLUMN IF NOT EXISTS checkpoint JSONB")
    op.execute(
        "CREATE INDEX IF NOT EXISTS studio_loop_runs_tenant_idx "
        "ON studio_loop_runs (tenant_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS studio_loop_runs_tenant_idx")
    op.execute("ALTER TABLE studio_loop_runs DROP COLUMN IF EXISTS checkpoint")
    op.execute("ALTER TABLE studio_loop_runs DROP COLUMN IF EXISTS tenant_id")
    op.execute("DROP INDEX IF EXISTS studio_projects_tenant_idx")
    op.execute("ALTER TABLE studio_projects DROP COLUMN IF EXISTS tenant_id")
