"""Additive tenant and checkpoint columns.

Adds a nullable tenant column and a temporary checkpoint placeholder. Revision
0003 makes tenant ownership mandatory and removes the placeholder: checkpoints
live in the tables of the LangGraph PostgreSQL checkpointer.

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
    op.execute("ALTER TABLE loop_runs ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE loop_runs ADD COLUMN IF NOT EXISTS checkpoint JSONB")
    op.execute(
        "CREATE INDEX IF NOT EXISTS loop_runs_tenant_idx ON loop_runs (tenant_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS loop_runs_tenant_idx")
    op.execute("ALTER TABLE loop_runs DROP COLUMN IF EXISTS checkpoint")
    op.execute("ALTER TABLE loop_runs DROP COLUMN IF EXISTS tenant_id")
