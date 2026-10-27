"""Runs waiting for an operator to reconcile an unknown release outcome.

When a consequential tool raises after it started, agentbarrier cannot tell
whether the action took effect; Agloom pauses the checkpointed run until an
operator reconciles it. This revision adds the `awaiting_reconciliation` run
and task state and counts it as active (one active run per task).

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

_STATUSES = (
    "'queued','running','awaiting_approval','awaiting_reconciliation',"
    "'completed','failed','cancelled','rejected'"
)
_ACTIVE = "'queued','running','awaiting_approval','awaiting_reconciliation'"
_PREVIOUS_STATUSES = (
    "'queued','running','awaiting_approval','completed','failed','cancelled','rejected'"
)
_PREVIOUS_ACTIVE = "'queued','running','awaiting_approval'"


def _constrain(statuses: str, active: str) -> None:
    for table, name in (
        ("studio_loop_runs", "studio_runs_status_ck"),
        ("studio_tasks", "studio_tasks_status_ck"),
    ):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK (status IN ({statuses}))"
        )
    op.execute("DROP INDEX studio_one_active_run_per_task_idx")
    op.execute(f"""CREATE UNIQUE INDEX studio_one_active_run_per_task_idx
        ON studio_loop_runs(tenant_id, task_id) WHERE status IN ({active})""")


def upgrade() -> None:
    _constrain(_STATUSES, _ACTIVE)


def downgrade() -> None:
    # Revision 0004 cannot represent a run that waits for reconciliation.
    op.execute("""UPDATE studio_loop_runs
        SET status='failed', finished_at=now(),
            result = result || jsonb_build_object(
                'error', 'awaiting reconciliation when the schema was downgraded')
        WHERE status='awaiting_reconciliation'""")
    op.execute(
        "UPDATE studio_tasks SET status='failed' WHERE status='awaiting_reconciliation'"
    )
    _constrain(_PREVIOUS_STATUSES, _PREVIOUS_ACTIVE)
