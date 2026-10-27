"""Run jobs with procrastinate; approvals and the extended run lifecycle.

Replaces the removed Agloom durable runtime:

- installs procrastinate's job queue schema (version pinned in `uv.lock`),
- drops `studio_jobs`; runs that were queued or running in it are failed,
  because their jobs cannot move to the new queue,
- adds the `queued`, `awaiting_approval`, `cancelled`, and `rejected` run
  states, the pending approvals, the procrastinate job id, and the API
  idempotency key to runs,
- makes evidence unique per run, iteration, and criterion, so a resumed run
  rewrites rather than duplicates the evidence of an interrupted iteration.

`studio_run_events` stays: it is the application's ordered event log.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from alembic import op
from procrastinate.schema import SchemaManager

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

_STATUSES = (
    "'queued','running','awaiting_approval','completed','failed','cancelled','rejected'"
)
_ACTIVE = "'queued','running','awaiting_approval'"


def upgrade() -> None:
    op.execute("""UPDATE studio_loop_runs
        SET status='failed', finished_at=now(),
            result = result || jsonb_build_object(
                'error', 'interrupted by the job queue migration')
        WHERE status='running'""")
    op.execute("UPDATE studio_tasks SET status='failed' WHERE status='running'")
    op.execute(
        "ALTER TABLE studio_loop_runs ADD COLUMN approvals JSONB NOT NULL "
        "DEFAULT '[]'::jsonb"
    )
    op.execute("ALTER TABLE studio_loop_runs ADD COLUMN job_id BIGINT")
    op.execute("ALTER TABLE studio_loop_runs ADD COLUMN idempotency_key TEXT")
    op.execute("""UPDATE studio_loop_runs r SET idempotency_key = j.idempotency_key
        FROM studio_jobs j WHERE j.run_id = r.id AND j.idempotency_key IS NOT NULL""")
    op.execute(f"""ALTER TABLE studio_loop_runs ADD CONSTRAINT studio_runs_status_ck
        CHECK (status IN ({_STATUSES}))""")
    op.execute(f"""ALTER TABLE studio_tasks ADD CONSTRAINT studio_tasks_status_ck
        CHECK (status IN ({_STATUSES}))""")
    op.execute("DROP INDEX studio_one_active_run_per_task_idx")
    op.execute(f"""CREATE UNIQUE INDEX studio_one_active_run_per_task_idx
        ON studio_loop_runs(tenant_id, task_id) WHERE status IN ({_ACTIVE})""")
    op.execute("""CREATE UNIQUE INDEX studio_runs_idempotency_idx
        ON studio_loop_runs(tenant_id, user_id, task_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL""")
    op.execute("""DELETE FROM studio_evidence older USING studio_evidence newer
        WHERE older.tenant_id = newer.tenant_id AND older.run_id = newer.run_id
          AND older.iteration = newer.iteration
          AND older.criterion_id = newer.criterion_id
          AND (older.created_at, older.id) < (newer.created_at, newer.id)""")
    op.execute("""CREATE UNIQUE INDEX studio_evidence_iteration_criterion_idx
        ON studio_evidence(tenant_id, run_id, iteration, criterion_id)""")
    op.execute("DROP TABLE studio_jobs")
    # The schema contains literal `%` characters; psycopg sends a statement
    # verbatim only when it is executed without parameters.
    op.get_bind().connection.driver_connection.execute(SchemaManager.get_schema())


def downgrade() -> None:
    op.execute("""DO $$
        DECLARE item record;
        BEGIN
          FOR item IN
            SELECT p.oid::regprocedure AS signature
            FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = current_schema()
              AND starts_with(p.proname, 'procrastinate')
          LOOP
            EXECUTE 'DROP FUNCTION IF EXISTS ' || item.signature || ' CASCADE';
          END LOOP;
        END $$""")
    op.execute("""DROP TABLE IF EXISTS procrastinate_events,
        procrastinate_periodic_defers, procrastinate_jobs, procrastinate_workers
        CASCADE""")
    op.execute("""DROP TYPE IF EXISTS procrastinate_job_to_defer_v1,
        procrastinate_job_event_type, procrastinate_job_status CASCADE""")
    op.execute("""CREATE TABLE studio_jobs (
            id UUID PRIMARY KEY,
            run_id UUID NOT NULL UNIQUE,
            idempotency_key TEXT,
            ownership JSONB NOT NULL DEFAULT '{}'::jsonb,
            payload JSONB NOT NULL,
            resume_mode TEXT NOT NULL DEFAULT 'restart',
            checkpoint_id TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL,
            lease_owner TEXT,
            lease_token BIGINT NOT NULL DEFAULT 0,
            lease_expires_at TIMESTAMPTZ,
            cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
            last_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            CONSTRAINT studio_jobs_identity_ck CHECK (
                jsonb_typeof(ownership->'tenant_id')='string'
                AND length(ownership->>'tenant_id') > 0
                AND jsonb_typeof(ownership->'user_id')='string'
                AND length(ownership->>'user_id') > 0)
        )""")
    op.execute("""CREATE UNIQUE INDEX studio_jobs_scope_key_idx
        ON studio_jobs (ownership, idempotency_key)
        WHERE idempotency_key IS NOT NULL""")
    op.execute("""CREATE INDEX studio_jobs_claim_idx
        ON studio_jobs (status, lease_expires_at, created_at)""")
    op.execute("DROP INDEX IF EXISTS studio_evidence_iteration_criterion_idx")
    op.execute("DROP INDEX IF EXISTS studio_runs_idempotency_idx")
    op.execute("DROP INDEX IF EXISTS studio_one_active_run_per_task_idx")
    op.execute(
        "ALTER TABLE studio_tasks DROP CONSTRAINT IF EXISTS studio_tasks_status_ck"
    )
    op.execute(
        "ALTER TABLE studio_loop_runs DROP CONSTRAINT IF EXISTS studio_runs_status_ck"
    )
    op.execute("""UPDATE studio_loop_runs SET status='failed', finished_at=now()
        WHERE status NOT IN ('running','completed','failed')""")
    op.execute("""UPDATE studio_tasks SET status='failed'
        WHERE status NOT IN ('queued','running','completed','failed')""")
    op.execute("""CREATE UNIQUE INDEX studio_one_active_run_per_task_idx
        ON studio_loop_runs(tenant_id, task_id) WHERE status='running'""")
    op.execute("ALTER TABLE studio_loop_runs DROP COLUMN idempotency_key")
    op.execute("ALTER TABLE studio_loop_runs DROP COLUMN job_id")
    op.execute("ALTER TABLE studio_loop_runs DROP COLUMN approvals")
