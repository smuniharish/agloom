"""Run on procrastinate jobs, with reviewer approvals and an application event log.

* `loop_runs` gains the submitted `request`, the per-user `idempotency_key`,
  the current procrastinate `job_id`, the reviewer `approval`, `published_at`,
  and `updated_at`; new runs start with `execution_status='queued'`.
* `decisions.action_id` links each reviewer decision to its agentbarrier action.
* `run_events` is the ordered, tenant-scoped event log behind the SSE stream; the
  events of `evidence_run_events` are copied into it.
* `evidence_jobs` and `evidence_run_events` are dropped. Runs that were still
  executing under the previous job tables cannot be resumed and are marked
  failed; submit them again.

The procrastinate job-queue schema, the LangGraph checkpointer tables, and the
agentbarrier runtime schema are created by `python -m backend.setup_database`.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE loop_runs ADD COLUMN request JSONB")
    op.execute("""UPDATE loop_runs r SET request = j.payload
        FROM evidence_jobs j WHERE j.run_id = r.id""")
    op.execute("""UPDATE loop_runs SET request = jsonb_build_object(
        'objective', objective, 'topics', topics, 'mode', 'loop')
        WHERE request IS NULL""")
    op.execute("ALTER TABLE loop_runs ALTER COLUMN request SET NOT NULL")
    op.execute("ALTER TABLE loop_runs ADD COLUMN idempotency_key TEXT")
    op.execute("""UPDATE loop_runs r SET idempotency_key = j.idempotency_key
        FROM evidence_jobs j WHERE j.run_id = r.id""")
    op.execute("""CREATE UNIQUE INDEX loop_runs_idempotency_idx
        ON loop_runs (tenant_id, user_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL""")
    op.execute("ALTER TABLE loop_runs ADD COLUMN job_id BIGINT")
    op.execute("ALTER TABLE loop_runs ADD COLUMN approval JSONB")
    op.execute("ALTER TABLE loop_runs ADD COLUMN published_at TIMESTAMPTZ")
    op.execute(
        "ALTER TABLE loop_runs ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
    )
    op.execute(
        "ALTER TABLE loop_runs ALTER COLUMN execution_status SET DEFAULT 'queued'"
    )
    op.execute("""UPDATE loop_runs SET status='failed', execution_status='failed',
        objective_state='failed', termination='execution_failed',
        error='the job runtime was replaced; submit the run again'
        WHERE execution_status='running'""")
    op.execute("ALTER TABLE decisions ADD COLUMN action_id TEXT")
    op.execute("""
        CREATE TABLE run_events (
            sequence BIGSERIAL PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            run_id UUID NOT NULL,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT run_events_tenant_run_fkey FOREIGN KEY (tenant_id, run_id)
                REFERENCES loop_runs(tenant_id, id) ON DELETE CASCADE
        )
    """)
    op.execute("""CREATE INDEX run_events_tenant_run_sequence_idx
        ON run_events (tenant_id, run_id, sequence)""")
    op.execute("""INSERT INTO run_events
        (tenant_id, run_id, event_type, payload, created_at)
        SELECT r.tenant_id, e.run_id, e.event_type, e.payload, e.created_at
        FROM evidence_run_events e
        JOIN loop_runs r ON r.id = e.run_id AND r.tenant_id = e.ownership->>'tenant_id'
        ORDER BY e.sequence""")
    op.execute("DROP TABLE evidence_run_events")
    op.execute("DROP TABLE evidence_jobs")


def downgrade() -> None:
    identity_check = """jsonb_typeof(ownership->'tenant_id')='string'
        AND length(ownership->>'tenant_id') > 0
        AND jsonb_typeof(ownership->'user_id')='string'
        AND length(ownership->>'user_id') > 0"""
    op.execute("""
        CREATE TABLE evidence_jobs (
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
            finished_at TIMESTAMPTZ
        )
    """)
    op.execute("""CREATE UNIQUE INDEX evidence_jobs_scope_key_idx
        ON evidence_jobs (ownership, idempotency_key)
        WHERE idempotency_key IS NOT NULL""")
    op.execute("""CREATE INDEX evidence_jobs_claim_idx
        ON evidence_jobs (status, lease_expires_at, created_at)""")
    op.execute(f"""ALTER TABLE evidence_jobs ADD CONSTRAINT evidence_jobs_identity_ck
        CHECK ({identity_check})""")
    op.execute("""
        CREATE TABLE evidence_run_events (
            sequence BIGSERIAL PRIMARY KEY,
            run_id UUID NOT NULL,
            ownership JSONB NOT NULL DEFAULT '{}'::jsonb,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""CREATE INDEX evidence_run_events_run_sequence_idx
        ON evidence_run_events (run_id, ownership, sequence)""")
    op.execute(
        f"""ALTER TABLE evidence_run_events ADD CONSTRAINT evidence_events_identity_ck
        CHECK ({identity_check})"""
    )
    op.execute("""INSERT INTO evidence_run_events
        (run_id, ownership, event_type, payload, created_at)
        SELECT e.run_id,
               jsonb_build_object('tenant_id', r.tenant_id, 'user_id', r.user_id),
               e.event_type, e.payload, e.created_at
        FROM run_events e
        JOIN loop_runs r ON r.id = e.run_id AND r.tenant_id = e.tenant_id
        ORDER BY e.sequence""")
    op.execute("DROP TABLE run_events")
    op.execute("ALTER TABLE decisions DROP COLUMN action_id")
    op.execute(
        "ALTER TABLE loop_runs ALTER COLUMN execution_status SET DEFAULT 'running'"
    )
    op.execute("DROP INDEX loop_runs_idempotency_idx")
    for column in (
        "updated_at",
        "published_at",
        "approval",
        "job_id",
        "idempotency_key",
        "request",
    ):
        op.execute(f"ALTER TABLE loop_runs DROP COLUMN {column}")
