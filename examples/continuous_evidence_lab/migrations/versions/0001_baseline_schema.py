"""Baseline schema: source/run/claim/evidence/iteration/decision tables plus
the job and event tables of the first job runtime (`evidence_jobs`,
`evidence_run_events`), which revision 0004 replaces with procrastinate and
the application event log.

For a database that already has this schema, stamp this revision instead of
upgrading to it -- see MIGRATIONS.md.

Revision ID: 0001
Revises:
Create Date: 2026-09-27

"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, uri TEXT NOT NULL,
            revision TEXT NOT NULL, body TEXT NOT NULL,
            retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            embedding vector(384) NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS loop_runs (
            id UUID PRIMARY KEY, objective TEXT NOT NULL, topics JSONB NOT NULL,
            status TEXT NOT NULL, objective_state TEXT NOT NULL,
            termination TEXT, accuracy JSONB NOT NULL, synthesis JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            execution_status TEXT NOT NULL DEFAULT 'running',
            error TEXT
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            run_id UUID REFERENCES loop_runs(id), claim_id TEXT, topic TEXT,
            text TEXT NOT NULL, status TEXT NOT NULL,
            PRIMARY KEY(run_id, claim_id)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            run_id UUID REFERENCES loop_runs(id), claim_id TEXT,
            source_id TEXT, quote TEXT NOT NULL, valid BOOLEAN NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS loop_iterations (
            run_id UUID REFERENCES loop_runs(id), iteration INTEGER,
            observation JSONB NOT NULL, decision JSONB NOT NULL,
            PRIMARY KEY(run_id, iteration)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id UUID PRIMARY KEY, run_id UUID REFERENCES loop_runs(id),
            decision TEXT NOT NULL, note TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # Job and event tables of the first job runtime; replaced in revision 0004.
    op.execute("""
        CREATE TABLE IF NOT EXISTS evidence_jobs (
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
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS evidence_jobs_scope_key_idx
        ON evidence_jobs (ownership, idempotency_key)
        WHERE idempotency_key IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS evidence_jobs_claim_idx
        ON evidence_jobs (status, lease_expires_at, created_at)
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS evidence_run_events (
            sequence BIGSERIAL PRIMARY KEY,
            run_id UUID NOT NULL,
            ownership JSONB NOT NULL DEFAULT '{}'::jsonb,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS evidence_run_events_run_sequence_idx
        ON evidence_run_events (run_id, ownership, sequence)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS evidence_run_events_run_sequence_idx")
    op.execute("DROP TABLE IF EXISTS evidence_run_events")
    op.execute("DROP INDEX IF EXISTS evidence_jobs_claim_idx")
    op.execute("DROP INDEX IF EXISTS evidence_jobs_scope_key_idx")
    op.execute("DROP TABLE IF EXISTS evidence_jobs")
    op.execute("DROP TABLE IF EXISTS decisions")
    op.execute("DROP TABLE IF EXISTS loop_iterations")
    op.execute("DROP TABLE IF EXISTS evidence")
    op.execute("DROP TABLE IF EXISTS claims")
    op.execute("DROP TABLE IF EXISTS loop_runs")
    op.execute("DROP TABLE IF EXISTS sources")
