"""Baseline schema: product/task/run/evidence/artifact tables plus the job
and event tables (`studio_jobs`, `studio_run_events`) of the job runtime the
studio used at the time; revision 0004 replaces the job table with
procrastinate.

This revision captures the schema that application startup code previously
created with `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` on
every boot. For a database that already has this schema (an existing volume),
stamp this revision instead of upgrading to it -- see MIGRATIONS.md.

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
        CREATE TABLE IF NOT EXISTS studio_projects (
            id UUID PRIMARY KEY, name TEXT NOT NULL, objective TEXT NOT NULL,
            acceptance_criteria JSONB NOT NULL, embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS studio_tasks (
            id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES studio_projects(id),
            title TEXT NOT NULL, objective TEXT NOT NULL, status TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS studio_loop_runs (
            id UUID PRIMARY KEY, task_id UUID NOT NULL REFERENCES studio_tasks(id),
            status TEXT NOT NULL, proof JSONB NOT NULL, result JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS studio_evidence (
            id UUID PRIMARY KEY, run_id UUID NOT NULL REFERENCES studio_loop_runs(id),
            iteration INTEGER NOT NULL, criterion_id TEXT NOT NULL, source TEXT NOT NULL,
            passed BOOLEAN NOT NULL, summary TEXT NOT NULL, details JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS studio_artifacts (
            id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES studio_projects(id),
            path TEXT NOT NULL, kind TEXT NOT NULL, sha256 TEXT NOT NULL,
            metadata JSONB NOT NULL, embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS studio_artifact_embedding_idx
        ON studio_artifacts USING hnsw (embedding vector_cosine_ops)
    """)

    # Durable job/event tables for namespace "studio"
    # (examples/durable_postgres.py: PostgresDurableRuntime(namespace="studio")).
    op.execute("""
        CREATE TABLE IF NOT EXISTS studio_jobs (
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
        CREATE UNIQUE INDEX IF NOT EXISTS studio_jobs_scope_key_idx
        ON studio_jobs (ownership, idempotency_key)
        WHERE idempotency_key IS NOT NULL
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS studio_jobs_claim_idx
        ON studio_jobs (status, lease_expires_at, created_at)
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS studio_run_events (
            sequence BIGSERIAL PRIMARY KEY,
            run_id UUID NOT NULL,
            ownership JSONB NOT NULL DEFAULT '{}'::jsonb,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS studio_run_events_run_sequence_idx
        ON studio_run_events (run_id, ownership, sequence)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS studio_run_events_run_sequence_idx")
    op.execute("DROP TABLE IF EXISTS studio_run_events")
    op.execute("DROP INDEX IF EXISTS studio_jobs_claim_idx")
    op.execute("DROP INDEX IF EXISTS studio_jobs_scope_key_idx")
    op.execute("DROP TABLE IF EXISTS studio_jobs")
    op.execute("DROP INDEX IF EXISTS studio_artifact_embedding_idx")
    op.execute("DROP TABLE IF EXISTS studio_artifacts")
    op.execute("DROP TABLE IF EXISTS studio_evidence")
    op.execute("DROP TABLE IF EXISTS studio_loop_runs")
    op.execute("DROP TABLE IF EXISTS studio_tasks")
    op.execute("DROP TABLE IF EXISTS studio_projects")
