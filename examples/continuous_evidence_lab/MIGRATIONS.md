# Database schema and migrations

The application uses one PostgreSQL database (with pgvector) for four schemas,
each with its own versioning:

| Objects | Owner | How it is versioned |
|---|---|---|
| `sources`, `loop_runs`, `claims`, `evidence`, `loop_iterations`, `decisions`, `run_events` | the application | Alembic revisions in `migrations/` |
| `procrastinate_*` tables, types, and functions | procrastinate | procrastinate's schema and SQL migrations |
| `checkpoint_*` tables | `langgraph-checkpoint-postgres` | `AsyncPostgresSaver.setup()` (idempotent, versioned) |
| `agentbarrier` schema (`AGENTBARRIER_SCHEMA`) | agentbarrier | `PostgresRuntimeStore(..., migrate=True)` (idempotent, versioned) |

## Setup command

```bash
uv run python -m backend.setup_database
```

with `DATABASE_URL` set. It runs, in order:

1. `alembic upgrade head`.
2. The procrastinate schema, applied once with procrastinate's schema manager
   when `procrastinate_jobs` does not exist yet.
3. `AsyncPostgresSaver.setup()`.
4. The agentbarrier runtime schema migration.

Every step is safe to repeat. The Compose `migrate` service runs this command
before the API and the workers start. Neither the API nor the workers run DDL:
at startup they verify Alembic revision `0004`, the procrastinate and
checkpoint tables (including the checkpointer's migration version), and the
agentbarrier schema version, and refuse to start with a message naming what is
missing.

## Alembic commands

Run from `examples/continuous_evidence_lab` with `DATABASE_URL` set:

```bash
uv run alembic upgrade head
uv run alembic current
uv run alembic history --verbose
uv run alembic revision -m "add some_column to some_table"
uv run alembic downgrade -1
```

`migrations/env.py` takes the URL from `-x db_url=...`, then from a URL set on
the Alembic config (the setup command and the tests), then from `DATABASE_URL`,
and uses the psycopg 3 driver for `postgresql://` URLs. Migrations hold a
PostgreSQL advisory lock, so concurrent runs are serialized.

## Upgrading procrastinate

procrastinate ships SQL migration scripts for schema changes and leaves
tracking them to the application. When upgrading procrastinate, list the
scripts newer than the installed version with
`uv run procrastinate schema --migrations-path` and apply them in order (`pre`
scripts before deploying the new version, `post` scripts after), for example by
adding an Alembic revision that executes them. Fresh databases get the current
schema from the setup command.

## Existing databases

A database whose schema predates Alembic matches revision `0001`. Stamp it
instead of upgrading to it, then upgrade:

```bash
uv run alembic stamp 0001
uv run alembic upgrade head
```

Revision `0003` makes tenant ownership mandatory. Rows created before tenant
isolation are assigned to `BACKFILL_TENANT_ID` and `BACKFILL_USER_ID`; the
migration aborts when such rows exist and no owner is configured.

## PostgreSQL major versions

The Compose `db` service runs `pgvector/pgvector:pg18`. PostgreSQL 18 images
keep the cluster in `/var/lib/postgresql/18/docker`, so the `postgresql_data`
volume is mounted at `/var/lib/postgresql`; with a volume at
`/var/lib/postgresql/data` the container refuses to start.

PostgreSQL 18 cannot open the PostgreSQL 17 database of earlier versions of
this example (the `postgres_data` volume). To keep that data, dump it with a
temporary PostgreSQL 17 container and restore it into the new service before
the other services start:

```bash
docker compose down        # named volumes are kept
docker run -d --name evidence-pg17 \
  -v continuous_evidence_lab_postgres_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
until docker exec evidence-pg17 pg_isready -U evidence; do sleep 1; done
docker exec evidence-pg17 pg_dump -U evidence -d evidence -Fc > evidence.dump
docker rm -f evidence-pg17
docker compose up -d --wait db
docker compose exec -T db pg_restore -U evidence -d evidence --no-owner < evidence.dump
docker compose up -d
```

The dump carries every schema: the application tables, procrastinate, the
checkpoints, and agentbarrier. The `migrate` service then finds each of them
at its current version. Volume names start with the Compose project name
(`continuous_evidence_lab` by default; `docker volume ls` lists them). Remove
the old volume once the restored data is verified.

## Revisions

- `0001_baseline_schema.py` — sources, runs, claims, evidence, iterations,
  decisions, and the job and event tables of the first job runtime.
- `0002_add_tenant_and_checkpoint_columns.py` — an additive tenant column and a
  temporary checkpoint placeholder.
- `0003_enforce_tenant_isolation.py` — ownership backfill, mandatory tenant and
  user columns, tenant lookup indexes, composite source identity, and removal
  of the checkpoint placeholder.
- `0004_procrastinate_jobs_and_approvals.py` — the submitted request,
  per-user idempotency keys, the current procrastinate job, the reviewer
  approval, and the publication time on `loop_runs`; the agentbarrier action
  on `decisions`; the tenant-scoped `run_events` log (events are copied from
  the previous table). The previous job tables are dropped; runs that were
  still executing on them are marked failed and must be submitted again.

## Tests

`tests/test_migrations.py` checks the revision graph offline. With PostgreSQL
(a private database on the `TEST_DATABASE_URL` server, or a throwaway
`pgvector/pgvector:pg18` container) it runs
upgrade and downgrade in a private schema, carries runs, requests, and events
from revision `0003` to `0004` and back, stamps and upgrades a baseline
database, and runs the setup command twice.
