# Database schema and migrations

The Product Studio stores four kinds of state in one PostgreSQL database. One
command creates or upgrades all of them:

```bash
uv run python -m backend.migrate      # needs DATABASE_URL
```

| State | Managed by | Applied by `backend.migrate` as |
| --- | --- | --- |
| Projects, tasks, runs, evidence, artifacts, run events | Alembic (`migrations/`) | `alembic upgrade head` |
| procrastinate job queue (`procrastinate_*` tables, types, functions) | Alembic revision `0004` | part of `alembic upgrade head` |
| LangGraph checkpoints (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`) | `langgraph-checkpoint-postgres` | `PostgresSaver.setup()` (idempotent, versioned by LangGraph) |
| agentbarrier actions, approvals, and audit receipts (schema `agentbarrier`, or `STUDIO_POLICY_SCHEMA`) | agentbarrier | `PostgresRuntimeStore(..., create_schema=True, migrate=True)` (idempotent, versioned by agentbarrier) |

Compose runs the command in the `migrate` service before the API and the
worker start.

## Startup behavior

The API and the worker never create or alter schema. At startup they verify
that Alembic is at revision `0005`, that every application, queue, and
checkpoint table and column they use exists, and that the agentbarrier schema
is current. Otherwise they fail with a message that names what is missing and
asks you to run `python -m backend.migrate`.

## Commands

Run these from `examples/autonomous_product_studio` with `DATABASE_URL` set:

```bash
uv run alembic upgrade head            # application and queue schema only
uv run alembic current
uv run alembic history --verbose
uv run alembic revision -m "add some_column to some_table"
uv run alembic downgrade -1
```

`migrations/env.py` takes the connection string from `-x db_url=...`, the
`database_url` attribute set by `backend.migrate`, the `sqlalchemy.url`
option, or `DATABASE_URL`, in that order, and uses the psycopg 3 driver.

## procrastinate's schema

Revision `0004` installs the schema of the procrastinate version pinned in
`uv.lock` (`procrastinate.schema.SchemaManager.get_schema()`), so the queue is
versioned with the rest of the application schema. When you upgrade
procrastinate, add an Alembic revision that executes the new version's
migration scripts, in order, from `procrastinate/sql/migrations/` (procrastinate
ships pre- and post-migration scripts per release; their directory is printed by
`procrastinate schema --migrations-path`). Apply pre-migrations before
deploying the new workers and post-migrations after all workers run the new
version.

## Revisions

- `0001_baseline_schema.py`: the original tables (`studio_projects`,
  `studio_tasks`, `studio_loop_runs`, `studio_evidence`, `studio_artifacts`)
  plus the job and event tables of the job runtime used at the time
  (`studio_jobs`, `studio_run_events`).
- `0002_add_tenant_and_checkpoint_columns.py`: nullable tenant columns and a
  temporary checkpoint placeholder.
- `0003_enforce_tenant_isolation.py`: explicit ownership backfill,
  mandatory tenant/user columns, tenant lookup indexes, ownership checks, one
  active run per task, and removal of the checkpoint placeholder. Upgrading a
  database with pre-`0003` rows requires `STUDIO_BACKFILL_TENANT_ID` and
  `STUDIO_BACKFILL_USER_ID` (the owner of rows created before tenant
  ownership); the migration aborts rather than assigning ownership
  silently.
- `0004_procrastinate_jobs_and_approvals.py`: installs procrastinate's queue;
  drops `studio_jobs` (runs that were still running from it are marked failed
  with `interrupted by the job queue migration`, and their API idempotency keys
  move to the runs); adds the `queued`, `awaiting_approval`, `cancelled`, and
  `rejected` run states with check constraints, the pending `approvals`, the
  procrastinate `job_id`, and the `idempotency_key` to runs; widens the
  one-active-run-per-task invariant to queued, running, and paused runs; and
  makes evidence unique per run, iteration, and criterion. `studio_run_events`
  remains the application's ordered event log. Downgrading `0004` drops the
  queue and restores the previous tables.
- `0005_release_reconciliation.py`: adds the `awaiting_reconciliation` run
  and task state (a release whose tool raised has an unknown outcome until an
  operator reconciles it) to the status check constraints and to the
  one-active-run-per-task index. Downgrading `0005` fails runs that are still
  waiting for reconciliation.

## Existing volumes created before Alembic

A database whose schema was created by the original startup code matches
revision `0001`. Mark it instead of re-running the baseline DDL, then upgrade:

```bash
uv run alembic stamp 0001
uv run python -m backend.migrate
```

## PostgreSQL 18 data volume

Compose runs `pgvector/pgvector:pg18`. Its data directory is
`/var/lib/postgresql/18/docker`, so the `studio-postgres` volume is mounted at
`/var/lib/postgresql` (the image refuses to start with a volume at
`/var/lib/postgresql/data`). A volume written by an older PostgreSQL major
version cannot be opened by PostgreSQL 18: start with a fresh volume
(`docker compose down -v`), or dump the old database with `pg_dumpall` before
switching and restore the dump into the new server, then run
`python -m backend.migrate`.

## Tests

`tests/test_migrations.py` checks the revision graph offline. With PostgreSQL
(`TEST_DATABASE_URL` or a throwaway Docker container), it upgrades a fresh
database to `head` and back to `base`, round-trips `0004`, stamps a baseline
schema, verifies that runs in flight in the removed job table are failed
by `0004`, and that `0005` counts runs waiting for reconciliation as active
(and fails them when downgraded).
