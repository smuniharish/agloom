# Autonomous Product Studio

A Claude-Code-like, bounded software-building workflow built on **Agloom Loop
Engineering**. An agent builds a React/FastAPI product inside a private
workspace; completion is decided by objective filesystem, allowlisted command,
and browser evidence, not by the model claiming it is done. Publishing the
result is a governed, human-approved action that runs exactly once, and every
run survives worker crashes by continuing from its last checkpoint.

## Architecture

```text
React studio UI ──> FastAPI API ─────────────────────────┐
  (OIDC + PKCE)     admission, approvals, SSE, cancel    │
                                                         ▼
                    PostgreSQL 18 + pgvector: application tables · procrastinate queue
                                              LangGraph checkpoints · agentbarrier actions
                                                         ▲
procrastinate worker(s) ─────────────────────────────────┘
  └─ Agloom agent: ReAct topology + LoopPolicy (≤ 5 iterations)
       ├─ workspace tools ─────── private run workspace (projects/.runs/<project>/<run>)
       ├─ allowlisted checks ──── isolated validation-executor (no credentials)
       ├─ Playwright MCP ──────── MCPRuntime, preview URL only
       ├─ PolicyManager ───────── agentbarrier: publish_release needs approval
       └─ ValidationLoopObserver  objective evidence after every iteration

Ollama/all-minilm ── embeddings     EXPLABS_* ── ChatOpenAI     Prometheus + Langfuse ── observers
```

- **API** (`backend/app.py`): authenticates every request, creates projects
  and tasks, admits runs, streams run events, cancels runs, and records
  reviewer decisions. It does not run agents.
- **Worker** (`python -m backend.worker`): a procrastinate worker. Each job
  builds the run's agent with the PostgreSQL checkpointer, the agentbarrier
  policy store, and the process-wide Playwright `MCPRuntime`, then starts or
  continues the run. The worker registers the Playwright server once at
  startup with `refresh=RefreshPolicy(on_register=True)`; without a refresh
  policy (or an explicit `await runtime.refresh()`), mcp-capability-router
  discovers no capabilities and the agent silently gets no browser tools. The
  stdio server is configured with `"keep_alive": false`, so it stops when the
  worker closes the runtime.
- **Agent** (`backend/agent.py`): `agloom.create_agent` with the `react`
  pattern, recursion disabled (`ExecutionLimits(max_depth=0)`), a
  `LoopPolicy` (5 iterations, 30 minutes of active time, 80 tool calls, 2
  retries, 1 replan), ContextSage summarization and a model-call limit
  (16 calls per iteration) as LangChain middleware, the run's
  `PolicyManager`, and the `ValidationLoopObserver`. An iteration
  completes the objective only when every acceptance criterion passes
  **and** the run published exactly the validated files. The run proof
  counts model calls, tool calls, and execution time over every job
  attempt, before and after an approval pause.
- **Run workspace**: a run builds in `projects/.runs/<project>/<run>`, a copy
  of the project's active workspace taken when the run first starts. Retries
  and resumed runs reopen the same directory, so a run continued from its
  checkpoint sees the files its earlier tool calls wrote. Only an approved
  release changes the project's active workspace.

## Run lifecycle

```text
queued ─> running ─> awaiting_approval ─(approve)─> queued ─> running ─> completed
          │  │              ├─(reject)─> rejected                     └──> failed
          │  │              └─(cancel)─> cancelled
          │  └─(release raised)─> awaiting_reconciliation ─(reconcile)─> queued
          └─(cancel)─> cancelled
```

`POST /api/tasks/{id}/run` creates the run and defers its job in **one
database transaction** (the run row and the procrastinate job commit
together). An `Idempotency-Key` header returns the existing run; a task has at
most one active run (queued, running, or waiting for a person).

## Policies and approvals

Every tool call goes through an agentbarrier `RuntimePolicy`; anything not
allowed is denied (`backend/governance.py`):

| Tool | Effect |
| --- | --- |
| `publish_release` | **requires approval** (release review) |
| `write_workspace_file` with a path containing `.agloom` | deny: internal evidence is read-only |
| `write_workspace_file`, `inspect_workspace`, `read_workspace_file`, `read_validation_feedback` | allow |
| `run_allowlisted_check` with `frontend-build`, `frontend-test`, or `backend-test` | allow |
| `validate_preview` | allow (the tool itself accepts only `TARGET_PREVIEW_URL`) |
| `playwright__browser_navigate` to the preview URL, `playwright__browser_snapshot` | allow |
| every other Playwright MCP tool | deny (hidden from the model) |

`publish_release` is the studio's **consequential** action: it atomically
replaces the project's active workspace with the run's files and indexes them
for semantic search. It is declared in `PolicyManager(consequential=...)` with
a business idempotency key derived from the run, the content fingerprint of
its files, and the request. agentbarrier binds the approval to that exact
request and executes it **exactly once**: if a worker crashes after the
release ran, the resumed run replays the stored result instead of publishing
again. Denied calls are reported back to the model (`on_denied="inform_model"`)
so it can correct itself.

The review flow:

1. The agent calls `publish_release` once every criterion passes. A release
   gate middleware re-validates first, so reviewers never see a release with a
   failing criterion.
2. The run pauses (`ExecutionPaused` from the LangGraph interrupt) and is
   recorded as `awaiting_approval` with its pending approvals.
3. A reviewer opens the run in the UI (or calls the API):
   - `GET /api/runs/{id}/approvals` lists the approvals with their live
     agentbarrier status;
   - `POST /api/runs/{id}/approvals/{action_id}` with
     `{"decision": "approve" | "reject", "reason": "..."}` decides one.
   Decisions are authorized inside the agentbarrier transaction: reviewers can
   decide only their own tenant's studio actions, and
   `STUDIO_REQUIRE_SEPARATE_APPROVER=true` forbids approving your own runs.
4. Approving queues a resume job. The worker continues the run from its
   checkpoint, the approved release executes once, and the loop finishes.
   Rejecting ends the run as `rejected`; cancelling a paused run rejects its
   pending release.
5. If `publish_release` raises after it started (a full disk, for example),
   agentbarrier cannot tell whether the release took effect and will not run
   it again on its own. Agloom pauses the run with a reconciliation interrupt;
   the run waits as `awaiting_reconciliation`, and its release is listed with
   `"kind": "reconciliation"` and status `unknown`. An operator checks the
   project workspace and records the finding in the UI or with
   `POST /api/runs/{id}/reconciliations/{action_id}` and
   `{"outcome": "committed" | "not_committed", "reason": "...", "result": ...}`
   (the reviewer rules above apply). `committed` resumes the run and gives
   `result` (any JSON value) to the agent as the release's result;
   `not_committed` resumes it, the release needs a new approval, and then runs
   exactly once. A run that waits for reconciliation cannot be cancelled.

Operators can replace the default policy with a strict agentbarrier JSON policy
file (`STUDIO_POLICY_FILE`), for example to allow automatic releases:

```json
{
  "version": "studio-automatic-releases",
  "rules": [
    {"name": "automatic release", "effect": "allow", "tool": "publish_release"},
    {"name": "workspace", "effect": "allow", "tool": "*"}
  ]
}
```

Changing the policy version invalidates pending approvals, because each
approval is bound to the policy version it was requested under.

## Durable execution

- **Queue**: [procrastinate](https://procrastinate.readthedocs.io/) on
  PostgreSQL. One `studio.execute_run` job executes or continues a run. Its
  `lock` is the run, so a run never executes twice at the same time, and its
  `queueing_lock` prevents duplicate queued jobs.
- **Checkpoints**: `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`).
  The checkpoint thread ID is the run ID, and the agent's
  `checkpoint_authorizer` admits that thread only for the tenant that owns the
  run.
- **Crash recovery**: a worker that dies stops sending procrastinate
  heartbeats. The periodic `studio.recover_stalled_runs` task (every minute)
  requeues its jobs, and the next attempt continues the run from its last
  checkpoint with `agent.ainvoke_loop(None, config)`: completed iterations and
  tool calls are not repeated, evidence of an interrupted iteration is
  rewritten rather than duplicated, and the run workspace is reused.
- **Retries**: transient failures outside the loop (`psycopg.OperationalError`,
  connection errors, timeouts, interrupted workers) retry with exponential
  backoff for up to five attempts, each continuing from the checkpoint.
  Failures inside an iteration are handled by the loop's own retry budget.
- **Cancellation**: `POST /api/runs/{id}/cancel` cancels a queued job, or asks
  procrastinate to abort a running one. The job maps the abort to
  `agent.stop()`, which stops the agent at its next model or tool boundary, so
  a tool (including a release) is never interrupted half-way.
- **Shutdown**: on SIGTERM the worker waits `STUDIO_SHUTDOWN_GRACE_SECONDS`,
  then stops in-flight runs the same cooperative way and retries them from
  their checkpoints. Compose gives workers a 60 second stop grace period.
- **Ambiguous releases**: if the release tool raises, or a worker dies while
  publishing, agentbarrier cannot prove the outcome and marks the action
  unknown instead of running it twice. The run waits in
  `awaiting_reconciliation` until an operator reconciles the release through
  the UI or the API (step 5 above), which also resumes the run; reconciling
  with the agentbarrier CLI instead would leave the run waiting.
- **Cleanup**: when a run reaches a terminal state, its checkpoint thread and
  private workspace are deleted.

## Run

Prerequisites: Docker Compose and an Experiential Labs API key.
The backend image uses the Chromium binary bundled with its pinned Playwright
base image and installs the pinned Playwright MCP server at build time.

```powershell
Set-Location examples\autonomous_product_studio
Copy-Item .env.example .env
# Put EXPLABS_API_KEY and fresh Keycloak/demo secrets in .env; do not commit it.
docker compose up --build
```

`docker compose up` starts Keycloak, provisions the `agloom` realm, runs the
`migrate` service (`python -m backend.migrate`), then starts the API, the
worker, the isolated validation executor, and the UI. Open
<http://localhost:5178>; API docs are at <http://localhost:8005/docs>. Scale
workers with `docker compose up --scale worker=3`.

To attach the repository observability stack:

```powershell
docker compose -f ..\observability\compose.yaml up -d
docker compose up --build
```

### Without containers

```powershell
uv sync --frozen
$env:DATABASE_URL = "postgresql://agloom:<password>@127.0.0.1:55435/product_studio"
uv run python -m backend.migrate
uv run uvicorn backend.app:app --port 8005     # add --loop asyncio:SelectorEventLoop on Windows
uv run python -m backend.worker                # in a second terminal; needs VALIDATION_EXECUTOR_URL
```

Set `STUDIO_EMBEDDED_WORKER=true` to run the worker inside the API process
for single-process development. `STUDIO_BROWSER_VALIDATION=false` disables
the Playwright MCP server; set `PLAYWRIGHT_BROWSER=msedge` (or `chrome`) to
use an installed browser instead of the image's Chromium. The server writes
its console logs to `<STUDIO_PROJECTS_ROOT>/.playwright-mcp`.

## Authentication and tenant isolation

All `/api` endpoints except `/api/health` and `/api/ready` require an OIDC
access token. The backend validates the signature through the configured JWKS
endpoint and requires matching `iss`, `aud`, `exp`, `nbf` (when present),
`sub`, and `tenant_id` claims. Only asymmetric algorithms in
`OIDC_ALGORITHMS` are accepted (`RS256` by default). Projects, tasks, runs,
evidence, artifacts, run events, idempotency keys, checkpoint threads, and
policy actions are scoped to the authenticated tenant; IDs from another
tenant return 404. The browser uses Authorization Code + PKCE.

## Workflow and API

The seeded `Pulseboard` specification contains four measurable criteria.

- `GET/POST /api/projects`, `GET /api/projects/{id}`
- `POST /api/projects/{id}/tasks`
- `POST /api/tasks/{id}/run` returns `202` with the `queued` run
- `GET /api/runs/{id}`, `/api/runs/{id}/evidence`
- `GET /api/runs/{id}/events`: Server-Sent Events with live status, Agloom
  runtime events, iteration progress, approvals, and the outcome; send
  `Last-Event-ID` to resume after a reconnect
- `GET /api/runs/{id}/approvals`, `POST /api/runs/{id}/approvals/{action_id}`
- `POST /api/runs/{id}/reconciliations/{action_id}`
- `POST /api/runs/{id}/cancel`
- `GET /api/projects/{id}/artifacts/search?q=...`
- `GET /api/health`, `GET /api/ready` (HTTP 503 when PostgreSQL or the queue
  is unavailable)

Run events are stored in order in PostgreSQL, so the API and any number of
workers share one stream. The browser keeps the last run ID in per-tab
storage scoped to the signed-in user and tenant and resumes the stream after a
reload.

## Observability

`backend/observability.py` turns on Agloom's observers when
`AGLOOM_OBSERVABILITY_ENABLED=true`: structured logging, Prometheus metrics
with the `agloom_autonomous_product_studio_` prefix on
`AGLOOM_METRICS_PORT` (9469), and Langfuse tracing when both Langfuse keys are
set (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`; the server is
`LANGFUSE_BASE_URL`). Besides execution, policy
(`policy_actions_total{tool,outcome}`), and loop metrics, the API and worker
export the queue state: `agloom_autonomous_product_studio_jobs{status}` and
`agloom_autonomous_product_studio_oldest_queued_job_seconds`.

In Compose, the API and worker join the shared observability stack's external
`agloom-observability_default` network as `autonomous-product-studio-api` and
`autonomous-product-studio-worker`, where its Prometheus scrapes port 9469.

## Database schema and migrations

`python -m backend.migrate` applies the Alembic revisions (application tables
and procrastinate's queue schema), LangGraph's checkpoint tables, and
agentbarrier's policy schema. The API and worker never create schema; they
verify it at startup and fail with migration guidance. See
[MIGRATIONS.md](MIGRATIONS.md).

## Safety model

- Every read/write is resolved beneath the run workspace; absolute paths,
  traversal, and writes to the internal `.agloom/` evidence directory are
  rejected (by the workspace and by the policy).
- The agent cannot supply command arguments. It can invoke only three fixed
  `shell=False` check pipelines: `frontend-build`, `frontend-test`, and
  `backend-test`.
- Checks are sent as bounded source snapshots to a separate, uncredentialed
  `validation-executor` container. Each check runs from a disposable tmpfs copy;
  the executor receives no model, database, OIDC, or application credentials
  and is attached only to an internal validation network. Results are reused
  for identical sources.
- Frontend checks parse `package.json`, allow only React/Vite/TypeScript/Vitest
  packages and exact approved scripts, reject lifecycle hooks and URL/file
  dependencies, then copy image-pinned dependencies into the disposable
  workspace before the fixed `npm run build` or `npm run test`. Validation does
  not download dependencies or run package lifecycle scripts.
- Backend checks reject build systems, entry points, optional dependencies,
  URLs, and packages outside the FastAPI/Pydantic/httpx/pytest allowlist. Tests
  run through a pinned `uv run --no-project --with ... pytest -q` environment,
  so the generated project is never installed and its build backend is never
  executed.
- Inventory, fingerprints, and artifact indexing consider at most 500 UTF-8
  text files of at most 256 KB each, with an explicit source/config suffix
  allowlist. They exclude virtual environments, dependencies, build output,
  coverage, VCS, and Python/tool cache directories; binary and symlinked files
  are never embedded.
- Browser evidence is recorded only by trusted application code
  (`validate_preview`) for the exact `TARGET_PREVIEW_URL`; the model's own
  Playwright MCP tools are limited by policy to navigating the preview and
  reading page snapshots.
- Files, browser pages, MCP output, and prompts are treated as untrusted data.
- The API holds no model key; only workers receive `EXPLABS_API_KEY`.
  Secrets are supplied only through environment variables.

Generated Vite configuration and test/application code execute only in the
validation executor, never in the credentialed worker.

## Tests

```powershell
$env:LANGGRAPH_STRICT_MSGPACK = "true"
uv run pytest -q
```

The suite runs the real Agloom agent with LangChain's fake chat model:
admission, the approval and rejection flows, cooperative cancellation,
checkpoint continuation after a simulated crash, policy decisions, tenant
isolation, migrations, and the safety model. Tests marked `postgres` use
`TEST_DATABASE_URL` (a role allowed to create databases) or start a throwaway
`pgvector/pgvector:pg18` container through Docker (`TEST_POSTGRES_IMAGE`
overrides the image); they include a worker
process that is hard-killed mid-run and resumed by stalled-job recovery. If
your container engine runs in a VM without localhost port forwarding, set
`TEST_DOCKER_PUBLISH_IP=0.0.0.0` and `TEST_DOCKER_HOST` to the VM address.

Frontend build (Node.js 24 or newer, as declared in `package.json` `engines`):

```powershell
Set-Location frontend
npm ci
npm run build
```

The project owns its `pyproject.toml` and `uv.lock`; it does not depend on the
root environment for execution.
