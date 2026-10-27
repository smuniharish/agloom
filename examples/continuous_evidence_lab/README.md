# Continuous Evidence Lab

This independent `uv` application demonstrates **Agloom Loop Engineering** on a
workflow where the answer can become stale while it is being researched. It
retrieves revisioned vendor security and compliance records semantically,
inspects a live trust page, synthesizes cited claims, rejects bad citations,
replans coverage gaps, refreshes changed evidence, and holds every validated
result for a human reviewer before it is published.

## Architecture

```text
React dashboard ── OIDC (Keycloak) ── FastAPI API
                                        │  admission: run row + procrastinate job, one transaction
                                        │  SSE stream of the run event log
                                        │  reviewer decisions (agentbarrier, tenant-authorized)
                                        ▼
                         PostgreSQL ─ app tables (Alembic) · run event log
                                    ─ procrastinate job queue
                                    ─ LangGraph checkpoints (AsyncPostgresSaver)
                                    ─ agentbarrier runtime store (approvals, exactly-once)
                                        ▲
procrastinate worker(s) ── create_agent(ChatOpenAI(EXPLABS_*), pattern="react")
   ├── LoopPolicy: up to 5 iterations · observe → validate → replan/refresh → publish
   ├── EvidenceLoopObserver: deterministic citation, coverage, and quality checks
   ├── PolicyManager(PostgresRuntimeStore): allow/deny every tool; publication needs approval
   ├── MCPRuntime: Filesystem MCP (workspace/) · Everything MCP · isolated Playwright browser
   ├── RefreshEngine: revision-aware re-indexing of data/sources.json into pgvector
   ├── ContextSage summarization middleware · Prometheus/Langfuse observers
   └── Ollama all-minilm embeddings ── pgvector semantic retrieval
```

### Run lifecycle

1. `POST /api/runs` inserts the run and defers its procrastinate job in the same
   transaction and returns `202`. A repeated `Idempotency-Key` from the same
   user returns the existing run.
2. A worker job builds the run's agent. The checkpoint thread is the run ID, and
   the `checkpoint_authorizer` admits only that thread for the run's tenant.
   Before the loop starts, the worker collects MCP observations through the
   agent (`aexecute_capability`), so they are governed by the same policy.
3. Each iteration calls `get_validation_feedback`, drafts a synthesis, and
   submits it once with the `publish_evidence` tool. A publication gate
   (LangChain middleware) scores the draft deterministically: drafts with gaps,
   invalid citations, or stale sources never reach the reviewer; the model gets
   the validation report instead and the loop replans or refreshes.
4. An approval-ready draft reaches the policy rule *reviewer approves every
   evidence publication*. agentbarrier records the pending action, bound to the
   exact synthesis, and the run pauses (`ExecutionPaused`). The job stores the
   synthesis, its deterministic accuracy, and the approval, sets the run to
   `awaiting_approval`, and ends.
5. A reviewer approves or rejects in the dashboard (`POST
   /api/runs/{id}/decision`). The API decides the agentbarrier action — tenant
   and namespace are enforced inside the store transaction, optionally with
   separation of duties — records the decision, and defers a resume job in one
   transaction.
6. The resume job calls `agent.aresume(config)`. On approval, the publication
   runs exactly once: it writes the claim and evidence ledger and marks the run
   `approved`; a replayed attempt returns the stored result instead of
   publishing again. On rejection, agentbarrier fails the call closed with
   `ApprovalRejected` and the run ends `rejected`.

Runs that end without approval-ready evidence (iteration, replan, or duration
budget exhausted, or the one-shot baseline) end as `unvalidated` with their
best draft and its deterministic accuracy. The 10-minute duration budget is
checked when an iteration ends and counts active work only: time spent waiting
for a reviewer does not count, and a resumed iteration counts its own time.

| Field | Values |
|---|---|
| `status` | `running`, `awaiting_approval`, `approved`, `rejected`, `unvalidated`, `failed`, `cancelled` |
| `execution_status` | `queued`, `running`, `paused`, `completed`, `failed`, `cancelled` |
| `objective_state` | `incomplete`, `awaiting_approval`, `validated`, `failed` |

### Deterministic validation

`EvidenceLoopObserver` never trusts the model's own assessment. A citation is
valid only when its source ID exists and its quote is a case-insensitive
verbatim substring of that source revision. Coverage is the fraction of
required topics with a non-unverified claim. Quality is the weighted
combination of citation precision (45%), supported-claim ratio (30%), and
coverage (25%). A draft is approval-ready only with full citation precision,
full coverage, no violations, and quality of at least 0.8.

The loop objective is complete only when reviewer-approved evidence has been
published. Gaps and violations request a replan; changed source revisions
request a refresh first. Objective progress is topic coverage; the observer's
explicit `objective_complete` signal takes precedence, so full coverage alone
never completes the loop.

## Policies

[`backend/policy.json`](backend/policy.json) is an agentbarrier runtime policy;
point `EVIDENCE_POLICY_FILE` at your own file to customize it. The default
effect is **deny**, and the first matching rule wins:

| Tool | Effect |
|---|---|
| `publish_evidence` | requires approval (7-day window); consequential, one publication per run |
| `get_validation_feedback` | allow |
| `refresh_evidence_source` (index re-writes by RefreshEngine) | allow |
| `evidence-filesystem__read_*`, `list_*`, `directory_tree`, `search_files`, `get_file_info` | allow; every other workspace tool is denied |
| `evidence-browser__browser_navigate` with an `http://` or `https://` URL, `browser_snapshot` | allow; every other browser tool is denied |
| `evidence-everything__echo` | allow; every other reference-server tool is denied |

MCP tools are named `<server_id>__<tool>`. Tools the policy denies for every
argument are hidden from the model. A denied call fails the run closed with
termination `policy_violation`. Rules that require approval must target a tool
declared consequential — here only `publish_evidence`, keyed by tenant and run.

Set `REQUIRE_SEPARATE_APPROVER=true` so a reviewer cannot approve a run they
submitted. Pending approvals can also be decided with agentbarrier's own tools
(CLI, approval API, dashboard); the next resume job picks the decision up.

## Durable execution

* **Jobs** — procrastinate on PostgreSQL. One `evidence.run` job advances a run
  (start, resume after a decision, or recovery). Jobs of one run share a lock,
  so at most one is queued and one is running.
* **Retries** — transient failures (database, network, provider outages) are
  retried with exponential backoff up to `JOB_MAX_ATTEMPTS`; the run is marked
  `failed` only after the last attempt. Failures inside an iteration are first
  handled by the loop's own retry budget.
* **Crash recovery** — workers send heartbeats; a periodic job requeues the jobs
  of workers that stopped without a graceful shutdown (`STALLED_JOB_SECONDS`).
  The retried job continues the run from its last LangGraph checkpoint with
  `agent.aresume(config)`: completed iterations and model calls are not
  repeated, and a publication that already happened is replayed from
  agentbarrier instead of running again.
* **Cancellation** — `POST /api/runs/{id}/cancel` cancels a queued job, aborts a
  running one (the job stops its agent cooperatively with `Agent.stop()`), or
  withdraws a pending approval.
* **Failed publication** — if writing the ledger raises, agentbarrier marks the
  publication's outcome unknown and the run pauses for reconciliation: it is
  shown as `paused` with the reason, and the job is retried. The retried job
  reconciles the action against the evidence ledger, where a publication is a
  single transaction: a committed publication is replayed to the model without
  running again, and one that did not commit goes back to the reviewer, whose
  new approval publishes it exactly once. Until then the run can be cancelled,
  unless its publication committed. Operators can also reconcile the action
  with agentbarrier's own tools first.
* **Graceful shutdown** — a stopping worker waits up to
  `WORKER_SHUTDOWN_TIMEOUT` seconds, then aborts its jobs; they are retried and
  resume from their checkpoints.

Every write is tenant scoped and idempotent (timeline rows upsert by iteration,
publication is guarded per run), so a retried job cannot corrupt a run.

## Authentication and tenant isolation

All business, governance, decision, and SSE endpoints require an OIDC bearer
token; only `/api/health` and `/api/ready` are public. JWT validation requires
the configured issuer, audience, JWKS, expiration, subject, and `tenant_id`,
with an asymmetric algorithm allowlist (`RS256` by default). SQL queries, the
event log, idempotency, approvals (agentbarrier `organization_id`), and the
checkpoint authorizer are tenant scoped; cross-tenant IDs return 404.

## Run

Requirements: Docker Compose, an Experiential Labs API key, and the repository's
external `agloom-observability_default` network (start the observability stack
first, or remove that network stanza for a standalone run).

```bash
cd examples/continuous_evidence_lab
cp .env.example .env
# set EXPLABS_API_KEY, POSTGRES_PASSWORD, BROWSER_BRIDGE_TOKEN, and the Keycloak values
docker compose up --build
```

Open <http://localhost:5180>. The API is on `8007`, Prometheus metrics on
`9470` (API and worker), PostgreSQL 18 (`pgvector/pgvector:pg18`) on `55437`.
The `migrate` service runs `python -m backend.setup_database` before the API
and worker start. Scale workers with `docker compose up --scale worker=3`.
`/api/health` reports process liveness; `/api/ready` returns HTTP 503 while
PostgreSQL is unavailable. To keep a database from an earlier PostgreSQL 17
setup, follow [MIGRATIONS.md](MIGRATIONS.md#postgresql-major-versions) before
starting the stack.

Playwright MCP and its Chromium process run as a non-root user in the dedicated
`browser` container, not in the API or the credential-bearing worker. The
browser container receives only its bridge token: no database, model,
Langfuse, or application secrets, and no application volume. Its internal
control network contains only the browser and the worker; its egress network
contains only the browser and the bundled `target-site`. The worker reaches
Playwright through an authenticated, newline-framed MCP bridge that hides and
rejects the unsafe browser-code tool. Chromium traffic passes through a
loopback-only outbound proxy that checks DNS answers and the numeric peer of
every connection, including redirects and subresources; non-public
destinations are denied and `target-site` is pinned to its startup addresses on
HTTP port 80. The image pins Playwright MCP `0.0.83`, Playwright
`1.64.0-alpha-1790635538000`, and Chromium revision `1247` (Chrome for Testing
`155.0.8059.12`), and keeps Chromium's sandbox enabled: the container drops
every capability except `SYS_CHROOT`, which the sandbox needs to confine its
processes.

### Local development

```bash
uv sync --system-certs
# .env holds DATABASE_URL, OLLAMA_URL, EXPLABS_*, and the OIDC settings
uv run --env-file .env python -m backend.setup_database
uv run --env-file .env python -m backend.worker          # one or more workers
uv run --env-file .env uvicorn backend.app:app_factory --factory --reload --port 8007
cd frontend && npm ci && npm run dev
```

`WORKER_IN_API=true` runs a worker inside the API process instead; the FastAPI
lifespan then registers the MCP servers on the worker's `MCPRuntime`. Servers
are registered with `refresh=RefreshPolicy(on_register=True)` so their tools are
discovered before the first run; `register_mcp` alone discovers nothing, and
the agent would see no MCP tools. Each worker holds one stdio session per MCP
server, configured without keep-alive, so closing the worker's `MCPRuntime`
also stops the server processes.

### Configuration

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL with pgvector; every schema lives here |
| `EXPLABS_API_KEY`, `EXPLABS_BASE_URL`, `EXPLABS_MODEL` | OpenAI-compatible model (worker only) |
| `OLLAMA_URL`, `OLLAMA_EMBED_MODEL` | embeddings for retrieval (worker only) |
| `EVIDENCE_POLICY_FILE` | runtime policy (default `backend/policy.json`) |
| `REQUIRE_SEPARATE_APPROVER` | reviewers cannot decide their own runs |
| `WORKER_CONCURRENCY`, `WORKER_IN_API`, `WORKER_SHUTDOWN_TIMEOUT` | worker processes |
| `JOB_MAX_ATTEMPTS`, `JOB_RETRY_BACKOFF_SECONDS`, `STALLED_JOB_SECONDS` | retries and stalled-job recovery |
| `AGENTBARRIER_SCHEMA` | PostgreSQL schema of the agentbarrier store (default `agentbarrier`) |
| `BROWSER_BRIDGE_HOST`, `BROWSER_BRIDGE_PORT`, `BROWSER_BRIDGE_TOKEN` | isolated browser; registered only when the token is set |
| `OBSERVABILITY_ENABLED`, `METRICS_HOST`, `METRICS_PORT` | Prometheus metrics and structured event logs |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | optional Langfuse tracing |
| `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URI`, `OIDC_ALGORITHMS` | token validation |

### Observability

With `OBSERVABILITY_ENABLED=true`, each process serves Prometheus metrics under
the `agloom_continuous_evidence_lab_` prefix (executions, durations, events,
loop iterations, decisions, terminations, and policy actions by tool and
outcome) next to the application's `continuous_evidence_runs_total`,
`continuous_evidence_decisions_total`, and `continuous_evidence_quality`. The
shared observability stack scrapes `continuous-evidence-lab-api:9470` and
`continuous-evidence-lab-worker:9470` as job `continuous-evidence-lab`.
Agloom events are also logged with structlog and streamed to the dashboard.
With Langfuse keys, model, tool, and graph calls are traced with the run as the
session and the tenant and user as metadata.

## Reproducible loop-accuracy proof

This protocol compares behavior; it intentionally publishes **no fabricated
live measurements**.

1. Start the stack and keep the seeded evidence revisions.
2. Submit the same objective and topics twice: first **One-shot baseline**, then
   **Loop Engineering**. Save both run documents from `GET /api/runs/{id}`.
3. Record only server-calculated `citation_precision`, `coverage`, `quality`,
   gaps, and violations. Do not grade or edit model output by hand.
4. Request an absent topic. Confirm the timeline records `replan`, no approval
   is requested, and the run ends `unvalidated`.
5. Add a newer revision to `data/sources.json`. Confirm the publication gate
   holds the draft back, the timeline records `refresh`, and the index holds the
   new revision.
6. Add a claim with an unknown source ID or a non-verbatim quote to a draft.
   Confirm citation precision falls and the draft never reaches the reviewer.
7. Compare the two saved runs. A run is only "loop-improved" when the
   deterministic metrics are higher and the timeline shows which iteration and
   decision produced the change. Report a tie or a regression as such.
8. Approve one run and reject another; restart the stack and confirm both
   decisions, the published ledger, and the agentbarrier audit trail remain.

## Tests

```bash
uv sync --system-certs
LANGGRAPH_STRICT_MSGPACK=true uv run pytest
cd frontend && npm ci && npm run build
```

Unit and API tests are offline: a scripted chat model, procrastinate's
in-memory connector, LangGraph's in-memory checkpointer, a SQLite agentbarrier
store, and in-process FastMCP servers. Tests marked `postgres` run the
migrations, the setup command, workers, the PostgreSQL checkpointer, and the
PostgreSQL agentbarrier store. With `TEST_DATABASE_URL` (a PostgreSQL server
with pgvector whose role may create databases), they create a private database
on that server and drop it afterwards, so the configured database is never
touched; otherwise they start a throwaway `pgvector/pgvector:pg18` container
through Docker (`TEST_POSTGRES_IMAGE` overrides the image, for example with a
pinned digest). They are skipped when neither is available. Tests marked `live`
drive the real model and MCP servers and run only with `EVIDENCE_LIVE_TESTS=1`.

## Database schema and migrations

Alembic versions the application tables; the setup command also creates the
procrastinate, checkpointer, and agentbarrier schemas. See
[MIGRATIONS.md](MIGRATIONS.md).
