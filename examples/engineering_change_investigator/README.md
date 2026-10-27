# Engineering Change Investigator

A FastAPI and React application that decides whether a release is ready to
deploy. An Agloom agent correlates change records, architecture decisions,
runbooks, and the live release status page, returns a grounded go/no-go report,
and publishes the decision to a release register under an agentbarrier policy:
no-go decisions are recorded immediately, go decisions wait for a release
manager, and every decision is published exactly once.

## Architecture

```text
React UI ──/api──▶ FastAPI ──▶ Investigator
                                 ├─ evidence index (PostgreSQL + pgvector, Ollama all-minilm)
                                 ├─ MCPRuntime: filesystem · everything · Playwright (stdio)
                                 ├─ Agloom agent (ReAct) ──▶ OpenAI-compatible model
                                 └─ PolicyManager (agentbarrier) ──▶ release register
```

The application starts everything in the FastAPI lifespan (`backend/runtime.py`)
and stops it in reverse order:

| Component | Role |
|---|---|
| `create_agent(pattern="react")` | Investigates with the local `semantic_change_search` tool and the read-only filesystem MCP tools, then answers with an `InvestigationReport`. |
| `agent.with_structured_output(InvestigationReport, repair=model)` | xstructured parsing, validation, and one repair attempt. |
| `MCPRuntime` (mcp-capability-router) | Registers the three stdio MCP servers in the FastAPI lifespan with `refresh=RefreshPolicy(on_register=True)`, so their tools, resources, and prompts are discovered before the first request; `"keep_alive": false` stops the server processes when the runtime closes. Capability metadata is stored in PostgreSQL. |
| `DefaultCapabilityRouter` | Offers the model only local tools and filesystem MCP tools, ranked with Ollama embeddings, the pgvector retriever, and a term-overlap reranker. |
| `IntelligentSummarizationMiddleware` (ContextSage) | Summarizes long tool conversations above 16k tokens. |
| `BehaviorWeaveMiddleware` + `behavior_signals` | Stops repeated identical tool calls; decisions become Loop Engineering signals. State is stored in PostgreSQL. |
| `ModelCallLimitMiddleware` (LangChain) | At most eight model calls per run. |
| `PolicyManager` | agentbarrier policy for every tool call and for publishing decisions (`PostgresRuntimeStore`). |
| `RefreshEngine` | Keeps the evidence index in sync with `workspace/`; fingerprints are stored in PostgreSQL, so restarts re-index only changes. |
| `FeedbackManager` | Stores reviewer feedback on an investigation thread. |
| Observers | Structured logs, Prometheus metrics (`agloom_engineering_change_investigator_*`), and Langfuse traces when keys are configured (`backend/observability.py`). |
| `XAIRuntime` (langgraph-xai) | Explainability: run provenance stored in PostgreSQL. |

## Investigation flow

1. Retrieve the closest `ADR-*`, `RELEASE-*`, and `RUNBOOK-*` documents from
   pgvector. Only their IDs may be cited.
2. Collect MCP evidence through `Agent.aexecute_capability`, so the same policy
   governs the application's own tool calls: the workspace listing, an Everything
   echo, resource, and prompt check, and a Playwright snapshot of the release
   status page.
3. Ask the agent for the report. Evidence is passed as untrusted data. Every
   claim must cite exactly one allowed ID; an ungrounded report gets one
   correction round and is otherwise rejected with HTTP 502.
4. Publish the decision with the consequential `record_release_decision`
   capability (idempotency key: the investigation ID).

The seeded workspace yields a high-risk no-go: release 4.9 lowers the checkout
pool to 16 connections, below the 40-connection minimum of ADR-101, and the load
test and database approval are missing.

## Policies

agentbarrier evaluates the rules in order, the first match wins, and everything
else is denied (`backend/governance.py`, version `engineering-change-2026-10`):

| Effect | Tool | Condition |
|---|---|---|
| require approval | `record_release_decision` | `decision == "go"` (release manager, 24 h) |
| allow | `record_release_decision` | `decision == "no_go"` |
| deny | `engineering-playwright__*` | a `filename` argument is present |
| allow | `engineering-playwright__browser_navigate` | `url` starts with `TARGET_SITE_URL/` |
| allow | `engineering-playwright__browser_snapshot` | |
| allow | `semantic_change_search` | |
| allow | `engineering-filesystem__` read, list, search, info, and tree tools | |
| allow | `engineering-everything__echo`, the architecture resource, `simple-prompt` | |

Workspace writes, other browser actions, `get-env`, and unknown tools are denied;
tools that are denied for every argument are hidden from the model. Denied model
tool calls are returned to the model as errors (`on_denied="inform_model"`).

### Release decision states

| State | Meaning |
|---|---|
| `recorded` | Published to the release register (a repeated call replays the stored receipt). |
| `awaiting_approval` | A go decision waits for a release manager. |
| `rejected` / `expired` | The release manager rejected it, or the approval window closed. |
| `outcome_unknown` | The register write failed after it started. agentbarrier fails closed and never retries it automatically. |

An operator resolves an `outcome_unknown` action with evidence from the release
register, then publishes it with `POST /api/investigations/{id}/publication`
(the UI button "Publish after reconciliation"):

```powershell
uv run --frozen agentbarrier approvals reconcile --postgres-dsn-env DATABASE_URL `
  --outcome not_committed --resolved-by "<operator>" --reason "<evidence>" <action_id>
```

`not_committed` lets the decision be published exactly once. Use
`--outcome committed --result-json '{"recorded_at": "..."}'` when the row was
written; publishing then replays that receipt without writing again. The
endpoint answers 409 while the action is still unknown. With Compose, run the
CLI in the backend container: `docker compose exec backend agentbarrier approvals reconcile ...`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Runtime, table counts, MCP servers, telemetry. |
| `GET` | `/api/capabilities` | MCP server health and discovered capabilities. |
| `GET` | `/api/governance` | The policy rules and consequential actions. |
| `POST` | `/api/investigate` | `{"question": ...}` → investigation with report and decision state. |
| `GET` | `/api/investigations/{id}` | A stored investigation. |
| `POST` | `/api/investigations/{id}/approval` | `{"approve": true, "reviewer": "..."}` decides a pending go decision. |
| `POST` | `/api/investigations/{id}/publication` | Publishes an `outcome_unknown` decision after an operator reconciled it. |
| `POST` | `/api/investigations/{id}/feedback` | `{"rating": "up" \| "down", "comment": ...}`. |
| `POST` | `/api/refresh` | Re-index changed workspace documents. |

The API has no authentication. Put the approval endpoint behind reviewer
authentication before using this design for real releases.

## Run

Copy `.env.example` to `.env` (git-ignored) and set `EXPLABS_API_KEY` and
`POSTGRES_PASSWORD`. Credentials are read only from the environment.

### Containers

```powershell
docker compose up --build
```

Open `http://localhost:5177`. The API is on port 8004 and metrics on 9468. The
backend joins the shared `agloom-observability_default` network, so start
`examples/observability` first (or remove that network from `compose.yaml`).
The backend image installs the pinned MCP servers and runs them without
downloads (`MCP_SERVER_LAUNCHER=installed`).

The database image is `pgvector/pgvector:pg18` (PostgreSQL 18). It keeps its data
in `/var/lib/postgresql/18/docker`, so the `engineering-postgres` volume is
mounted at `/var/lib/postgresql`. A volume from an older PostgreSQL major
version is not upgraded: remove it with `docker compose down -v` (the evidence
index is rebuilt from `workspace/` at startup; investigations and approvals are
lost) or move the data with `pg_dump`/`pg_restore`.

### Local backend

Requires Python 3.12+, uv, Node.js 24, PostgreSQL with pgvector, and Ollama with
`all-minilm` (for example the `postgres` and `ollama` services of the compose
file). The MCP servers start with `npx` at pinned versions.

```powershell
uv sync --frozen
$env:DATABASE_URL = "postgresql://agloom:<password>@127.0.0.1:55434/engineering"
$env:OLLAMA_URL = "http://127.0.0.1:11436"
$env:TARGET_SITE_URL = "http://127.0.0.1:8080"   # serve ./target here
$env:BROWSER_CHANNEL = "chrome"                  # or BROWSER_EXECUTABLE
uv run --frozen uvicorn backend.app:app --host 127.0.0.1 --port 8004 --env-file .env
cd frontend; npm install; npm run dev            # http://localhost:5177
```

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | required | PostgreSQL with pgvector. |
| `EXPLABS_API_KEY`, `EXPLABS_BASE_URL`, `EXPLABS_MODEL` | key required | OpenAI-compatible model. Without a key the API starts and investigations return 503. |
| `OLLAMA_URL`, `OLLAMA_EMBED_MODEL` | `http://127.0.0.1:11434`, `all-minilm` | Embeddings (384 dimensions). |
| `TARGET_SITE_URL` | `http://target-site` | The only site the browser may open. |
| `WORKSPACE_DIR` | `./workspace` | Evidence documents (read-only MCP root). |
| `MCP_SERVER_LAUNCHER` | `npx` | `installed` runs preinstalled server executables. |
| `BROWSER_EXECUTABLE`, `BROWSER_CHANNEL`, `BROWSER_SANDBOX`, `BROWSER_OUTPUT_DIR` | | Playwright MCP browser. |
| `METRICS_ENABLED`, `METRICS_HOST`, `METRICS_PORT` | `false`, `127.0.0.1`, `9468` | Prometheus endpoint. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | | Langfuse tracing (both keys or none). |
| `INVESTIGATION_TIMEOUT_SECONDS`, `APPROVAL_TTL_SECONDS` | `240`, `86400` | Request deadline and approval window. |

## Tests

```powershell
$env:LANGGRAPH_STRICT_MSGPACK = "true"
uv run --frozen pytest
```

The default suite is offline: it composes the real runtime with in-memory
stores, a SQLite agentbarrier store, in-process FastMCP servers that mirror the
real tool names, and a scripted chat model. Set `ENGINEERING_TEST_DATABASE_URL`
to a disposable PostgreSQL database with pgvector to also run the PostgreSQL
store tests, including the production wiring end to end. The tests drop and
recreate their tables and the `agentbarrier` schema:

```powershell
docker run -d --rm --name engineering-test-db -e POSTGRES_USER=agloom `
  -e POSTGRES_PASSWORD=agloom -e POSTGRES_DB=engineering_test -p 55440:5432 `
  pgvector/pgvector:pg18
$env:ENGINEERING_TEST_DATABASE_URL = "postgresql://agloom:agloom@127.0.0.1:55440/engineering_test"
uv run --frozen pytest
```

## Persistence

| Table | Contents |
|---|---|
| `engineering_documents` | Evidence documents and embeddings. |
| `engineering_refresh_state` | RefreshEngine fingerprints. |
| `engineering_mcp_capabilities` | Discovered MCP capabilities. |
| `engineering_feedback` | Reviewer feedback. |
| `engineering_behavior_state` | BehaviorWeave pattern state. |
| `engineering_xai_records` | Explainability records. |
| `engineering_investigations` | Investigations and their decision state. |
| `engineering_release_register` | Published release decisions. |
| schema `agentbarrier` | Policy actions, approvals, and audit receipts. |
