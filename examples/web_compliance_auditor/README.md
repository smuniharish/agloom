# Web Compliance Auditor

A FastAPI and React application that audits a public URL against approved web
policies. An Agloom agent inspects the page through the Playwright MCP server,
returns a structured audit whose findings may cite only retrieved policy IDs,
and publishes the verdict to a compliance register under an agentbarrier
policy: failing and inconclusive audits are recorded immediately, a passing
audit (a compliance certification) waits for a compliance officer, and every
audit is recorded exactly once.

This is a demo, not legal advice. The policies in `data/policies.json` are
fictional.

## Architecture

```text
React UI ──/api──▶ FastAPI ──▶ Auditor
                                 ├─ policy index (PostgreSQL + pgvector, Ollama all-minilm)
                                 ├─ MCPRuntime: Playwright · filesystem · everything (stdio)
                                 ├─ Agloom agent (ReAct) ──▶ OpenAI-compatible model
                                 └─ PolicyManager (agentbarrier) ──▶ compliance register
```

The application starts everything in the FastAPI lifespan (`backend/runtime.py`)
and stops it in reverse order:

| Component | Role |
|---|---|
| `create_agent(pattern="react")` | Writes the audit; it may search the policies with the local `search_policies` tool. |
| `agent.with_structured_output(AuditReport, repair=model)` | xstructured parsing, validation, and one repair attempt. |
| `MCPRuntime` (mcp-capability-router) | Registers the Playwright, filesystem, and Everything stdio servers in the FastAPI lifespan with `refresh=RefreshPolicy(on_register=True)`, so their tools, resources, and prompts are discovered before the first request; `"keep_alive": false` stops the server processes when the runtime closes. |
| `DefaultCapabilityRouter` with a local-tools filter | Audited pages are untrusted, so the model gets no browsing or file tools; the application collects that evidence itself. |
| `IntelligentSummarizationMiddleware` (ContextSage) | Summarizes conversations above 100k tokens. |
| `BehaviorWeaveMiddleware` + `behavior_signals` | Stops repeated identical policy searches; decisions become Loop Engineering signals. |
| `PolicyManager` | agentbarrier policy for every tool call and for recording audits (`PostgresRuntimeStore`). |
| `RefreshEngine` | Keeps the policy index in sync with `data/policies.json`; edits and removals are applied incrementally, and fingerprints are stored in PostgreSQL. |
| `FeedbackManager` | Stores reviewer ratings and corrections in PostgreSQL. |
| Observers | Structured logs, Prometheus metrics under `agloom_web_compliance_auditor_*` (including `agloom_web_compliance_auditor_audits_total{verdict}`), and Langfuse traces when keys are configured (`backend/observability.py`). |

## Audit flow

1. Retrieve the closest policies from pgvector (optionally limited by
   `policy_ids`). Findings may cite only these IDs.
2. Collect MCP evidence through `Agent.aexecute_capability`, so the same policy
   governs the application's own tool calls: a filesystem check, an Everything
   echo, resource, and prompt check, and a Playwright navigation and accessibility
   snapshot of the URL. An unreachable page is reported as unavailable, and the
   model is told to answer `needs_review`.
3. Ask the agent for the audit. The page snapshot is passed as untrusted data.
   An ungrounded report gets one correction round and is otherwise rejected with
   HTTP 502.
4. Record the verdict with the consequential `record_audit` capability
   (idempotency key: the audit ID).

The bundled `target/` page collects an email address over HTTP without a privacy
notice, so it fails the transport and privacy policies.

## Policies

agentbarrier evaluates the rules in order, the first match wins, and everything
else is denied (`backend/governance.py`, version `web-compliance-2026-10`):

| Effect | Tool | Condition |
|---|---|---|
| require approval | `record_audit` | `verdict == "pass"` (compliance officer, 24 h) |
| allow | `record_audit` | `verdict` is `fail` or `needs_review` |
| deny | `web-compliance-playwright__*` | a `filename` argument is present |
| deny | `web-compliance-playwright__browser_navigate` | `url` starts with `http://169.254.169.254` |
| allow | `web-compliance-playwright__browser_navigate` | `url` starts with `http://` or `https://` |
| allow | `web-compliance-playwright__browser_snapshot` | |
| allow | `search_policies` | |
| allow | `web-compliance-filesystem__list_allowed_directories` | |
| allow | `web-compliance-everything__echo`, the architecture resource, `simple-prompt` | |

Page interactions (`browser_click`, `browser_type`, `browser_evaluate`,
`browser_run_code_unsafe`, ...), file writes, `get-env`, and unknown tools are
denied; a URL that the policy refuses is answered with HTTP 403.

### Compliance record states

| State | Meaning |
|---|---|
| `recorded` | Written to the compliance register (a repeated call replays the stored receipt). |
| `awaiting_approval` | A passing audit waits for a compliance officer. |
| `rejected` / `expired` | The officer rejected the certification, or the approval window closed. |
| `outcome_unknown` | The register write failed after it started. agentbarrier fails closed and never retries it automatically. |

An operator resolves an `outcome_unknown` action with evidence from the
compliance register, then records it with `POST /api/audits/{id}/record` (the
UI button "Record after reconciliation"):

```powershell
uv run --frozen agentbarrier approvals reconcile --postgres-dsn-env DATABASE_URL `
  --outcome not_committed --resolved-by "<operator>" --reason "<evidence>" <action_id>
```

`not_committed` lets the audit be recorded exactly once. Use
`--outcome committed --result-json '{"recorded_at": "..."}'` when the row was
written; recording then replays that receipt without writing again. The
endpoint answers 409 while the action is still unknown. With Compose, run the
CLI in the backend container: `docker compose exec backend agentbarrier approvals reconcile ...`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Runtime, table counts, MCP servers, telemetry. |
| `GET` | `/api/capabilities` | MCP server health and discovered capabilities. |
| `GET` | `/api/governance` | The policy rules and consequential actions. |
| `POST` | `/api/audits` | `{"url": ..., "policy_ids": [...]}` → audit with report and record state. |
| `GET` | `/api/audits/{id}` | A stored audit. |
| `POST` | `/api/audits/{id}/approval` | `{"approve": true, "reviewer": "..."}` decides a pending passing audit. |
| `POST` | `/api/audits/{id}/record` | Records an `outcome_unknown` audit after an operator reconciled it. |
| `POST` | `/api/audits/{id}/feedback` | `{"rating": "up" \| "down", "correction": ...}`. |
| `POST` | `/api/policies/refresh` | Re-index changed policies. |

The API has no authentication or rate limiting. Put the approval endpoint behind
reviewer authentication, and restrict which URLs may be audited, before using
this design beyond a demo.

## Run

Copy `.env.example` to `.env` (git-ignored) and set `EXPLABS_API_KEY` and
`POSTGRES_PASSWORD`. Credentials are read only from the environment.

### Containers

```powershell
docker compose up --build
```

Open `http://localhost:5176` and audit `http://target-site`. The API is on port
8003 and metrics on 9467. The backend joins the shared
`agloom-observability_default` network, so start `examples/observability` first
(or remove that network from `compose.yaml`). The database image is
`pgvector/pgvector:pg18` (PostgreSQL 18). It keeps its data in
`/var/lib/postgresql/18/docker`, so the `postgres_data` volume is mounted at
`/var/lib/postgresql`. A volume from an older PostgreSQL major version is not
upgraded: remove it with `docker compose down -v` (the policy index is rebuilt
from `data/policies.json` at startup; audits are lost) or move the data with
`pg_dump`/`pg_restore`.

### Local backend

Requires Python 3.12+, uv, Node.js 24, PostgreSQL with pgvector, and Ollama with
`all-minilm` (for example the `db` and `ollama` services of the compose file).
The MCP servers start with `npx` at pinned versions.

```powershell
uv sync --frozen
$env:DATABASE_URL = "postgresql://compliance:<password>@127.0.0.1:55433/compliance"
$env:OLLAMA_URL = "http://127.0.0.1:11435"
$env:BROWSER_CHANNEL = "chrome"                  # or BROWSER_EXECUTABLE
uv run --frozen uvicorn backend.app:app --host 127.0.0.1 --port 8003 --env-file .env
cd frontend; npm install; npm run dev            # http://localhost:5176
```

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | required | PostgreSQL with pgvector. |
| `EXPLABS_API_KEY`, `EXPLABS_BASE_URL`, `EXPLABS_MODEL` | key required | OpenAI-compatible model. Without a key the API starts and audits return 503. |
| `OLLAMA_URL`, `OLLAMA_EMBED_MODEL` | `http://127.0.0.1:11434`, `all-minilm` | Embeddings (384 dimensions). |
| `POLICIES_FILE` | `./data/policies.json` | Approved policies. |
| `WORKSPACE_DIR` | `./workspace` | Filesystem MCP root. |
| `MCP_SERVER_LAUNCHER` | `npx` | `installed` runs preinstalled server executables. |
| `BROWSER_EXECUTABLE`, `BROWSER_CHANNEL`, `BROWSER_SANDBOX`, `BROWSER_OUTPUT_DIR` | | Playwright MCP browser. |
| `METRICS_ENABLED`, `METRICS_HOST`, `METRICS_PORT` | `false`, `127.0.0.1`, `9467` | Prometheus endpoint. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | | Langfuse tracing (both keys or none). |
| `AUDIT_TIMEOUT_SECONDS`, `APPROVAL_TTL_SECONDS` | `240`, `86400` | Request deadline and approval window. |

## Tests

```powershell
$env:LANGGRAPH_STRICT_MSGPACK = "true"
uv run --frozen pytest
```

The default suite is offline: it composes the real runtime with in-memory
stores, a SQLite agentbarrier store, in-process FastMCP servers that mirror the
real tool names, and a scripted chat model. Set `COMPLIANCE_TEST_DATABASE_URL`
to a disposable PostgreSQL database with pgvector to also run the PostgreSQL
tests, including the production wiring end to end. The tests drop and recreate
their tables and the `agentbarrier` schema:

```powershell
docker run -d --rm --name compliance-test-db -e POSTGRES_USER=agloom `
  -e POSTGRES_PASSWORD=agloom -e POSTGRES_DB=compliance_test -p 55441:5432 `
  pgvector/pgvector:pg18
$env:COMPLIANCE_TEST_DATABASE_URL = "postgresql://agloom:agloom@127.0.0.1:55441/compliance_test"
uv run --frozen pytest
```

## Persistence

| Table | Contents |
|---|---|
| `compliance_policies` | Approved policies and embeddings. |
| `compliance_refresh_state` | RefreshEngine fingerprints. |
| `compliance_audits` | Audits and their record state. |
| `compliance_register` | Recorded audit verdicts. |
| `compliance_feedback` | Reviewer feedback. |
| schema `agentbarrier` | Policy actions, approvals, and audit receipts. |
