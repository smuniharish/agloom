# Enterprise Knowledge Assistant

A standalone example application built with FastAPI, React, and the local
Agloom package. Six **fictional, demonstration-only** enterprise documents in
`data/documents.json` cover access, security, leave, expenses, deployments, and
privacy. Replace them with reviewed, access-controlled content before any
real-world use.

## Architecture

- **Retrieval.** The backend ranks document passages locally and puts the best
  matches into the question it sends to the agent. Retrieval is lexical, not
  embeddings: it requires shared terms and can miss synonyms.
- **Agent.** An Agloom agent with the fixed **ReAct** topology
  (`create_agent(pattern="react")`) runs on a real OpenAI-compatible
  `ChatOpenAI` model. Recursive child runs are disabled with
  `ExecutionLimits(max_depth=0)`. The agent is called with `await agent.ainvoke(...)`.
- **Tools.** `search_knowledge` and `read_passage` let the agent inspect
  further passages. Both only read the bundled knowledge base.
- **Tool policy.** An Agloom `PolicyManager` (agentbarrier) governs every tool
  call; see [Tool policy](#tool-policy).
- **Citations.** Returned citations must refer to passages supplied to or read
  by the agent during the request. An answer without valid passage citations
  fails rather than presenting unsupported text as sourced. Treat answers as
  assistance, not authoritative policy interpretation.
- **Telemetry.** `backend/observability.py` creates Agloom observers from the
  environment at startup; see [Observability](#observability). On shutdown the
  service calls `agent.close()`, which flushes telemetry and stops the
  observers.

## Tool policy

agentbarrier denies every tool that no rule allows. The policy
(`enterprise-knowledge-assistant-v1`) allows exactly the two read-only tools:

| Rule | Effect | Tool |
| --- | --- | --- |
| search approved knowledge | allow | `search_knowledge` |
| read approved passages | allow | `read_passage` |

Neither tool changes external state, so no tool is declared consequential.
Calls are checked statelessly, and the agentbarrier `SQLiteRuntimeStore` is an
in-memory database that never records an action.

A denied call, such as a tool name the model invents, is not executed. The
model receives an error tool message (`on_denied="inform_model"`) and can
answer with the allowed tools. Each denial is emitted as a `ToolCallDenied`
event. `GET /api/governance` publishes the active policy.

## Requirements

- uv, Python 3.12 or newer, and Node.js 24 or newer.
- A valid `EXPLABS_API_KEY` for the configured OpenAI-compatible endpoint.
- Run the commands below **from this directory**. `uv sync` creates `.venv`,
  installs the locked dependencies, and installs the local Agloom source from
  `../..` in editable mode.

```powershell
uv sync --frozen
$env:EXPLABS_API_KEY = '<your key>'
$env:EXPLABS_BASE_URL = 'https://api.experientiallabs.ai/v1' # optional default
$env:EXPLABS_MODEL = 'gpt-5.6-luna' # optional default
uv run --frozen uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`; Vite proxies `/api` to the local backend.

For a production frontend build, run `npm run build` and serve `frontend/dist`
behind your own web server. Set `VITE_API_BASE_URL` at build time if the API is
not hosted at `/api`.

Never put model credentials in Vite environment variables: they are exposed to
browsers. This example writes no key, conversation, or prompt to disk. Do not
commit `.env` files.

## API

- `GET /api/health` returns:
  - the document count;
  - whether a model is configured (`unconfigured` until a key is provided);
  - the active Agloom features;
  - the telemetry configuration.

  No secret value is returned.
- `GET /api/governance` returns the evidence and citation rules, the tool
  scope, and the active tool policy.
- `GET /api/documents` returns the indexed document catalog.
- `POST /api/chat` accepts `{"question":"How do I report a lost device?"}` and
  returns `conversation_id`, `answer`, and passage metadata and excerpts under
  `citations`. Add `"conversation_id":"<returned UUID>"` for later turns.
- `GET /api/conversations/{conversation_id}` returns recent user and assistant
  turns.

Status codes:

| Status | Cause |
| --- | --- |
| 404 | Unknown conversation ID |
| 422 | Empty or invalid question |
| 503 | Missing credentials |
| 502 | Agent failure or citation failure |

Conversation history is process-local and lost on restart. It is limited to 12
turns per conversation and 100 conversations per process.

This is a demo, not an authenticated multi-tenant service. Before connecting
sensitive company data, add:

- identity;
- authorization and per-document access policies;
- rate limits;
- audit retention;
- persistent conversation storage.

## Observability

Telemetry is off by default. Set `AGLOOM_OBSERVABILITY_ENABLED=true` to turn
it on. At startup the service then:

- logs every Agloom event with structlog;
- serves Prometheus metrics on `AGLOOM_METRICS_HOST:AGLOOM_METRICS_PORT`
  (default `127.0.0.1:9464`) with the prefix
  `agloom_enterprise_knowledge_assistant_`;
- traces runs in Langfuse when both `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` are set. The Langfuse SDK also reads
  `LANGFUSE_BASE_URL`. Events carry `application_id` and `tenant_id` metadata.

## Tests

```powershell
$env:LANGGRAPH_STRICT_MSGPACK = 'true'
uv run --frozen pytest
cd frontend
npm run build
```

The offline suite covers:

- retrieval ranking;
- citation validation and conversation history;
- missing configuration;
- the tool policy and its denials;
- telemetry;
- the real Agloom agent, driven by a scripted chat model. The scripted model
  reads a passage through the policy-governed tools and cites it, so no request
  is sent to a model provider.

To exercise live answers against the configured model, run:

```powershell
uv run --frozen pytest -m live
```

This loads the repository-root `.env` when present; variables already set in
the environment take precedence.

## Containers

With Podman Desktop or Docker available, set `EXPLABS_API_KEY` and run:

```powershell
podman compose up --build
```

The backend image is based on `python:3.14-slim`, installs the locked
dependencies with uv, and runs as a non-root user. The frontend image builds
the UI with `node:24-alpine` and serves it with `nginx:1.30-alpine`. The UI is
served at `http://localhost:3101` and proxies API requests to the backend
container. Credentials remain backend-only.
