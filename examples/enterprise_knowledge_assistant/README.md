# Enterprise Knowledge Assistant

A standalone example application built with FastAPI, React, and the local
Agloom package. Six **fictional, demonstration-only** enterprise documents in
`data/documents.json` cover access, security, leave, expenses, deployments, and
privacy. Replace them with reviewed, access-controlled content before any
real-world use.

The backend ranks document passages locally, supplies the best matches to an
Agloom ReAct agent, and gives the agent `search_knowledge` and `read_passage`
tools to inspect further passages. The agent uses a real OpenAI-compatible
`ChatOpenAI` model. Returned citations must refer to passages supplied to or
read by the agent; an answer without valid passage citations fails rather than
presenting unsupported text as sourced. Retrieval is lexical, not embeddings:
it requires shared terms and can miss synonyms. Treat answers as assistance,
not authoritative policy interpretation.

## Requirements

- uv, Python 3.12 (the current Agloom package supports 3.12), and Node.js 20+.
- A valid `EXPLABS_API_KEY` for the configured OpenAI-compatible endpoint.
- Run commands below **from this directory**. `uv sync` creates `.venv`,
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

Open `http://localhost:5173`. Vite proxies `/api` to the local backend.
For a production frontend build, run `npm run build` and serve `frontend/dist`
behind your own web server; set `VITE_API_BASE_URL` at build time if the API
is not hosted at `/api`. Never put model credentials in Vite environment
variables: they are exposed to browsers. No key, conversation, or prompt is
written to disk by this example. Do not commit `.env` files.

## API

- `GET /api/health`: document count and model configuration state; reports
  `unconfigured` until a key is provided, plus active Agloom runtime features
  (no secret value is returned).
- `GET /api/documents`: indexed document catalog.
- `POST /api/chat`: `{"question":"How do I report a lost device?"}` returns
  `conversation_id`, `answer`, and passage metadata/excerpts under `citations`.
  Add `"conversation_id":"<returned UUID>"` for subsequent turns.
- `GET /api/conversations/{conversation_id}`: recent user/assistant turns.
  Unknown IDs return 404. Empty/invalid questions return 422. Missing
  credentials return 503; model/citation failures return 502.

Conversation history is process-local, limited to 12 turns per conversation
and 100 conversations per process, and is lost on restart. This is a demo,
not an authenticated multi-tenant service. Add identity, authorization,
per-document access policies, rate limits, audit retention, and persistent
conversation storage before connecting sensitive company data.

## Tests

```powershell
uv run --frozen python -m pytest -q tests
cd frontend
npm run build
```

The backend tests cover retrieval ranking, citation validation, conversation
history, missing configuration, actual Agloom agent construction, and tool
wiring without sending requests to a model provider. To exercise a live answer,
configure the key and ask a question in the UI or through `/api/chat`.
They also run the shared full-surface Agloom regression contract described in
`../REAL_WORLD_APPLICATIONS.md`.

## Containers

With Podman Desktop or Docker available, set `EXPLABS_API_KEY` and run:

```powershell
podman compose up --build
```

The UI is served at `http://localhost:3101` and proxies API requests to the
backend container. Credentials remain backend-only.
