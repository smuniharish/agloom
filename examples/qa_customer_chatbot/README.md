# QA customer chatbot

A self-contained customer-support demo: FastAPI serves a grounded Agloom
ReAct agent backed by a real OpenAI-compatible ChatOpenAI model; React/Vite
provides the customer UI. The checked-in seed records are fictional. No live
API credential is shipped or stored by the application. Data, conversations,
feedback, and BehaviorWeave state live in process memory; restarts clear them.

## Run locally (uv, Python 3.12, and Node.js)

From this directory:

```powershell
uv sync --frozen
$env:EXPLABS_API_KEY = "<your provider key>"
$env:EXPLABS_MODEL = "gpt-5.6-luna"
$env:EXPLABS_BASE_URL = "https://api.experientiallabs.ai/v1"
uv run --frozen uvicorn backend.app:app --reload --port 8000
```

Use an OpenAI-compatible chat-completions provider by setting all three
`EXPLABS_*` variables. A missing key leaves `/api/health` at `unconfigured`;
grounded chat returns HTTP 503 rather than a pretend model answer.
In another terminal, run `cd frontend; npm.cmd install; npm.cmd run dev` and open
the Vite URL. Vite proxies `/api` to port 8000. For another backend origin,
set `VITE_API_BASE_URL` in `frontend/.env.local` (never put API keys there).
When the frontend is on a different origin, set `CORS_ORIGINS` on the backend
to a comma-separated list of explicitly allowed origins.
The production frontend build is `cd frontend; npm.cmd run build`. Serve the
generated `frontend/dist` with a reverse proxy that forwards `/api` to FastAPI.

For a containerized run, set `EXPLABS_API_KEY` and use
`podman compose up --build`. The UI is served at `http://localhost:3102` and
the backend at `http://localhost:8102`.

Try customer `CUST-1024` with order `ORD-5001`, or `CUST-2048` with
`ORD-6001`. Ask about a return, shipping, or an address change. The backend
loads [customers.json](data/customers.json) and [policies.json](data/policies.json),
uses per-customer order lookup and policy tools, and validates citations
against the current request's approved policies. Order fields in the HTTP
response always come from the store, not from model output. The agent uses
`with_structured_output` to validate a typed draft. A BehaviorWeave
`repeated_tool_call` policy stops the third identical lookup per conversation.
Model/guardrail failures return HTTP 502 and are logged; no unsupported
source is returned as if it were verified.

## API

* `GET /api/health`: `status`, `model_configured`, seed counts, and active
  Agloom runtime features.
* `POST /api/chat`: `{ "message": "...", "customer_id": "CUST-1024" }`;
  include the returned `conversation_id` on later turns. Response:
  `conversation_id`, `message_id`, `answer`, a verified `order` (or null),
  policy `sources`, and `guardrail`. Customer ID cannot change mid-conversation.
* `POST /api/feedback`: `{ "message_id": "...", "rating": "up" | "down",
  "correction": "..." }` (rating or nonblank correction required). A rating
  and a correction create separate feedback-manager records in `received`
  state; the response includes `feedback_ids`.
* `GET /api/feedback/{message_id}`: category and lifecycle status of each
  feedback item attached to the answer.
* `POST /api/feedback/{feedback_id}/resolve`: a human review operation
  requiring `X-Review-Token` matching the server-only `REVIEW_TOKEN` environment
  variable, with `{ "approved": false, "reason": "..." }`. It acknowledges,
  handles, and resolves the item; it **does not** rewrite seed policies or
  blindly apply customer-provided corrections.

This demo does **not authenticate customers**: a seeded customer ID is not
proof of identity. Never connect real customer data without real
authentication/authorization, access controls for feedback status, persistent
storage, retention controls, rate limits, and appropriate abuse protection.
Do not enter passwords, payment information, or actual personal data. The
frontend only retains messages in browser memory.

## Focused checks

```powershell
uv run --frozen python -m pytest tests\test_app.py -q
cd frontend
npm.cmd run build
```

The complete `tests` directory also runs the shared full-surface Agloom
regression contract described in `../REAL_WORLD_APPLICATIONS.md`.
