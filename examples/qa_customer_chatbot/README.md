# QA customer chatbot

A self-contained customer-support demo. FastAPI serves a grounded Agloom ReAct
agent backed by a real OpenAI-compatible `ChatOpenAI` model, and React/Vite
provides the customer UI.

- The checked-in seed records are fictional.
- No live API credential is shipped or stored by the application.
- Data, conversations, feedback, and BehaviorWeave state live in process
  memory, so restarts clear them.

## Architecture

- **Agent.** An Agloom agent with the fixed **ReAct** topology
  (`create_agent(pattern="react")`) and recursion disabled
  (`ExecutionLimits(max_depth=0)`). The backend calls it with
  `await agent.ainvoke(...)` and passes the conversation ID as the LangGraph
  `thread_id`.
- **Tools.** `order_status` returns only the order that the backend verified
  for the current customer and question. `policy_lookup` reads an approved
  policy by ID. Both are read-only.
- **Structured drafts.** `agent.with_structured_output(Draft)` appends the
  xstructured format instructions to the latest user message and validates the
  agent's answer as a typed `Draft`. A malformed or ungrounded draft gets one
  correction round. After that the request fails with HTTP 502 rather than
  returning an unsupported answer.
- **Grounding.** Citations are validated against the current request's
  approved policies. Order fields in the HTTP response always come from the
  store, not from model output.
- **Guardrail.** The `stop-repeated-lookup` guardrail is a BehaviorWeave
  `repeated_tool_call` policy, applied by `BehaviorWeaveMiddleware` with Agloom's
  `behavior_signals` bridge. It is scoped to the conversation, so the third
  identical lookup in a conversation is not executed. The model receives the
  guardrail message instead, and the response reports
  `guardrail.status = "intervened"`.
- **Feedback.** The service owns a `FeedbackManager` (feedback-manager).
  Feedback works without a configured model.
- **Telemetry.** `backend/observability.py` creates Agloom observers at
  startup; see [Observability](#observability). On shutdown the service calls
  `agent.close()`.

## Tool policy

agentbarrier denies every tool that no rule allows. The policy
(`qa-customer-chatbot-v1`) allows exactly the two read-only tools:

| Rule | Effect | Tool |
| --- | --- | --- |
| look up the selected order | allow | `order_status` |
| read approved support policies | allow | `policy_lookup` |

The assistant cannot change orders, start returns, or issue refunds, so no
tool is declared consequential. Calls are checked statelessly against an
in-memory agentbarrier `SQLiteRuntimeStore` that never records an action.

A denied call, such as an invented `cancel_order` tool, is not executed. The
model receives an error tool message and each denial is emitted as a
`ToolCallDenied` event. `GET /api/governance` publishes the policy.

The service's only state-changing operations are customer feedback records and
reviewer resolutions. Both are API operations, not agent tools.

- **Submissions.** Feedback submissions are idempotent: each record carries a
  feedback-manager `idempotency_key` derived from the message, category, and
  content. Resubmitting the same feedback returns the stored record instead of
  creating a duplicate.
- **Resolutions.** Resolutions require the reviewer token.

## Run locally (uv, Python 3.12+, and Node.js 24+)

From this directory:

```powershell
uv sync --frozen
$env:EXPLABS_API_KEY = "<your provider key>"
$env:EXPLABS_MODEL = "gpt-5.6-luna"
$env:EXPLABS_BASE_URL = "https://api.experientiallabs.ai/v1"
uv run --frozen uvicorn backend.app:app --reload --port 8000
```

Use an OpenAI-compatible chat-completions provider by setting all three
`EXPLABS_*` variables. Without a key, `/api/health` reports `unconfigured`, and
grounded chat returns HTTP 503 rather than a pretend model answer.

To run the frontend:

1. In another terminal, run `cd frontend; npm.cmd install; npm.cmd run dev`.
2. Open the Vite URL. Vite proxies `/api` to port 8000.

To host the frontend separately:

- **Another backend origin.** Set `VITE_API_BASE_URL` in `frontend/.env.local`.
  Never put API keys there.
- **Another frontend origin.** Set `CORS_ORIGINS` on the backend to a
  comma-separated list of explicitly allowed origins.

The production frontend build is `cd frontend; npm.cmd run build`. Serve the
generated `frontend/dist` with a reverse proxy that forwards `/api` to FastAPI.

For a containerized run, set `EXPLABS_API_KEY` and use
`podman compose up --build`. The backend image is based on `python:3.14-slim`
and runs as a non-root user. The frontend image builds the UI with
`node:24-alpine` and serves it with `nginx:1.30-alpine`. The UI is served at
`http://localhost:3102` and the backend at `http://localhost:8102`.

Try customer `CUST-1024` with order `ORD-5001`, or `CUST-2048` with
`ORD-6001`. Ask about a return, shipping, or an address change. The backend
loads [customers.json](data/customers.json) and
[policies.json](data/policies.json).

## API

* `GET /api/health` returns `status`, `model_configured`, seed counts, the
  active Agloom features, and the telemetry configuration.
* `GET /api/governance` returns:
  * the evidence policy and tool scope;
  * the tool policy;
  * the guardrail policy;
  * the feedback review requirement.
* `POST /api/chat` accepts
  `{ "message": "...", "customer_id": "CUST-1024" }`. Include the returned
  `conversation_id` on later turns; the customer ID cannot change
  mid-conversation. The response contains:
  * `conversation_id` and `message_id`;
  * `answer`;
  * a verified `order`, or null;
  * policy `sources`;
  * `guardrail` (`active` or `intervened`).
* `POST /api/feedback` accepts
  `{ "message_id": "...", "rating": "up" | "down", "correction": "..." }`.
  A rating or a nonblank correction is required. A rating and a correction
  create separate feedback-manager records in `received` state, and the
  response includes `feedback_ids`. Resubmissions are deduplicated.
* `GET /api/feedback/{message_id}` returns the category and lifecycle status of
  each feedback item attached to the answer.
* `POST /api/feedback/{feedback_id}/resolve` is a human review operation:
  * It requires an `X-Review-Token` header that matches the server-only
    `REVIEW_TOKEN` environment variable.
  * The body is `{ "approved": false, "reason": "..." }`.
  * It acknowledges, handles, and resolves the item.
  * It **does not** rewrite seed policies or blindly apply customer-provided
    corrections.

Model failures and drafts that stay ungrounded return HTTP 502 and are logged. No
unsupported source is returned as if it were verified.

This demo does **not authenticate customers**: a seeded customer ID is not
proof of identity. Before connecting real customer data, add:

* real authentication and authorization;
* access controls for feedback status;
* persistent storage and retention controls;
* rate limits and appropriate abuse protection.

Do not enter passwords, payment information, or actual personal data. The
frontend only retains messages in browser memory.

## Observability

Telemetry is off by default. Set `AGLOOM_OBSERVABILITY_ENABLED=true` to turn
it on. At startup the service then:

* logs every Agloom event with structlog;
* serves Prometheus metrics on `AGLOOM_METRICS_HOST:AGLOOM_METRICS_PORT`
  (default `127.0.0.1:9465`) with the prefix `agloom_qa_customer_chatbot_`;
* traces runs in Langfuse when both `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` are set. The Langfuse SDK also reads
  `LANGFUSE_BASE_URL`.

## Tests

```powershell
$env:LANGGRAPH_STRICT_MSGPACK = "true"
uv run --frozen pytest
cd frontend
npm.cmd run build
```

The offline suite drives the real Agloom agent with a scripted chat model, so
no request reaches a model provider. It covers:

* order scoping and grounding;
* malformed and ungrounded draft correction;
* the BehaviorWeave guardrail;
* tool-policy denials;
* feedback idempotency and review;
* telemetry.

To exercise the live model, run `uv run --frozen pytest -m live`. This loads
the repository-root `.env` when present; variables already set in the
environment take precedence.
