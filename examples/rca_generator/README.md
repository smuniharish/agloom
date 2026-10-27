# RCA Generator

An incident-scoped root cause analysis example with a FastAPI API, an Agloom
agent, and a React UI. Two synthetic but operationally realistic seeded
incidents include incident events, timestamped logs, and metric readings. No
credentials or reports are persisted.

## Architecture

- **Evidence.** The backend gathers a bounded incident, log, and metric bundle
  with `agent.aexecute_capability(...)`. The bundle passes through the same tool
  policy as the model's own tool calls.
- **Pipeline.** Agloom's fixed **pipeline** topology
  (`create_agent(pattern="pipeline", stages=...)`) runs three stages:
  `hypothesize`, `verify`, and `report`. Recursion is disabled with
  `ExecutionLimits(max_depth=0)`. Every stage can call the scoped evidence
  tools `get_incident`, `get_logs`, and `get_metrics`.
- **Structured report.** `agent.with_structured_output(RcaReport)` appends the
  xstructured format instructions to the analysis request and validates the
  pipeline's final answer as a typed `RcaReport`.
- **Validation.** The server enforces that every evidence ID refers to the
  selected incident. It populates authoritative citation text and timestamps,
  and rejects fabricated citation metadata. A report that fails these checks
  gets one correction round; after that the request fails with HTTP 502.

  Confidence is a model judgment, not a calibrated probability, and human
  review remains essential. Citation-ID validation cannot prove that every
  natural-language inference is correct.
- **Telemetry.** `backend/observability.py` creates Agloom observers at
  startup; see [Observability](#observability). On shutdown the service calls
  `agent.close()`.

## Tool policy

agentbarrier denies every tool that no rule allows. The policy
(`rca-generator-v1`) allows the three read-only evidence tools only for seeded
incident IDs:

| Rule | Effect | Tool | Condition |
| --- | --- | --- | --- |
| read seeded incident events | allow | `get_incident` | `incident_id` in the seeded incidents |
| read seeded incident logs | allow | `get_logs` | `incident_id` in the seeded incidents |
| read seeded incident metrics | allow | `get_metrics` | `incident_id` in the seeded incidents |

The service only reads incident evidence and returns a report for human
review; nothing is published or written. So no tool is declared
consequential, and calls are checked statelessly against an in-memory
agentbarrier `SQLiteRuntimeStore` that never records an action.

A denied call is not executed. This covers an unknown incident ID or a tool
the model invents.

- **Model-initiated calls.** The model receives an error tool message and can
  continue with the supplied evidence.
- **Direct capability calls.** These raise `PolicyViolationError`.

Each denial is emitted as a `ToolCallDenied` event. `GET /api/governance`
publishes the active policy.

## Run locally

Requires:

- uv;
- Python 3.12 or newer;
- Node.js 24 or newer;
- an OpenAI-compatible chat endpoint;
- a model that can follow the structured-output envelope instructions.

From this directory:

```powershell
uv sync --frozen
$env:EXPLABS_API_KEY = "your-token"
# Optional: $env:EXPLABS_MODEL = "gpt-5.6-luna"
# Optional: $env:EXPLABS_BASE_URL = "https://api.experientiallabs.ai/v1"
uv run --frozen uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. The frontend proxies `/api` to `127.0.0.1:8000`.

For a separately hosted frontend:

- set `VITE_API_BASE_URL` to your trusted API prefix at build time;
- configure CORS or a reverse proxy appropriately. The backend does not enable
  cross-origin access by default.

Never put a model token in `VITE_` variables or commit a populated `.env`.

## API

- `GET /api/health` returns `status` (`ok` or `unconfigured`), the incident
  count, `model_configured`, the active Agloom features, and the telemetry
  configuration. It does not contact the model provider.
- `GET /api/governance` returns:
  - the evidence policy and tool scope;
  - the tool policy;
  - the structured schema and its confidence bounds;
  - the human review requirement.
- `GET /api/incidents` lists public incident metadata without raw records.
- `POST /api/analyze` with `{"incident_id":"INC-2026-041"}` returns a typed
  `RcaReport` containing:
  - the root cause and the overall confidence, each in `[0,1]`;
  - a timestamped timeline;
  - recommendations;
  - evidence with exact citation IDs.

  An unknown incident returns 404, missing configuration returns 503, and a
  model or validation failure returns 502.

## Observability

Telemetry is off by default. Set `AGLOOM_OBSERVABILITY_ENABLED=true` to turn
it on. At startup the service then:

- logs every Agloom event with structlog;
- serves Prometheus metrics on `AGLOOM_METRICS_HOST:AGLOOM_METRICS_PORT`
  (default `127.0.0.1:9466`) with the prefix `agloom_rca_generator_`;
- traces runs in Langfuse when both `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` are set. The Langfuse SDK also reads
  `LANGFUSE_BASE_URL`.

## Focused validation

```powershell
$env:LANGGRAPH_STRICT_MSGPACK = "true"
uv run --frozen pytest
cd frontend
npm run build
```

The backend suite covers:

- the response contract and unsupported-citation rejection;
- the evidence policy, including denials of model and direct calls for unknown
  incidents;
- telemetry;
- the real three-stage Agloom pipeline, driven by a scripted chat model.

No request reaches a model provider. To exercise live analysis against the
configured model, run `uv run --frozen pytest -m live`. This loads the
repository-root `.env` when present; variables already set in the environment
take precedence.

## Containers

With Podman Desktop or Docker available, set `EXPLABS_API_KEY` and run:

```powershell
podman compose up --build
```

The backend image is based on `python:3.14-slim` and runs as a non-root user.
The frontend image builds the UI with `node:24-alpine` and serves it with
`nginx:1.30-alpine`. The UI is served at `http://localhost:3103` and proxies
`/api` to the backend.
