# RCA Generator

An incident-scoped root cause analysis example with a FastAPI API, Agloom agent,
and React UI. Two synthetic but operationally realistic seeded incidents include
incident events, timestamped logs, and metric readings. No credentials or reports
are persisted.

## Run locally

Requires uv, Python 3.12, Node.js, an OpenAI-compatible chat endpoint, and a
model that can follow the structured-output envelope instructions. From this
directory:

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
For separately hosted frontends, set `VITE_API_BASE_URL` to your trusted API
prefix at build time and configure CORS or a reverse proxy appropriately; the
backend does not enable cross-origin access by default. Never put a model token
in `VITE_` variables or commit a populated `.env`.

## API

- `GET /api/health` returns `status` (`ok` or `unconfigured`), incident count
  `model_configured`, and active Agloom runtime features. It does not contact
  the model provider.
- `GET /api/incidents` lists public incident metadata without raw records.
- `POST /api/analyze` with `{"incident_id":"INC-2026-041"}` returns a typed
  `RcaReport` with root-cause and overall confidence in `[0,1]`, timestamped
  timeline, recommendations, evidence and exact citation IDs. An unknown incident
  returns 404, missing configuration 503, and model/validation failure 502.

The app fetches a bounded incident/log/metric bundle through its scoped tools
before the model runs. Agloom's fixed **pipeline** topology hypothesizes,
checks the hypothesis, and drafts a structured report with `RcaReport` as its
structured-output schema. The server enforces that IDs refer only to the
selected incident, populates authoritative citation text and timestamps, and
rejects fabricated citation metadata. Confidence is a model judgment, not a
calibrated probability; human review remains essential. Citation-ID validation
cannot prove that every natural-language inference is correct.

## Focused validation

```powershell
uv run --frozen python -m pytest -q tests
cd frontend
npm run build
```

The backend suite uses a fake agent to test the response contract and
unsupported-citation rejection without contacting a model. To exercise live
analysis, configure a model and submit an incident through the UI or API.
It also runs the shared full-surface Agloom regression contract described in
`../REAL_WORLD_APPLICATIONS.md`.

## Containers

With Podman Desktop or Docker available, set `EXPLABS_API_KEY` and run:

```powershell
podman compose up --build
```

The UI is served at `http://localhost:3103` and proxies `/api` to the backend.
