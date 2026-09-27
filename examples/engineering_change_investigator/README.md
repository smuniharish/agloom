# Engineering Change Investigator

A real Agloom application that assesses release readiness using:

- a live OpenAI-compatible model from `EXPLABS_*` environment variables;
- PostgreSQL 17 with pgvector and local Ollama `all-minilm` embedding
  inference, with no Hugging Face dependency;
- the real Playwright, Filesystem, and Everything MCP servers;
- MCP Capability Router discovery, execution, health, and a durable registry;
- durable FeedbackManager, BehaviorWeave, RefreshEngine, and langgraph-xai state;
- ContextSage, xstructured, Prometheus, Langfuse, and Agloom xAI instrumentation;
- FastAPI and a responsive React interface.

## Run

Python dependencies are managed by uv and locked in `uv.lock`. For a local
backend, copy `.env.example` to `.env`, supply environment-only model and
optional Langfuse credentials, then run:

```powershell
uv sync --frozen
uv run --frozen uvicorn backend.app:app --host 127.0.0.1 --port 8004
```

For the complete container stack, run:

```powershell
podman compose up --build
```

Open `http://localhost:5177`. API and metrics are exposed on ports 8004 and
9468. No credential is stored in this example.

The filesystem MCP server is restricted to `workspace/`. The Playwright MCP
server runs headlessly and inspects only the Compose-local target site in the
seeded workflow.

The seeded live workflow has been validated end to end. It rejects release 4.9
because its 16-connection pool is below the 40-connection minimum and the
required load test and database approval are absent. Every rendered evidence
citation must exactly match a canonical `ADR-*`, `RELEASE-*`, or `RUNBOOK-*`
document ID. Prometheus and the shared Grafana dashboard expose runtime
metrics; Langfuse records model/graph spans with application and tenant
metadata, xAI run IDs, latency, and cost. MCP verification is carried in the
traced workload input rather than emitted as standalone LangChain tool spans.

## What is durable

The app implements package-native persistence contracts for FeedbackManager,
BehaviorWeave, RefreshEngine, and MCP Capability Router in PostgreSQL.
`langgraph-xai` uses its native `PostgresProvenanceStore`. ContextSage is
context middleware and xstructured is a structured-output parser; neither has
a database persistence contract, so the validated investigation report is the
record persisted by the application boundary rather than inventing unsupported
storage APIs.
