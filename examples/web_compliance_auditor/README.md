# Web Compliance Auditor

An isolated, production-style FastAPI and React/Vite example that audits a
public URL against seeded local web policies. It uses a real OpenAI-compatible
`ChatOpenAI` through environment-only `EXPLABS_*` variables, PostgreSQL +
pgvector persistence, and local Ollama `all-minilm` vectors. Ollama pulls the
model from its own registry; this application has no Hugging Face dependency.
The agent's strict `AuditReport` is grounded: every finding must
name only retrieved policy IDs or the API rejects it.

## Included integration surfaces

- Real `MultiServerMCPClient` stdio configuration for Microsoft [`@playwright/mcp`](https://www.npmjs.com/package/@playwright/mcp), official [`@modelcontextprotocol/server-filesystem`](https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem), and official [`@modelcontextprotocol/server-everything`](https://www.npmjs.com/package/@modelcontextprotocol/server-everything). The factory registers the filesystem server through `mcp-capability-router`; before the first audit the same runtime registers Playwright and Everything, with tool/resource discovery enabled. Prompt discovery remains available from the same real client where the server exposes prompts.
- `FeedbackManager` (human feedback is both persisted and submitted to the capability), `BehaviorWeave`, `ContextSage`, and a real `RefreshEngine` policy-workspace source; plus `langgraph-xai` provenance identifiers persisted with the audit, xstructured structured output, and Langfuse/Prometheus observers (metrics on `9467`).
- `tests/test_agloom_features.py` invokes the shared `examples.agloom_feature_regression` suite. `tests/test_app.py` compares the explicit production call's captured parameter names against `inspect.signature(create_agent).parameters`.

## Run

Do not put keys in this repository or frontend environment. Install uv, copy
`.env.example` privately, set `EXPLABS_API_KEY`, then from this directory:

```powershell
uv sync --frozen
uv run --frozen uvicorn backend.app:app --host 127.0.0.1 --port 8003
cd frontend; npm install; npm run dev
```

The API listens on `8003`; Vite listens on `5176`; metrics listen on `9467`.
For the complete stack, set the same environment variables and run
`podman compose up --build`. PostgreSQL is exposed on `55433`. The UI defaults
to the Compose-local `http://target-site`, an intentionally noncompliant page
that deterministically exercises privacy, transport, and accessibility checks.

The live stack has been validated with Ollama embeddings, pgvector retrieval,
all three MCP servers, and the configured OpenAI-compatible model. Prometheus
exports the application’s runtime series, Grafana includes them in the shared
Agloom dashboard, and Langfuse records model/graph spans with application and
tenant metadata, xAI run IDs, latency, and cost. MCP evidence is included in
the traced model workload; these pre-model MCP calls are not separate
LangChain tool spans.

## Deliberate boundaries

This is a demo, not legal advice. Policies are fictional seeded content in
`data/policies.json`. It does not provide authentication, authorization, rate
limiting, or production migration management. The live application executes
real MCP tool, resource, and prompt calls. Unit tests avoid live providers,
databases, containers, model downloads, MCP processes, and network calls.
