# Real-world Agloom applications

These examples are complete FastAPI and React applications rather than
single-file demonstrations:

- `enterprise_knowledge_assistant/` retrieves and cites internal knowledge.
- `qa_customer_chatbot/` answers customer questions, checks operational data,
  captures feedback, and applies behavioral guardrails.
- `rca_generator/` turns incident evidence into a structured root-cause
  analysis.
- `web_compliance_auditor/` combines Playwright, Filesystem, and Everything MCP
  evidence with Ollama embeddings and pgvector-backed policy retrieval.
- `engineering_change_investigator/` evaluates release readiness from
  canonical engineering records, live browser evidence, and durable
  integration state.

Each application owns its backend and frontend dependencies, seeded demo data,
tests, and container configuration. Python dependencies are declared in each
application's `pyproject.toml`, locked in `uv.lock`, and installed with
`uv sync --frozen`; backend container images use the same lockfiles. They use
the OpenAI-compatible endpoint configured through:

```text
EXPLABS_API_KEY
EXPLABS_BASE_URL
EXPLABS_MODEL
```

Credentials are runtime-only and must not be committed. See each application's
README for local and container commands.

## Regression contract

Each application runs its domain tests plus the shared
`agloom_feature_regression.py` contract. The contract independently verifies,
from every application's test suite:

- all eight execution topologies plus DIRECT;
- every public `create_agent()` parameter;
- all 128 enabled/disabled combinations of feedback, behavior, ContextSage,
  MCP, RefreshEngine, xAI, and xstructured;
- capability routing and execution, observers and event sinks;
- feedback lifecycle and BehaviorWeave intervention;
- MCP registration and health, refresh execution, and structured output;
- checkpoint interruption/resume, bounded recursion, and streaming.

The production paths still enable only integrations that serve the
application's domain. `GET /api/health` reports those active runtime features;
the broader matrix is regression-only so examples do not add unnecessary
orchestration to normal requests.

## Live observability

The compliance and engineering examples join the shared observability network.
Prometheus scrapes both applications, and the provisioned Agloom Runtime
dashboard includes their execution, duration, active-execution, and lifecycle
series. Langfuse records model and LangGraph spans with application and tenant
metadata, xAI run identifiers, latency, and model cost. MCP calls execute
before the model invocation, so their verified results appear in the traced
workload input rather than as standalone LangChain tool spans.
