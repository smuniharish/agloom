# Dependency matrix

| Component | Status | Responsibility |
|---|---|---|
| Python | Required | Python 3.12 runtime target |
| `langchain-core` | Required | Models, messages, tools, runnable contracts |
| `langgraph` | Required | Graph execution, interrupts, checkpoint integration |
| `pydantic` | Required | Validated configuration and structured contracts |
| MkDocs / Material / mkdocstrings | Development/docs | Documentation site |
| pytest / pytest-asyncio | Development/test | Deterministic tests |
| Ruff / Black / Pyrefly | Development/quality | Lint, formatting, type checking |
| `feedback-manager` | Required | Feedback capability |
| `behaviorweave` | Required | Behavior capability |
| `contextsage` | Required | Context middleware |
| `mcp-capability-router` | Required | MCP routing |
| `refresh-engine` | Required | Refresh capability |
| `langgraph-xai` | Required | Explainability instrumentation |
| `xstructured` | Required | Structured output |

No LangSmith, provider-specific model package, Docker, database, or hosted
service is required by core. Package constructor options pass through to real
upstream APIs. Dependency bounds remain within verified Python 3.12 support.

<picture class="agloom-diagram">
<source media="(max-width: 600px)" srcset="../../assets/diagrams/dependency-matrix-mobile.png">
<img src="../../assets/diagrams/dependency-matrix.png" alt="Agloom runtime dependency relationships">
</picture>
