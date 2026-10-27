# Installation

Agloom requires **Python 3.12 or later** and supports CPython 3.12, 3.13, and
3.14 on Linux, macOS, and Windows.

## Install the package

=== "pip"

    ```bash
    pip install agloom
    ```

=== "uv"

    ```bash
    uv add agloom
    ```

Agloom does not bundle a model provider. Install the LangChain integration for
the provider you use, for example:

=== "pip"

    ```bash
    pip install langchain-openai      # OpenAI and OpenAI-compatible endpoints
    pip install langchain-anthropic   # Anthropic
    ```

=== "uv"

    ```bash
    uv add langchain-openai      # OpenAI and OpenAI-compatible endpoints
    uv add langchain-anthropic   # Anthropic
    ```

## Optional extras

| Extra | Installs | Use it for |
|---|---|---|
| `agloom[postgres]` | `agentbarrier[postgres]`, `langgraph-checkpoint-postgres` | A PostgreSQL policy store and a PostgreSQL LangGraph checkpointer for production |
| `agloom[postgres-binary]` | `agentbarrier[postgres-binary]`, `langgraph-checkpoint-postgres` | The same PostgreSQL features on hosts without the PostgreSQL client library (libpq), using psycopg's bundled binary build; no separate psycopg installation is needed |
| `agloom[sqlite]` | `langgraph-checkpoint-sqlite` | A durable SQLite LangGraph checkpointer for single-node deployments |

```bash
pip install "agloom[postgres]"
pip install "agloom[postgres-binary]"
pip install "agloom[sqlite]"
```

The agentbarrier SQLite policy store (`agentbarrier.runtime.SQLiteRuntimeStore`)
and LangGraph's in-memory checkpointer are available without extras.

## What comes with Agloom

Agloom builds on the LangChain 1.x and LangGraph 1.x runtimes and installs the
ecosystem packages it integrates with:

| Package | Role in Agloom |
|---|---|
| `langchain`, `langgraph` | Agents, middleware, graphs, checkpoints, interrupts, streaming |
| `agentbarrier` | Policy enforcement, approvals, exactly-once consequential tools, audit receipts |
| `mcp-capability-router` | MCP server registration, discovery, and invocation (FastMCP) |
| `behaviorweave` | Behavioral patterns and interventions, as LangChain middleware |
| `contextsage` | Information-aware summarization, as LangChain middleware |
| `feedback-manager` | Human and evaluator feedback for Loop Engineering |
| `refresh-engine` | Source refresh for Loop Engineering |
| `langgraph-xai` | Opt-in explainability and provenance |
| `xstructured` | Validated structured output |
| `prometheus-client`, `langfuse`, `structlog` | Metrics, tracing, and structured logs |

## Model requirements

Any LangChain `BaseChatModel` works for DIRECT execution. Some features need
more from the model:

- **Tool calling** for tools and every tool-capable topology.
- **Structured output** (`with_structured_output`) for automatic execution
  selection, the Supervisor, Swarm, Planner, and Reflection decisions, and the
  [`ModelLoopEvaluator`][agloom.ModelLoopEvaluator]. With a model that lacks it,
  pin a topology that needs no structured decisions, for example
  `pattern="react"`.

## Verify the installation

```bash
python -c "import agloom; print(agloom.__version__)"
```

## Credentials

Agloom never reads credentials itself; your model client does. Keep keys in
environment variables or a secret manager, never in code.

The repository examples read an OpenAI-compatible endpoint from three
variables. Copy `.env.example` to `.env` (which is git-ignored) and fill in your
values:

```bash
EXPLABS_API_KEY=...        # your API key
EXPLABS_BASE_URL=...       # an OpenAI-compatible base URL
EXPLABS_MODEL=...          # the model name
```

In your own application, use your provider's standard variables, such as
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Next steps

Build your first agent in the [Quickstart](quickstart.md).
