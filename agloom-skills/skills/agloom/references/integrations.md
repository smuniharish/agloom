# Integrations

Agloom connects to ecosystem packages through `create_agent()` arguments. The
application creates and owns each component, and closes it when it shuts down.

## MCP servers

```python
from mcp_capability_router import MCPRuntime, RefreshPolicy

async with MCPRuntime() as runtime:
    await runtime.register_mcp(
        "github", "https://mcp.example.com/mcp",
        refresh=RefreshPolicy(on_register=True),   # required to discover capabilities
    )
    agent = create_agent(model=model, mcp=runtime)
    answer = await agent.ainvoke("Summarize the open issues")
```

- Targets can be a URL, a script path, an MCP configuration mapping, an
  in-process `fastmcp.FastMCP` server, a transport, or a `fastmcp.Client`.
- MCP tools become tools named `<server_id>__<tool>`. Resources and prompts
  are capabilities named `mcp:<capability_id>`.
- `agent.aexecute_capability(name, input)` returns MCP results unchanged: a
  tool returns a LangChain `ToolMessage`, a resource returns its contents, and a
  prompt returns MCP's `GetPromptResult`.
- Agents with `mcp=` are async only.
- For stdio servers, set `"keep_alive": false` in the server configuration so
  closing the runtime also stops the server process.

## LangChain middleware

`middleware=[...]` takes any LangChain `AgentMiddleware`. It wraps every
tool-capable agent that Agloom builds, inside capability routing and outside
the Policy Manager.

- **Context management:** `contextsage.IntelligentSummarizationMiddleware(model, trigger=("messages", 40), keep=("messages", 20))`.
- **Behavioral guardrails:** BehaviorWeave with Loop Engineering signals:

  ```python
  from behaviorweave import BehaviorEngine
  from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware
  from agloom import behavior_signals

  guard = BehaviorWeaveMiddleware(BehaviorEngine(policies=[...]),
                                  on_decision=behavior_signals)
  ```

  The scope defaults to the LangGraph `thread_id`; pass `scope=` when running
  without a thread.
- **Tool review:** `langchain.agents.middleware.HumanInTheLoopMiddleware(interrupt_on={"restock": True})`
  pauses before the tool runs (needs a checkpointer). Resume with
  `agent.resume(config, {"decisions": [{"type": "approve"}]})`, an `edit`
  decision with `edited_action`, or `reject`.
- **Other LangChain middleware** (PII redaction, model-call limits, to-do
  lists) works the same way. Tools that middleware provides are registered as
  capabilities and governed by the Policy Manager like any other tool.

## Feedback and refresh

- `feedback=feedback_manager.FeedbackManager(...)` supplies human feedback to
  loops (notes targeted at the thread) and records evaluator feedback. Async
  only.
- `refresh=refresh_engine.RefreshEngine(source, operation)` with
  `LoopPolicy(allow_refresh=True)` reloads stale sources between iterations.
  Async only.

## Structured output

```python
from pydantic import BaseModel

class Report(BaseModel):
    summary: str
    risks: list[str]

structured = agent.with_structured_output(Report)
report = (await structured.ainvoke("Assess the release")).structured
```

xstructured validates the answer, recovers conservatively from formatting
slips, and asks a model to repair a failing answer when you pass `repair=`.
Pass `inject_instructions=False` to place `structured.instructions` in your own
prompt.

## Persistence

- Checkpointers come from LangGraph: `InMemorySaver`, `SqliteSaver`
  (`agloom[sqlite]`), and `PostgresSaver` or `AsyncPostgresSaver`
  (`agloom[postgres]`, or `agloom[postgres-binary]` without a system libpq).
- Every checkpointer needs `checkpoint_authorizer=lambda thread_id: ...`, which
  must confirm the caller owns the thread.

## Observability

```python
from agloom import LangfuseObserver, LoggingObserver, PrometheusObserver

prometheus = PrometheusObserver(namespace="agloom_support")
prometheus.start_http_server(9464, address="0.0.0.0")
agent = create_agent(model=model, observers=[
    LoggingObserver(),
    prometheus,
    LangfuseObserver(),   # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
])
```

- Plain callables are observers too: `observers=[lambda event: ...]`.
- `explainability=langgraph_xai.XAIRuntime(application_id="support")` records
  decision traces (opt-in). Call `agent.flush()` before exit.
- `agent.close()` flushes and shuts down the observers. Components the
  application passed in stay open.

## Deep Agents

Deep Agents middleware such as `FilesystemMiddleware` runs inside Agloom
through `middleware=`. An Agloom agent can also serve as a Deep Agents
subagent: `CompiledSubAgent(name=..., description=..., runnable=agent)`.
