# Why Agloom?

Building one LLM call is easy. Building an agent that remains predictable when
tasks need tools, planning, multiple workers, retries, long context, human
approval, or remote MCP capabilities is not.

Agloom provides one harness for that transition. The application keeps one
construction API while Agloom selects the lightest execution shape that can
complete the request.

## The problems developers face

| Problem | Without a harness | How Agloom helps |
|---|---|---|
| Simple prompts pay an orchestration tax | Every request enters a planner or multi-agent graph | Straightforward tasks use DIRECT execution |
| One graph does not fit every task | Teams accumulate branching, hard-to-test workflow code | Eight bounded topologies provide explicit execution semantics |
| Automatic routing ignores available tools | The model plans work it cannot execute | Selection receives the actual capability inventory and fails when a required capability is unavailable |
| Adding MCP creates a second tool system | Local tools, app services, and remote capabilities are routed differently | One capability catalog covers local, application, and MCP providers |
| Multi-agent work can grow without bounds | Recursive tasks and handoffs consume unbounded time and workers | Recursion and concurrency are controlled by explicit policies |
| Long-running work is hard to operate | Pause, resume, cancellation, and checkpoints are added ad hoc | LangGraph checkpoints, interrupts, streaming, and lifecycle controls are part of the agent handle |
| Production failures are opaque | Logs do not connect model, tool, graph, and application activity | Observers integrate structured logs, Prometheus, Grafana, Langfuse, and langgraph-xai |
| Structured output breaks on model prose | Each application writes custom parsing and repair logic | xstructured integration validates and repairs typed results |

## One API from simple to complex

### A simple question stays simple

```python
agent = create_agent(model=model, tools=[lookup_exchange_rate])
answer = agent.invoke("What is 6 multiplied by 7?")
```

**Result:** the verified DIRECT example returns `42` without calling the
irrelevant exchange-rate tool.

### A tool request gets tool-capable execution

```python
agent = create_agent(model=model, tools=[average_speed])
answer = agent.invoke(
    "Use average_speed for 120 kilometers in 2 hours. Return km/h."
)
```

**Result:** automatic selection calls
`average_speed(distance_km=120, duration_hours=2)` once and returns `60 km/h`.

### Developer intent still wins

```python
planner = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
)
```

A single explicit topology bypasses automatic selection. For a constrained
choice, pass an allow-list:

```python
agent = create_agent(
    model=model,
    tools=tools,
    pattern=["planner", "supervisor"],
)
```

This selects one candidate. It does not compose them. Use the Hybrid topology
for explicit composition.

## Capabilities without parallel infrastructure

```python
agent = create_agent(
    model=model,
    tools=local_tools,
    capabilities={"document_store": document_store},
    mcp_client=mcp_client,
    mcp_server_name="operations",
    capability_policy=policy,
    capability_retriever=retriever,
    capability_reranker=reranker,
)

selection = await agent.aroute_capabilities(
    "Find the deployment guide and current service status."
)
print(selection.names)
```

A representative selection is:

```text
("document_store", "mcp:operations:tool:service_status")
```

The exact selection depends on the configured catalog and policy. The important
contract is systematic: policy filtering runs before optional retrieval and
reranking, and only selected capabilities reach task analysis and execution.

## Production controls are configuration, not rewrites

```python
agent = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
    recursion=True,
    recursion_policy=RecursionPolicy(
        max_depth=2,
        max_subtasks=8,
        max_workers=4,
        max_execution_time=120.0,
    ),
    observers=[prometheus, langfuse],
    structured_output_schema=IncidentReport,
)
```

The same runnable API now has bounded child work, telemetry, explainability, and
validated output. Applications can add checkpoint-backed approval without
replacing their agent abstraction.

## Proof beyond toy snippets

The repository includes five complete FastAPI/React applications:

- an enterprise knowledge assistant;
- a QA customer chatbot;
- an RCA generator;
- a web compliance auditor using three real MCP servers;
- an engineering change investigator using real MCP, PostgreSQL/pgvector, and
  Ollama embeddings.

See [examples and verified results](../examples/overview.md) for executable
snippets, semantic assertions, application coverage, and current validation
results.
