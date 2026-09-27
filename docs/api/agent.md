# `Agent`

`create_agent()` returns an `Agent`. Applications should construct it through
`create_agent()` rather than calling the class constructor.

## Execution methods

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `invoke(input, config=None)` | `input` **required**; `config` optional `RunnableConfig` | Final model message or topology result | Runs one request synchronously. |
| `ainvoke(input, config=None)` | `input` **required**; `config` optional `RunnableConfig` | Awaitable final result | Runs one request asynchronously. |
| `stream(input, config=None)` | `input` **required**; `config` optional `RunnableConfig` | Iterator of chunks | Streams one request synchronously. |
| `astream(input, config=None)` | `input` **required**; `config` optional `RunnableConfig` | Async iterator of chunks | Streams one request asynchronously. |

`input` may be a string, a supported message sequence, or a mapping containing
`messages`. Checkpoint resume accepts a LangGraph `Command` when the agent has
one explicit topology and a configured checkpointer. Unexpected keyword
options raise `TypeError` rather than being ignored.

## Capability methods

| Method | Parameters | Returns | Description |
|---|---|---|---|
| `route_capabilities(task)` | `task: str` **required** | `CapabilitySelection` | Selects allowed, relevant capabilities synchronously. |
| `aroute_capabilities(task)` | `task: str` **required** | `CapabilitySelection` | Async equivalent for retrievers and rerankers with async support. |
| `execute_capability(name, input=None)` | `name: str` **required**; `input` optional | Capability result | Executes a named local synchronous capability. |
| `aexecute_capability(name, input=None)` | `name: str` **required**; `input` optional | Awaitable capability result | Executes local or asynchronous remote capabilities. |

## Runtime methods and properties

| Member | Parameters | Returns | Description |
|---|---|---|---|
| `get_state(config)` | `config: RunnableConfig` **required** | LangGraph state snapshot | Reads authorized checkpoint state. Requires one explicit topology and a checkpointer. |
| `stop()` | None | `None` | Requests cooperative cancellation of active invocations. |
| `flush_observability()` | None | `None` | Flushes explainability and observer buffers. |
| `close_observability()` | None | `None` | Flushes telemetry and releases observer resources. |
| `xai` | Property | Runtime or `None` | The agent's explainability runtime when enabled. |
| `structured_output` | Attribute | Runnable or `None` | xstructured runnable when a schema was configured. |

## Invocation example

```python
result = await agent.ainvoke(
    {"messages": [("user", "Summarize the incident report.")]},
    config={"configurable": {"thread_id": "incident-42"}},
)
```

Use async methods when your tools, retriever, reranker, or MCP client performs
asynchronous I/O.
