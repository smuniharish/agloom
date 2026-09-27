# Production runtime

Agloom is embedded in your application process. Your application remains
responsible for authentication, request admission, data access, deployment,
and infrastructure scaling; Agloom manages agent execution within those
boundaries.

## Isolate agents by configuration

Each call to `create_agent()` returns an independent agent. Do not reuse a
tenant-specific capability registry, checkpointer authorization function, or
observer configuration across trust boundaries.

## Secure checkpoint access

When checkpointing is enabled:

1. use one explicit topology;
2. supply a durable LangGraph checkpointer;
3. provide `checkpoint_authorizer`;
4. pass a stable `configurable.thread_id` on invoke, resume, and state reads;
5. authorize the thread against the current application principal.

```python
run_config = {"configurable": {"thread_id": conversation_id}}
result = agent.invoke(task, config=run_config)
```

## Handle cancellation and shutdown

`agent.stop()` requests cooperative cancellation for active invocations.
Before process handoff or shutdown, flush and close telemetry:

```python
agent.flush_observability()
agent.close_observability()
```

Use `RuntimeController` only when your application needs to coordinate stop
across a known set of agents.

## Prefer async for async capabilities

Use `ainvoke()`, `astream()`, and `aexecute_capability()` when MCP or another
capability performs asynchronous I/O. Do not call synchronous execution from
inside an active event loop.

## Make failures observable

Catch the narrowest public Agloom exception that your application can handle.
Configuration, capability resolution, topology selection, recursion limits,
execution, and cancellation have distinct exception types. Observer failures
are not silently discarded.

See the [Agent API](../api/agent.md), [observability API](../api/observability.md),
and [error reference](../api/errors.md).
