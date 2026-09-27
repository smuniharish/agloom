# Public API

The primary entry point is:

```python
agent = create_agent(model=model, tools=tools)
result = await agent.ainvoke({"messages": [("user", "Hello")]})
```

`pattern` accepts one topology name or a nonempty allow-list. `recursion`
defaults to `False`; policy, checkpointer, strategy extensions, event sink,
generic capabilities, and trusted `system_prompt` are explicit optional
configuration. Automatic and allow-list selection use framework-managed
LLM-based analysis with the configured tool and capability inventory. A
checkpointer also requires a
`checkpoint_authorizer` so the application can enforce access to thread IDs.
Untrusted message inputs accept user and assistant roles only. The returned handle
supports LangChain runnable-style invocation and async invocation; streaming is
available for compatible paths. Public names and accepted shapes are documented
and tested. Package constructor overrides use the corresponding
`feedback_options`, `behavior_options`, `context_options`, `mcp_options`,
`refresh_options`, and `xai_options` mappings; structured output uses
`structured_output_schema` and `structured_output_options`. There is no
`create_harness()` or `harness=True` switch.

Automatic, allow-list, and recursive-child resolution may recommend bounded
workers, pipeline stages, task-specific execution instructions, and Hybrid
composition. Explicit `workers`, `stages`, `system_prompt`, and `composition`
always win. A single explicit topology still bypasses analysis.

Capability routing can be configured with `capability_registry`,
`capability_router`, `capability_policy`, `capability_embeddings`,
`capability_retriever`, `capability_reranker`, and
`max_selected_capabilities`. Embeddings, retrievers, and rerankers use the
existing LangChain contracts.

`observers` accepts one or more `Observer` implementations. Built-in observers
provide structured logging, Prometheus metrics, and Langfuse runtime events plus
LangChain model/tool callbacks. The legacy `event_sink` remains supported and
is adapted into the same observer pipeline. Observer failures are explicit;
Agloom does not silently discard telemetry failures. Call
`agent.flush_observability()` before a process handoff and
`agent.close_observability()` during shutdown.

Explainability is enabled by default. Each agent receives an isolated
`langgraph-xai` runtime, and Agloom instruments analysis, DIRECT execution,
streaming, and compiled topology graphs automatically. `xai_options` overrides
the upstream `XAIRuntime` constructor defaults; `explainability` injects a
prebuilt runtime; `xai_enabled=False` is the explicit opt-out.

MCP client setup remains owned by `mcp-capability-router`. Agloom depends on its
`mcp` dependency extra, registers the client through `MCPRuntime`, and imports
neither MCP transports nor `LangChainMCPAdapter` directly. Discovered MCP
tools, resources, and prompts enter the unified capability catalog.
