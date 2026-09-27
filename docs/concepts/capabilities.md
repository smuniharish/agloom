# Capabilities

Capabilities are independent of topology. Agloom installs
`feedback-manager`, `behaviorweave`, `contextsage`, `mcp-capability-router`,
`refresh-engine`, `langgraph-xai`, and `xstructured` as required dependencies.
Configure them through `create_agent`; no capability-specific extras are
needed.

## Catalog and routing

Each agent has its own capability catalog. Add LangChain tools through
`tools=`, application services through `capabilities=`, and remote tools or
resources through the MCP options. Agloom presents them through one selection
and execution interface.

The default router supports exact matching, a `CapabilityPolicy`, optional
LangChain `Embeddings`, an optional `BaseRetriever`, and an optional
`BaseDocumentCompressor` reranker:

```python
agent = create_agent(
    model=model,
    tools=tools,
    capabilities={"document_store": document_store},
    capability_policy=policy,
    capability_embeddings=embeddings,
    capability_retriever=retriever,
    capability_reranker=reranker,
    max_selected_capabilities=12,
)

selection = agent.route_capabilities("Search the document archive")
print(selection.names)
```

Custom retrievers and rerankers use normal LangChain contracts. Every returned
`Document` must contain:

```python
Document(
    page_content="Document archive search",
    metadata={"capability_name": "document_store"},
)
```

Pass a complete custom `CapabilityRouter` with `capability_router=...` when the
default pipeline is not appropriate. Router component options cannot be mixed
with a complete custom router.

Named capabilities can also be executed explicitly:

```python
result = agent.execute_capability("local_lookup", {"query": "Agloom"})
result = await agent.aexecute_capability(
    "mcp:documents:remote_search",
    {"query": "Agloom"},
)
```

Each `*_options` mapping is forwarded to the corresponding upstream package
constructor, so its documented constructor arguments override that package's
defaults. For example:

```python
agent = create_agent(
    model=model,
    tools=tools,
    feedback_options={"failure_policy": failure_policy},
    behavior_options={"policies": policies, "state_store": state_store},
    context_options={
        "policy": "maximum_compression",
        "trigger": ("tokens", 80_000),
        "keep": ("messages", 12),
        "safety_margin": 0.2,
    },
    mcp_options={"max_concurrency": 8, "operation_timeout": 10.0},
    refresh_source=source,
    refresh_operation=refresh_operation,
    refresh_options={"config": refresh_config},
    xai_options={"application_id": "my-agent", "tenant_id": "tenant-1"},
    structured_output_schema=ResponseSchema,
    structured_output_options={"inject_instructions": True},
)
```

`feedback_options` configures `FeedbackManager`; `behavior_options` configures
`BehaviorEngine`; `context_options` configures ContextSage middleware using the
agent's model; `mcp_options` configures `MCPRuntime`. Supply `mcp_client` and
`mcp_server_name` to register a concrete LangChain MCP client. Refresh requires
both `refresh_source` and `refresh_operation`. `xai_options` configures
`XAIRuntime`; alternatively inject a prebuilt runtime using `explainability`.
Structured output is available as `agent.structured_output`. The adapters pass
options to the actual upstream constructors, which reject unsupported keys.
MCP routing remains application-configured; do not indiscriminately add every
discovered MCP tool to a model's context.

The supported upstream constructor arguments are:

- `feedback_options`: `FeedbackManager`'s `store`, `router`, `correlator`,
  `xai_runtime`, `provenance_adapter`, `lifecycle_policy`, `redaction_policy`,
  `failure_policy`, and `observability_sink`.
- `behavior_options`: `BehaviorEngine`'s `policies`, `patterns`, and
  `state_store`.
- `context_options`: ContextSage's `trigger`, `keep`, `policy`,
  `maximum_context_tokens`, `reserved_output_tokens`, `safety_margin`,
  `summarization_overhead_tokens`, validation/observability/provenance settings,
  token counter, parsers, preservation regexes, `summary_prompt`, and
  `trim_tokens_to_summarize`. Use `context_model` to override the model supplied
  to that middleware.
- `mcp_options`: `MCPRuntime`'s `registry`, `retriever`, `max_concurrency`,
  `operation_timeout`, `retry`, `rate_limiter`, `interceptors`, and `metrics`.
  `mcp_server_id`, `mcp_client_server_name`, `mcp_discover_resources`,
  `mcp_metadata`, and `mcp_refresh` map to `register_mcp_client` arguments.
- `refresh_options`: `RefreshEngine`'s `fingerprint`, `store`, `config`, and
  `events`.
- `xai_options`: `XAIRuntime`'s `config`, `registry`, `application_id`,
  `tenant_id`, `graph_id`, `plugins`, and `custom_state_capture`.
- `structured_output_options`: xstructured's `envelope`, `parser_config`,
  `multiple`, `multiple_envelopes`, `inject_instructions`, `repair`, and
  `repair_config`.

`invoke`, `stream`, `ainvoke`, and `astream` connect a configured MCP client
when first needed, so applications do not need a separate initialization call.
Use async execution for discovered MCP tools and resources. Known prompts
remain available through the application-owned MCP client.

See the [capability API](../api/capabilities.md) for supported extension
contracts and their parameters.
