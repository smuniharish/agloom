# `create_agent()`

`create_agent()` is the primary Agloom API. It returns an isolated
LangChain-compatible [`Agent`](agent.md).

```python
from agloom import create_agent

agent = create_agent(model=model, tools=tools)
```

## Signature

```python
create_agent(
    *,
    model,
    tools=(),
    pattern=None,
    recursion=False,
    recursion_policy=None,
    composition=None,
    workers=(),
    stages=(),
    checkpointer=None,
    checkpoint_authorizer=None,
    interrupt_before=(),
    interrupt_after=(),
    event_sink=None,
    observers=(),
    capabilities=None,
    capability_registry=None,
    capability_router=None,
    capability_policy=None,
    capability_embeddings=None,
    capability_retriever=None,
    capability_reranker=None,
    max_selected_capabilities=20,
    middleware=(),
    explainability=None,
    feedback_options=None,
    behavior_options=None,
    context_model=None,
    context_options=None,
    mcp_options=None,
    mcp_client=None,
    mcp_server_name=None,
    mcp_server_id=None,
    mcp_client_server_name=None,
    mcp_discover_resources=False,
    mcp_metadata=None,
    mcp_refresh=None,
    refresh_source=None,
    refresh_operation=None,
    refresh_options=None,
    xai_options=None,
    xai_enabled=True,
    structured_output_schema=None,
    structured_output_options=None,
    system_prompt=None,
    max_reflections=1,
    strategy=None,
    compiler=None,
) -> Agent
```

All parameters are keyword-only.

## Core execution

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `model` | `BaseChatModel` | **Required** | — | LangChain chat model used for analysis and execution. |
| `tools` | `Sequence[BaseTool \| Callable]` | Optional | `()` | LangChain tools or compatible callables available to the agent. |
| `pattern` | `str \| TopologyName \| Sequence[...] \| None` | Optional | `None` | `None` enables automatic DIRECT/topology selection. One topology bypasses selection. A sequence is an allow-list. |
| `system_prompt` | `str \| None` | Optional | `None` | Trusted application instruction prepended to model execution. Do not place untrusted user input here. |
| `max_reflections` | `int` | Optional | `1` | Maximum Reflection revision rounds. Accepted range: 1–5. |

Accepted topology names are `react`, `supervisor`, `pipeline`, `planner`,
`reflection`, `swarm`, `blackboard`, and `hybrid`. DIRECT is an execution mode,
not a topology name.

## Topology configuration

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `composition` | `Sequence[str \| TopologyName] \| None` | Conditional | `None` | Two to four unique, non-Hybrid child topologies. Required when Hybrid is an allowed or explicit pattern. |
| `workers` | `Sequence[WorkerSpec]` | Optional | `()` | Explicit workers for delegation-oriented topologies. Developer-supplied workers override generated suggestions. |
| `stages` | `Sequence[PipelineStage]` | Optional | `()` | Explicit ordered stages for Pipeline. Developer-supplied stages override generated suggestions. |
| `strategy` | `Strategy \| None` | Advanced | `None` | Custom execution-selection strategy. The built-in strategy is used when omitted. |
| `compiler` | `ArchitectureCompiler \| None` | Advanced | `None` | Custom topology compiler extension. The built-in compiler is used when omitted. |

## Recursion

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `recursion` | `bool` | Optional | `False` | Allows topology-created subtasks to resolve independently within one bounded recursion session. |
| `recursion_policy` | `RecursionPolicy \| None` | Optional | `None` | Bounds depth, workers, subtasks, execution time, and nested topologies. `None` creates the default policy. |

## Checkpointing and human approval

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `checkpointer` | `BaseCheckpointSaver \| None` | Optional | `None` | LangGraph checkpointer used for persistence, interrupts, state reads, and resume. Requires one explicit topology. |
| `checkpoint_authorizer` | `Callable[[str], bool] \| None` | Conditional | `None` | Authorizes every checkpoint thread ID. Required at runtime when a checkpointer is configured. |
| `interrupt_before` | `Sequence[str]` | Optional | `()` | Graph node names that pause execution before the node runs. Requires `checkpointer`. |
| `interrupt_after` | `Sequence[str]` | Optional | `()` | Graph node names that pause execution after the node runs. Requires `checkpointer`. |

## Capabilities and routing

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `capabilities` | `Mapping[str, Any] \| None` | Optional | `None` | Named application services added to the agent capability catalog. Names must be non-empty and unique. |
| `capability_registry` | `CapabilityRegistry \| None` | Optional | `None` | Custom agent-local capability registry. Uses `InMemoryCapabilityRegistry` when omitted. |
| `capability_router` | `CapabilityRouter \| None` | Optional | `None` | Complete custom router. Cannot be combined with the five router component options below. |
| `capability_policy` | `CapabilityPolicy \| None` | Optional | `None` | Policy applied before relevance selection. The default allows all registered capabilities. |
| `capability_embeddings` | `Embeddings \| None` | Optional | `None` | LangChain embeddings used for semantic capability matching. |
| `capability_retriever` | `BaseRetriever \| None` | Optional | `None` | LangChain retriever that returns capability names in document metadata. |
| `capability_reranker` | `BaseDocumentCompressor \| None` | Optional | `None` | LangChain document compressor used to rerank candidate capabilities. |
| `max_selected_capabilities` | `int` | Optional | `20` | Maximum capabilities returned by the default router. Must be at least 1. |
| `middleware` | `Sequence[Any]` | Optional | `()` | LangChain-compatible middleware applied to supported topology agents. |

The router component options are `capability_policy`,
`capability_embeddings`, `capability_retriever`, `capability_reranker`, and
`max_selected_capabilities`.

## Observability and explainability

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `observers` | `Sequence[Observer]` | Optional | `()` | Structured logging, metrics, Langfuse, or custom observers. |
| `event_sink` | `Callable[[AgentEvent], None] \| None` | Optional | `None` | Legacy single-event callback. Prefer `observers` for new applications. |
| `explainability` | `Any` | Optional | `None` | Preconfigured `langgraph-xai` runtime. Cannot be combined with `xai_options`. |
| `xai_options` | `Mapping[str, Any] \| None` | Optional | `None` | Options forwarded to the `langgraph-xai` runtime constructor. |
| `xai_enabled` | `bool` | Optional | `True` | Enables the isolated explainability runtime. Set `False` for an explicit opt-out. |

## Ecosystem integrations

These mappings are passed to the corresponding installed package constructor.
Unsupported keys fail explicitly.

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `feedback_options` | `Mapping[str, Any] \| None` | Optional | `None` | Constructs and registers a `feedback-manager` capability. |
| `behavior_options` | `Mapping[str, Any] \| None` | Optional | `None` | Constructs and registers a `behaviorweave` capability. |
| `context_model` | `BaseChatModel \| str \| None` | Optional | `None` | Model override for ContextSage. Requires `context_options`; otherwise the agent model is used. |
| `context_options` | `Mapping[str, Any] \| None` | Optional | `None` | Constructs ContextSage middleware and registers it as a capability. |
| `mcp_options` | `Mapping[str, Any] \| None` | Optional | `None` | Constructs the `mcp-capability-router` runtime. |
| `refresh_source` | `Any` | Conditional | `None` | RefreshEngine source. Must be supplied with `refresh_operation`. |
| `refresh_operation` | `Any` | Conditional | `None` | Operation refreshed by RefreshEngine. Must be supplied with `refresh_source`. |
| `refresh_options` | `Mapping[str, Any] \| None` | Optional | `None` | Additional RefreshEngine constructor options. Requires the source and operation. |
| `structured_output_schema` | `Any` | Optional | `None` | Schema used to create `agent.structured_output` through xstructured. |
| `structured_output_options` | `Mapping[str, Any] \| None` | Optional | `None` | xstructured wrapper options. Requires `structured_output_schema`. |

See [Capabilities](../concepts/capabilities.md) for supported upstream option
names and integration examples.

## MCP registration

| Parameter | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `mcp_client` | `Any` | Optional | `None` | Concrete LangChain MCP client to register and discover at first execution. |
| `mcp_server_name` | `str \| None` | Conditional | `None` | Non-empty server name. Required when `mcp_client` is supplied. |
| `mcp_server_id` | `str \| None` | Optional | `None` | Registry ID override; defaults to `mcp_server_name`. |
| `mcp_client_server_name` | `str \| None` | Optional | `None` | Server name understood by the supplied client when different from the registry ID. |
| `mcp_discover_resources` | `bool` | Optional | `False` | Includes MCP resources during registration and discovery. |
| `mcp_metadata` | `Mapping[str, Any] \| None` | Optional | `None` | Application metadata attached to the MCP server registration. |
| `mcp_refresh` | `Any` | Optional | `None` | Registration refresh policy. Defaults to refresh-on-register when omitted. |

All MCP registration options other than `mcp_options` require `mcp_client`.
Use async execution for discovered remote capabilities.

## Return value

Returns an [`Agent`](agent.md), which implements LangChain Runnable-style
synchronous, asynchronous, and streaming execution.

## Errors

Invalid combinations raise
[`ConfigurationError`](errors.md). Capability, selection,
execution, recursion, and cancellation failures use distinct public exception
types.
