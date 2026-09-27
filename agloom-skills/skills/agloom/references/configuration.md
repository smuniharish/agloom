# Configuration decisions

Configure Agloom through `create_agent()`. Use the smallest public option set
that owns the requested behavior.

Authoritative resources:

- [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples)
- [Agloom documentation](https://agloom.readthedocs.io/en/latest/)

## Execution selection

| Need | Configuration |
| --- | --- |
| Automatic DIRECT/topology selection | Omit `pattern` |
| Deterministic topology | `pattern="planner"` or another topology |
| Constrained automatic choice | `pattern=["planner", "supervisor"]` |
| Explicit composition | `pattern="hybrid", composition=[...]` |
| Bounded child tasks | `recursion=True, recursion_policy=...` |
| Task-specific trusted instruction | `system_prompt=...` |
| Explicit worker roles | `workers=[WorkerSpec(...)]` |
| Explicit ordered stages | `stages=[PipelineStage(...)]` |

`composition` requires Hybrid. Hybrid accepts two to four non-Hybrid children.
A list in `pattern` remains an allow-list.

## Capability routing

| Need | Public option |
| --- | --- |
| Local callable or LangChain tools | `tools` |
| Named application services | `capabilities` |
| Alternate catalog | `capability_registry` |
| Complete routing replacement | `capability_router` |
| Authorization/filtering | `capability_policy` |
| Semantic ranking | `capability_embeddings` |
| External candidate retrieval | `capability_retriever` |
| Final compression/reranking | `capability_reranker` |
| Selection bound | `max_selected_capabilities` |

A complete `capability_router` is mutually exclusive with the default router's
policy, embeddings, retriever, reranker, and non-default result bound.

## Runtime controls

| Need | Public option or API |
| --- | --- |
| Durable graph state | `checkpointer` |
| Thread authorization | `checkpoint_authorizer` |
| Human approval boundary | `interrupt_before`, `interrupt_after` |
| Resume | `agent.invoke(Command(resume=...), config=...)` |
| Cancellation | `agent.stop()` |
| Streaming | `agent.stream(...)`, `agent.astream(...)` |
| Runtime events | `observers` or legacy `event_sink` |

A checkpointer requires a `checkpoint_authorizer`. Use a stable
`configurable.thread_id`; never treat that identifier as an authorization
credential.

## Ecosystem integrations

| Integration | Configuration |
| --- | --- |
| FeedbackManager | `feedback_options` |
| BehaviorWeave | `behavior_options` |
| ContextSage | `context_options`, optional `context_model` |
| MCP Capability Router | `mcp_options` and MCP client registration options |
| RefreshEngine | `refresh_source`, `refresh_operation`, `refresh_options` |
| langgraph-xai | `xai_options`, `explainability`, `xai_enabled` |
| xstructured | `structured_output_schema`, `structured_output_options` |

Option mappings are passed to the corresponding package constructors. Verify
keys against installed versions instead of inventing compatibility aliases.

## Extension decisions

Inject `strategy` or `compiler` only when the application has a demonstrated
extension requirement. Prefer normal `create_agent()` configuration for
selection and topology behavior. Preserve public contracts and add focused
tests for every custom extension.
