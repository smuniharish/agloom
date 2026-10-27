# Changelog

All notable changes to Agloom are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-10-06

Agloom 1.0.1 redesigns the public API around standard LangChain and LangGraph
contracts, adds policy-governed tools, and rebuilds Loop Engineering. Most
changes below require code changes; see the
[documentation](https://agloom.readthedocs.io/en/latest/) for the current API.

### Added

- `PolicyManager`: agentbarrier policies for every tool call. Ordered allow,
  deny, and approval rules with argument conditions, defined in Python, strict
  JSON files, or mappings; tools denied for every call are hidden from the
  model; `on_denied="raise"` or `"inform_model"`. Calls made from code with
  `execute_capability` are governed too, and their denials emit
  `ToolCallDenied`.
- Policies judge the arguments a tool actually receives: every call is
  validated against the tool's schema first (the tool's own Pydantic model,
  with its validators and defaults; strict JSON Schema validation, with
  top-level defaults, for MCP tools), so optional arguments the model omits
  reach rule conditions with their defaults. An invalid call is returned to
  the model, emits `ToolCallDenied` with reason `InvalidArguments`, and records
  no action; `execute_capability` raises `CapabilityResolutionError` instead.
- Consequential tools (`consequential={tool: idempotency_key}`) with approval
  binding, exactly-once execution, replay of recorded results, emergency
  pauses, action limits, and audit receipts, backed by a SQLite or PostgreSQL
  agentbarrier store. A consequential tool that raises fails closed: its
  outcome is recorded as unknown (with an `ActionFailed` event), and calls with
  the same key are refused until an operator reconciles the action in
  agentbarrier. A duplicate model call made while the action runs waits for it
  and replays its result; a duplicate direct call raises `ActionInProgress`.
- Durable approvals and reconciliations: with a checkpointer, a pending
  approval pauses the run (`APPROVAL_INTERRUPT`), and so does a consequential
  tool that raised (`RECONCILIATION_INTERRUPT`). After a reviewer decides, or
  an operator reconciles the action, in agentbarrier, `agent.resume(config)`
  continues: an approved or not-committed call executes exactly once, and a
  committed result is replayed to the model. Without a checkpointer, and for
  direct `execute_capability` and `PolicyManager.execute` calls,
  `ApprovalRequired` or `ActionOutcomeUnknown` is raised instead.
- `ExecutionLimits` for recursion depth, child count, parallel fan-out, plan
  length, handoffs, wall-clock time, and the topologies recursive children may
  select. One recursion budget covers the whole invocation, including every
  loop iteration.
- `middleware=`: LangChain `AgentMiddleware` applied to every tool-capable agent
  in every topology, to DIRECT answers, and to synthesized answers. Tools
  contributed by middleware are registered as capabilities (provider
  `middleware`): routed, shown to task analysis, governed by the policy, and
  declarable as consequential.
- `behavior_signals`, an `on_decision` hook that feeds BehaviorWeave decisions
  into Loop Engineering.
- `mcp=`: an application-owned `MCPRuntime` from mcp-capability-router
  (FastMCP, MCP 2.x) whose catalog is read on every request; MCP tools are
  named `<server_id>__<tool>`.
- `DefaultCapabilityRouter(capability_filter=, embeddings=, retriever=,
  reranker=, max_results=)` and the `CapabilityFilter` contract.
- Loop Engineering on the LangGraph functional API: checkpointed iterations,
  `ModelLoopEvaluator` and `LoopAssessment` (the default judge for
  multi-iteration policies), iteration context passed to the model as data,
  standing human feedback (from `agent.resume` or
  `feedback=FeedbackManager(...)`) applied to every later iteration, source
  refresh through `refresh=RefreshEngine(...)`, loop pauses (`LOOP_INTERRUPT`)
  resumed with feedback, `LoopPolicy.max_context_chars`, and the
  `behavior_intervention` termination reason.
- `Agent.resume` and `Agent.aresume` continue approvals, breakpoints
  (`BREAKPOINT_INTERRUPT`), loop pauses, and runs interrupted by a crash or
  cancellation from the last checkpoint, without repeating completed steps;
  `invoke(None, config)` and `invoke_loop(None, config)` also continue a
  checkpointed thread. A resumed run keeps the plan it paused in, with the
  same capabilities and topology at every recursion depth.
- `Agent.invoke_loop` and `Agent.ainvoke_loop` return a `LoopResult` with the
  answer, the typed loop state, pending interrupts, and the error.
- `Agent.with_structured_output(schema, **options)` validates answers with
  xstructured for any execution mode and accepts text, messages, or mappings.
- Streaming: `stream_mode="messages"` and `"custom"` include nested topology
  graphs; internal decisions are never streamed as tokens.
- Policy and loop events (`ToolCallDenied`, `ApprovalRequired`,
  `ActionExecuted`, `ActionReplayed`, `ActionFailed`, `LoopRefreshed`) and the
  Prometheus metric `policy_actions_total{tool,outcome}`;
  `PrometheusObserver.start_http_server()` and `server_port`.
- Errors: `ExecutionPaused`, `PolicyViolationError`,
  `PolicyConfigurationError`, and `HarnessError`.
- Support for Python 3.13 and 3.14, and the `agloom[postgres]`,
  `agloom[postgres-binary]`, and `agloom[sqlite]` extras.
- Release artifacts built in CI with a CycloneDX SBOM and build provenance
  attestations, published with PyPI trusted publishing.

### Changed

- `recursion=True, recursion_policy=RecursionPolicy(...)` became
  `limits=ExecutionLimits(...)`; `max_subtasks` is now `max_children`, and
  `max_depth=0` disables recursion.
- `ExecutionLimits.max_workers` bounds Planner waves and Blackboard
  contributors. A `max_concurrency` in the run configuration limits only how
  many inputs `batch` runs at once.
- `create_agent` rejects policy-governed tools whose arguments the policy cannot
  judge: schemas other than a Pydantic v2 model or a JSON schema, arguments
  with no JSON form, and tools without an `args_schema` whose `_run` takes
  `*args`, `**kwargs`, or a `RunnableConfig` (such as a plain single-input
  `Tool`); and string idempotency keys that do not name an argument that can
  be a string.
- `capability_policy=` (`CapabilityPolicy.allows`) became
  `capability_router=DefaultCapabilityRouter(capability_filter=...)` with
  `CapabilityFilter.allows(task, capability)`.
- Capability routing is enforced at execution time: a call to an agent tool
  that was not offered for the request is answered with an error tool message
  and a `ToolCallDenied` event with reason `ToolNotOffered`.
- `capability_embeddings`, `capability_retriever`, `capability_reranker`, and
  `max_selected_capabilities` moved to `DefaultCapabilityRouter(embeddings=,
  retriever=, reranker=, max_results=)`.
- `context_options` became ContextSage `IntelligentSummarizationMiddleware` in
  `middleware=`.
- `behavior_options` became BehaviorWeave `BehaviorWeaveMiddleware` in
  `middleware=`, with `on_decision=agloom.behavior_signals`.
- `feedback_options` became `feedback=FeedbackManager(...)`.
- `refresh_source`, `refresh_operation`, and `refresh_options` became
  `refresh=RefreshEngine(source, operation)`; the loop refreshes only with
  `LoopPolicy(allow_refresh=True)`.
- Explainability is opt-in: `xai_options` and `xai_enabled` became
  `explainability=XAIRuntime(...)`.
- `structured_output_schema` and `agent.structured_output` became
  `agent.with_structured_output(schema)`.
- MCP moved from `mcp_client`, `mcp_server_name`, and related parameters
  (langchain-mcp-adapters) to `mcp=MCPRuntime`. Register servers with
  `register_mcp(server_id, target, refresh=RefreshPolicy(on_register=True))`
  (or call `await runtime.refresh()`) so that their capabilities are
  discovered.
- `event_sink=` became `observers=[callable]`.
- `RuntimeError` and `RecursionError` became `HarnessError` and
  `RecursionLimitError`.
- `invoke` and `ainvoke` raise `ExecutionPaused` when a run pauses.
- A checkpointer requires a `checkpoint_authorizer`; `interrupt_before` and
  `interrupt_after` require a checkpointer and an explicit single `pattern`,
  and recursive children never stop at them; `LoopPolicy(allow_topology_change=True)`
  requires automatic selection or an allow-list.
- An invalid Planner plan gets one correction round with the validation error
  before the run fails.
- Input messages may contain only user and assistant turns; trusted
  instructions belong in `system_prompt`.
- Requests that ask for actions no available tool can perform no longer fail:
  the execution is told which actions are unavailable and answers what cannot
  be done, and the `TopologySelected` event reports how many were missing.
- `LoopPolicy.max_retries` alone bounds retries after failed iterations; it
  replaces `max_failures`.
- Agents configured with `mcp=`, `refresh=`, or `feedback=` use the async API.
- Python 3.12 or later; LangChain 1.4, LangGraph 1.2, and all integrations
  upgraded to their current releases.

### Removed

- The durable runtime (`agloom.runtime.durable`, including `DurableWorker`,
  `InMemoryDurableStore`, `RunControl`, `CheckpointTakeoverPolicy`,
  `verify_durable_store`, and `ResumeMode`). Use a LangGraph checkpointer with
  `agent.resume(config)`, consequential tools for exactly-once side effects,
  and your application's job queue for background execution.
- `RuntimeController` (call `agent.stop()` on each agent), `Lifecycle`, and
  `LifecycleState` (observe events instead).
- `Loop` and `LoopExecution`; every invocation runs Loop Engineering through
  the agent, and `invoke_loop` returns the loop state.
- The capability providers (`CapabilityProvider`, `LocalLangChainToolProvider`,
  `ApplicationCapabilityProvider`, `MCPCapabilityProvider`, and
  `MCPRemoteCapability`); capabilities come from `tools=`, `capabilities=`,
  `middleware=`, and `mcp=`.
- `LifecycleError`, `LoopConfigurationError`, `LoopPolicyError`,
  `LoopIterationLimitError`, `LoopTimeoutError`, `LoopStagnationError`, and
  `LoopOscillationError`. Iteration and duration budgets, stagnation, and
  oscillation end a loop with a termination reason, and an invalid
  `LoopPolicy` raises Pydantic's `ValidationError`.
- `application_observers`, `observability_enabled`, and
  `observability_status`.
- `CallableObserver` and `CompositeObserver` from the public API; pass
  callables in `observers=` directly.
- `LoopPolicy(max_failures=...)` and `allow_delegation`.
- The `*_options` parameters, `context_model`, and the `mcp_*` parameters of
  `create_agent`.
- The `langchain-mcp-adapters` dependency.
- The `langchain-openai` dependency; install the LangChain integration for
  your model provider.

### Fixed

- Agents no longer set `name` on assistant messages, which several
  OpenAI-compatible providers reject; agent names are kept as LangChain
  metadata.
- Checkpoints contain only JSON-compatible values and LangChain messages, so
  agents run with `LANGGRAPH_STRICT_MSGPACK=true`.
- An agent that pauses inside another graph, or whose run LangGraph drains,
  reports `ExecutionPaused` instead of `ExecutionFailed`, and a stream consumer
  that stops early or an `asyncio` cancellation (a timeout or a client
  disconnect) reports `ExecutionCancelled`.

## [1.0.0] - 2026-09-27

Initial release of the Agloom agent harness: `create_agent` with automatic,
constrained, or explicit selection between DIRECT execution and eight
topologies; capability routing across LangChain tools, application services,
and MCP; bounded recursion; Loop Engineering; checkpointing, human-in-the-loop,
and streaming; and observers for structured logs, Prometheus, and Langfuse.

[1.0.1]: https://github.com/smuniharish/agloom/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/smuniharish/agloom/releases/tag/v1.0.0
