# Architecture

An Agloom agent is a LangChain `Runnable` built from a few composable layers.
Each layer is a standard LangChain or LangGraph construct, so everything you
know about those libraries still applies.

![Agloom architecture](../assets/diagrams/architecture-light.png#only-light)
![Agloom architecture](../assets/diagrams/architecture-dark.png#only-dark)

## Layers

| Layer | What it does | Built on |
|---|---|---|
| **Agent** | The public runnable: `invoke`, `stream`, `resume`, `invoke_loop`, and their async twins | LangChain `Runnable` |
| **Loop Engineering** | The outer objective loop: budgets, evaluation, decisions, pauses, and termination | LangGraph functional API (`entrypoint` and checkpointed `task`s) |
| **Capability routing** | Selects the local tools, application services, and MCP capabilities relevant to the request | LangChain retrievers, embeddings, and document compressors |
| **Execution selection** | Structured task analysis plus a deterministic strategy choose DIRECT or one topology | Provider-native structured output |
| **Topologies** | Eight compiled graphs, cached per configuration | LangGraph `StateGraph` |
| **Tool-capable agents** | Every worker, stage, step, and peer that calls tools | LangChain `create_agent` |
| **Middleware stack** | Tool routing, your middleware, then policy enforcement, around every model and tool call | LangChain `AgentMiddleware` |
| **Policy** | Allow, deny, or approval for every tool call; exactly-once consequential actions | agentbarrier |
| **Observers** | Normalized events, metrics, logs, and traces | LangChain callbacks |

## Request lifecycle

![Request lifecycle](../assets/diagrams/request-lifecycle-light.png#only-light)
![Request lifecycle](../assets/diagrams/request-lifecycle-dark.png#only-dark)

1. **Admission.** `invoke` normalizes the input (text, messages, or a mapping)
   and rejects system or tool messages. With a checkpointer, the
   `checkpoint_authorizer` must accept the thread before anything runs.
2. **Loop start.** Loop Engineering creates the loop state and budget for the
   objective, which is the latest user request.
3. **Routing.** Each iteration routes capabilities for the task: filtering,
   then optional name matching, semantic retrieval, and reranking.
4. **Selection.** With an explicit `pattern`, the topology is fixed. Otherwise
   a structured [`TaskAnalysis`][agloom.TaskAnalysis] describes the task
   against the routed inventory, and the [`StrategyEngine`][agloom.StrategyEngine]
   maps it to DIRECT or one topology. Requested actions that no routed tool can
   perform are passed to the execution as actions it must not claim to perform.
5. **Execution.** DIRECT makes one model call without tools, through your
   middleware when configured. A topology runs its compiled graph; every
   tool-capable agent inside it is offered only the routed tools, calls to
   tools that were not offered are refused, and every tool call passes
   through your middleware and the policy.
6. **Assessment.** The iteration result is observed, evaluated against the
   objective, and a controller decides to complete, continue, adapt, pause, or
   stop, within the [`LoopPolicy`][agloom.LoopPolicy].
7. **Result.** `invoke` returns the answer, raises
   [`ExecutionPaused`][agloom.ExecutionPaused] for a pause, or raises the
   failure. `invoke_loop` returns all of it as a
   [`LoopResult`][agloom.LoopResult].

Every step emits a [runtime event](observability.md#event-catalog).

## Trust boundaries

- **Trusted instructions** come only from `system_prompt` and your topology
  configuration (worker descriptions and instructions, pipeline stage
  instructions).
- **Input messages** may contain only user and assistant turns. System and tool
  messages are rejected, so callers cannot inject instructions or forge tool
  results.
- **Generated guidance is data.** Task-analysis notes, Loop Engineering
  feedback, previous results, and human feedback are appended to the user turn
  and labelled as data, never promoted to system instructions.
- **Policy sits below the model.** The policy middleware is always innermost,
  directly around the tool. A middleware that retries tool calls re-enters the
  policy, where completed consequential actions are replayed rather than
  executed twice.
- **Routing narrows, policy authorizes.** Tools that were not routed to a
  request are neither offered nor executed, but whether a call may run is
  decided only by the policy, for model calls and `execute_capability` alike.

## Runtime properties

- **Sync and async parity.** Every operation has an async twin. Agents
  configured with `mcp=`, `refresh=`, or `feedback=` require the async API,
  because those components are bound to an event loop.
- **Compiled-graph cache.** Topology graphs are compiled once per
  configuration and reused across requests.
- **Conversation memory.** With a checkpointer, each `thread_id` keeps its
  conversation automatically: send only the new user message on every turn.
- **Checkpoint safety.** Agloom checkpoints only JSON-compatible values and
  LangChain messages, so it runs with LangGraph's strict serialization
  (`LANGGRAPH_STRICT_MSGPACK=true`).
- **Cooperative cancellation.** `agent.stop()` cancels every active invocation
  of the agent at the next model call, tool call, or graph step.
- **Failure isolation.** An observer that raises is logged and skipped; it
  never changes the outcome of a run.

## Extension points

| Extend | With |
|---|---|
| How DIRECT or a topology is chosen | `strategy=` implementing [`Strategy`][agloom.Strategy] |
| How a topology graph is built | `compiler=ArchitectureCompiler(topologies={...})` with your [`Topology`][agloom.Topology] builders |
| Which capabilities a request sees | `capability_router=`, `capability_registry=`, or a [`CapabilityFilter`][agloom.CapabilityFilter] |
| How loop iterations are judged and decided | `loop_observer=`, `loop_evaluator=`, `loop_controller=` |
| Model and tool behavior | Any LangChain `AgentMiddleware` in `middleware=` |
| Telemetry | `observers=` with an [`Observer`][agloom.Observer] or any callable |
