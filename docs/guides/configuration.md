# Configuration

[`create_agent`][agloom.create_agent] takes keyword-only arguments; only
`model` is required. This guide walks through every parameter, grouped by
concern. Configuration is validated when the agent is created, so mistakes
fail fast with [`ConfigurationError`][agloom.ConfigurationError].

```python
from agloom import create_agent

agent = create_agent(model=model, tools=[order_status])
```

## Model and instructions

| Parameter | Default | Description |
|---|---|---|
| `model` | required | Any LangChain `BaseChatModel`. Tools need tool calling; automatic selection, the Supervisor, Swarm, Planner, and Reflection decisions, and the model loop evaluator need structured output. |
| `tools` | `()` | LangChain tools or annotated functions (converted with LangChain's `@tool`). Names must be unique. |
| `system_prompt` | `None` | Trusted instructions for every model call that executes the request: DIRECT answers, every worker, stage, and step, and the topologies' routing, planning, and review decisions. Task analysis and loop evaluation use their own instructions. |

## Execution selection

| Parameter | Default | Description |
|---|---|---|
| `pattern` | `None` | `None` for automatic selection, one topology name (`"planner"`) to pin it, or a list (`["planner", "supervisor"]`) as an allow-list. See [Execution selection](../concepts/execution-selection.md). |
| `composition` | `None` | Two to four distinct, non-Hybrid topologies for Hybrid, such as `["react", "reflection"]`. Valid only when Hybrid can be selected, and required when `pattern` pins Hybrid or lists it in an allow-list. |
| `strategy` | `StrategyEngine()` | A custom [`Strategy`][agloom.Strategy] mapping a task analysis to DIRECT or a topology. |
| `compiler` | `ArchitectureCompiler()` | A custom [`ArchitectureCompiler`][agloom.ArchitectureCompiler], for example with replacement topology builders. |

## Topology configuration

| Parameter | Default | Description |
|---|---|---|
| `workers` | `()` | [`WorkerSpec`][agloom.WorkerSpec] entries (`name`, `description`, `instructions`) for Supervisor, Swarm, and Blackboard. Names must be unique; `FINISH` and `STOP` are reserved. |
| `stages` | `()` | [`PipelineStage`][agloom.PipelineStage] entries (`name`, `instruction`, optional `transform`) for Pipeline. |
| `max_reflections` | `1` | Review-and-revise rounds for Reflection, from 1 to 5. |
| `limits` | `ExecutionLimits()` | Recursion, parallelism, plan length, handoffs, and wall-clock bounds. See [Execution limits and recursion](../concepts/limits.md). |

```python
from agloom import ExecutionLimits, PipelineStage, WorkerSpec

agent = create_agent(
    model=model,
    tools=tools,
    workers=[
        WorkerSpec(name="researcher", description="Finds and cites facts."),
        WorkerSpec(name="writer", description="Writes the final answer."),
    ],
    stages=[
        PipelineStage(name="outline", instruction="Outline the answer."),
        PipelineStage(name="write", instruction="Write the full answer."),
    ],
    max_reflections=2,
    limits=ExecutionLimits(max_handoffs=6, max_workers=3),
)
```

## Capabilities

| Parameter | Default | Description |
|---|---|---|
| `capabilities` | `None` | Application services by name: `{"policy_store": store}`. They appear in routing and analysis, and your code runs them with `execute_capability`. |
| `capability_router` | `DefaultCapabilityRouter()` | Routing per request: a [`CapabilityFilter`][agloom.CapabilityFilter], `embeddings`, `retriever`, `reranker`, and `max_results`. Calls to tools that were not offered are refused; routing is not authorization, which belongs to `policy`. |
| `capability_registry` | in-memory | A custom [`CapabilityRegistry`][agloom.CapabilityRegistry], for example one shared with other components. Agloom registers the agent's tools, middleware tools, and services in it. |
| `mcp` | `None` | An `MCPRuntime` from mcp-capability-router that your application owns and registers servers on, with `refresh=RefreshPolicy(on_register=True)` (or `await runtime.refresh()`) so that their capabilities are discovered. Requires the async API. |

See [Capabilities and MCP](../concepts/capabilities.md).

## Middleware

| Parameter | Default | Description |
|---|---|---|
| `middleware` | `()` | LangChain `AgentMiddleware` applied, in order, to every tool-capable agent and to DIRECT answers: ContextSage, BehaviorWeave, Deep Agents, PII redaction, retries, and more. Tools that middleware contributes are registered as capabilities and governed by `policy`. |

See [Integrations via LangChain middleware](../concepts/middleware.md).

## Policy

| Parameter | Default | Description |
|---|---|---|
| `policy` | `None` | A [`PolicyManager`][agloom.PolicyManager] enforcing an agentbarrier policy on every tool call, including middleware tools and `execute_capability` calls. Tools the policy always denies are removed; every `consequential=` tool must be configured (or come from MCP). |

See [Policies and approvals](../concepts/policies.md) and
[Writing policies](writing-policies.md).

## Persistence and human-in-the-loop

| Parameter | Default | Description |
|---|---|---|
| `checkpointer` | `None` | A LangGraph checkpointer. Enables conversation memory per `thread_id`, pauses, approvals that resume, and recovery from interruptions. |
| `checkpoint_authorizer` | `None` | `Callable[[str], bool]` that decides whether the current caller may use a `thread_id`. **Required** with a checkpointer; every invocation then needs `configurable.thread_id`. |
| `interrupt_before` | `()` | Topology node names to pause before. Requires a checkpointer and an explicit single `pattern`. |
| `interrupt_after` | `()` | Topology node names to pause after. Requires a checkpointer and an explicit single `pattern`. |

Node names are topology specific, which is why breakpoints need a pinned
topology. See [Topologies](../concepts/topologies.md) for node names and
[Human-in-the-loop and resume](human-in-the-loop.md).

## Loop Engineering

| Parameter | Default | Description |
|---|---|---|
| `loop_policy` | `LoopPolicy()` (one-shot) | Iteration, duration, tool-call, retry, and replan budgets, plus permitted adaptations. |
| `loop_observer` | `DefaultLoopObserver()` | Turns iteration results into observations. |
| `loop_evaluator` | see description | Judges observations. Defaults to a `ModelLoopEvaluator` when `max_iterations > 1` and no custom observer is set, otherwise to the deterministic `DefaultLoopEvaluator`. |
| `loop_controller` | `DefaultLoopController()` | Chooses the next action within the policy. |
| `feedback` | `None` | A `FeedbackManager` from feedback-manager: human feedback submitted for the thread applies to every later iteration, and evaluator feedback is recorded. Requires the async API. |
| `refresh` | `None` | A `RefreshEngine` from refresh-engine used by the `refresh` action. Required by `LoopPolicy(allow_refresh=True)`. Requires the async API. |

See [Loop Engineering](../concepts/loop-engineering.md) and the
[Loops guide](loops.md).

## Observability

| Parameter | Default | Description |
|---|---|---|
| `observers` | `()` | [`Observer`][agloom.Observer] instances or plain callables receiving every [`AgentEvent`][agloom.AgentEvent]. |
| `explainability` | `None` | An `XAIRuntime` from langgraph-xai instrumenting analysis, DIRECT, topologies, and the loop evaluator; available as `agent.xai`. |

See [Observability](../concepts/observability.md).

## Validation rules

`create_agent` raises [`ConfigurationError`][agloom.ConfigurationError] when,
among other checks:

- `model` is not a LangChain chat model, or `max_reflections` is outside 1 to 5;
- `pattern` names an unknown topology, is empty, or repeats a topology;
- `pattern` pins Hybrid or lists it in an allow-list without `composition`, or
  `composition` is given when Hybrid cannot be selected;
- tool, worker, stage, or capability names repeat, or a worker uses a reserved
  name;
- a `consequential=` tool of the policy is not configured and no `mcp` runtime
  is set;
- the policy governs a tool whose arguments it cannot judge, or a string
  idempotency key does not name an argument of the tool that can be a string
  (see [The arguments the policy judges](../concepts/policies.md#the-arguments-the-policy-judges));
- `interrupt_before` or `interrupt_after` is set without a checkpointer, or
  without an explicit single `pattern`;
- a checkpointer is set without `checkpoint_authorizer`;
- `LoopPolicy(allow_refresh=True)` is set without `refresh`;
- `LoopPolicy(allow_topology_change=True)` is combined with a single explicit
  `pattern` (topology change needs automatic selection or an allow-list).

A pinned topology is compiled immediately, so its errors also surface at
creation: a Supervisor without workers raises `ConfigurationError`, and an
unknown breakpoint node raises [`CompilationError`][agloom.CompilationError].

## Every parameter in one place

[`examples/24_complete_configuration.py`](https://github.com/smuniharish/agloom/blob/master/examples/24_complete_configuration.py)
constructs agents that together exercise every `create_agent` parameter.

??? example "examples/24_complete_configuration.py"

    ```python
    --8<-- "examples/24_complete_configuration.py"
    ```
