# API reference

Everything below is imported from `agloom`. The complete reference with
docstrings is at <https://agloom.readthedocs.io/en/latest/api/>.

## `create_agent()`

All arguments are keyword-only. Only `model` is required.

| Argument | Type | Purpose |
|---|---|---|
| `model` | `BaseChatModel` | Any LangChain chat model. Tools need tool calling; automatic selection and the routing topologies need structured output. |
| `tools` | LangChain tools or annotated functions | Tools the agent may call. |
| `system_prompt` | `str` | Trusted instructions for every model call that executes the request (task analysis and loop evaluation use their own). |
| `pattern` | name, `TopologyName`, or a list of them | Omit for automatic selection; one name forces it; a list is an allow-list. |
| `composition` | list of 2-4 names | Hybrid children; valid only when Hybrid can be selected, and required when `pattern` names Hybrid. |
| `workers` | `Sequence[WorkerSpec]` | Specialists for Supervisor, Swarm, and Blackboard. |
| `stages` | `Sequence[PipelineStage]` | Ordered stages for Pipeline. |
| `max_reflections` | `int`, 1-5 | Review rounds for Reflection. |
| `limits` | `ExecutionLimits` | Recursion, fan-out, and time bounds. |
| `capabilities` | `Mapping[str, Any]` | Application services exposed as capabilities. |
| `capability_registry` | `CapabilityRegistry` | Replaces the in-memory registry. |
| `capability_router` | `CapabilityRouter` | Chooses the tools offered per task; see `DefaultCapabilityRouter`. |
| `mcp` | `mcp_capability_router.MCPRuntime` | Application-owned MCP runtime. Async only. |
| `middleware` | LangChain `AgentMiddleware` list | Applied to every tool-capable agent Agloom builds. |
| `policy` | `PolicyManager` | Governs every tool call. |
| `checkpointer` | LangGraph `BaseCheckpointSaver` | Persistence, memory, pauses, and resume. |
| `checkpoint_authorizer` | `Callable[[str], bool]` | Required with a checkpointer; authorizes each thread ID. |
| `interrupt_before`, `interrupt_after` | node names | Static breakpoints; need one explicit `pattern` and a checkpointer. |
| `loop_policy` | `LoopPolicy` | Loop Engineering budgets and permissions. |
| `loop_observer`, `loop_evaluator`, `loop_controller` | protocols | Replace the default loop components. |
| `feedback` | `feedback_manager.FeedbackManager` | Human feedback for loops. Async only. |
| `refresh` | `refresh_engine.RefreshEngine` | Source refresh for loops. Async only. |
| `explainability` | `langgraph_xai.XAIRuntime` | Opt-in decision traces. |
| `observers` | `Observer` or `Callable[[AgentEvent], object]` | Receive runtime events. |
| `strategy`, `compiler` | `Strategy`, `ArchitectureCompiler` | Extension points for selection and graph compilation. |

`create_agent()` validates the whole configuration and raises
`ConfigurationError` for invalid combinations, such as duplicate tool names,
a checkpointer without an authorizer, breakpoints without one explicit
pattern, or a policy-governed tool whose arguments the policy cannot judge.

## `Agent`

`Agent` is a LangChain `Runnable`, so it works in LCEL compositions and with
`batch`, `with_config`, and callbacks.

| Method | Returns | Notes |
|---|---|---|
| `invoke(input, config=None, *, context=None)` | `AIMessage` or mapping | Raises `ExecutionPaused` when the run pauses. |
| `ainvoke(...)` | same | Async. |
| `invoke_loop(input, config=None, *, context=None)` | `LoopResult` | `result`, `state`, `interrupts`, `error`, and `.paused`. |
| `ainvoke_loop(...)` | `LoopResult` | Async. |
| `stream(input, config=None, *, stream_mode="updates", subgraphs=False)` | iterator | `"messages"` for tokens, `"custom"` for Agloom events. |
| `astream(...)` | async iterator | Async. |
| `resume(config, value=True)` | like `invoke` | Answers pending interrupts or continues a stopped thread. |
| `aresume(...)` | same | Async. |
| `get_state(config)`, `aget_state(config)` | LangGraph `StateSnapshot` | Needs a checkpointer and an authorized thread. |
| `stop()` | `None` | Cooperative cancellation; the run raises `CancellationError`. |
| `route_capabilities(task)`, `aroute_capabilities(task)` | `CapabilitySelection` | The capabilities a task would receive. |
| `execute_capability(name, input=None)`, `aexecute_capability(...)` | tool result | Policy-enforced direct execution. |
| `with_structured_output(schema, **options)` | `Runnable` | Validated output through xstructured. |
| `flush()`, `close()` | `None` | Flush observers; release resources. |

Properties: `id`, `capabilities`, `policy`, `loop_policy`, `limits`,
`feedback`, `xai`.

## Configuration models

`ExecutionLimits`:

| Field | Default | Range |
|---|---|---|
| `max_depth` | `0` (recursion off) | 0-8 |
| `max_children` | `16` | 1-256 |
| `max_workers` | `4` | 1-32 |
| `max_plan_steps` | `8` | 1-64 |
| `max_handoffs` | `8` | 1-64 |
| `max_execution_time` | `120.0` seconds | up to 3600 |
| `allowed_nested_topologies` | all eight | topologies children may select |

`WorkerSpec(name, description, instructions="")` and
`PipelineStage(name, instruction, transform=None)` describe workers and
stages. Names are 1-80 characters. `FINISH` and `STOP` are reserved worker
names.

`LoopPolicy` is described in [loop-engineering.md](loop-engineering.md).

## Interrupt types

| Constant | Value | Raised for |
|---|---|---|
| `APPROVAL_INTERRUPT` | `"agloom.approval"` | A Policy Manager approval |
| `RECONCILIATION_INTERRUPT` | `"agloom.reconciliation"` | A consequential action with an unknown outcome |
| `LOOP_INTERRUPT` | `"agloom.loop"` | A loop waiting for human intervention |
| `BREAKPOINT_INTERRUPT` | `"agloom.breakpoint"` | A static breakpoint |

## Errors

All errors derive from `AgloomError`.

| Error | Meaning |
|---|---|
| `ConfigurationError` | Invalid configuration, or a sync call on an async-only agent. |
| `ExecutionPaused` | The run paused; inspect `.interrupts` and `.thread_id`. |
| `ExecutionError` | The run failed; the cause is chained. |
| `CancellationError` | `stop()` cancelled the run. |
| `TopologySelectionError` | No permitted topology can run the task. |
| `CompilationError` | A topology graph could not be built. |
| `RecursionLimitError` | A depth, child, or time limit was reached. |
| `CapabilityResolutionError` | A capability is unknown, not executable, or async-only, or (with a policy) its tool arguments do not match the schema. |
| `PolicyViolationError` | The policy denied a tool call. |
| `PolicyConfigurationError` | The policy and tools disagree, for example an approval rule on a non-consequential tool. |
| `LoopEvaluationError`, `LoopDecisionError`, `LoopStateError`, `LoopBudgetExceededError` | Loop Engineering failures. |

agentbarrier runtime errors pass through unchanged, for example
`ApprovalRequired` (an approval without a checkpointer),
`ActionOutcomeUnknown` (a consequential action with an unknown outcome when
the agent has no checkpointer, or from direct `execute_capability` calls;
`.action` identifies it), and `ActionInProgress` (a direct call made while the
same action is still running). Other failures are raised as `ExecutionError`
with the cause chained.
