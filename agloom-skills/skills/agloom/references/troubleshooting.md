# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConfigurationError: this agent uses asynchronous integrations` | `mcp=`, `feedback=`, or `refresh=` with a sync call | Use `ainvoke`, `ainvoke_loop`, `astream`, or `aresume`. |
| `ConfigurationError` about a checkpoint authorizer | A checkpointer without `checkpoint_authorizer` | Pass a function that confirms the caller owns the `thread_id`. |
| `ConfigurationError` about breakpoints | `interrupt_before`/`interrupt_after` without exactly one explicit `pattern` | Pin one topology and use its node names (see topologies.md). |
| `ConfigurationError` requiring structured output | Automatic selection or `ModelLoopEvaluator` with a model that lacks structured output | Use a model with tool calling, or pin `pattern="react"` and use `DefaultLoopEvaluator`. |
| `ExecutionPaused` | An approval, an unknown action outcome, a tool-call review, a loop waiting for a person, or a breakpoint | Inspect `.interrupts[0].value` (its `type` for Agloom pauses), act (decide, reconcile, review, or give feedback), then `agent.resume(config, value)`. |
| `ApprovalRequired` (agentbarrier) | An approval rule matched, and the agent has no checkpointer | Add a checkpointer so approvals pause and resume. |
| `PolicyViolationError` | The policy denied a tool call | Add an allowing rule, or use `on_denied="inform_model"` so the model can adjust. |
| `PolicyConfigurationError` | An approval rule for a non-consequential tool, or a consequential tool that returned a graph `Command` | Declare the tool in `consequential`; return plain JSON-compatible results. |
| `ConfigurationError: the policy cannot judge the arguments of ...` | A governed tool without a usable schema: a schema class that is not Pydantic v2, arguments with no JSON form, or no `args_schema` and a `_run` with `*args`, `**kwargs`, or a `RunnableConfig` parameter (for example a plain `Tool`) | Declare the arguments with a Pydantic v2 model or a JSON schema of JSON values, for example with `@tool` or `StructuredTool`. |
| `ConfigurationError: the idempotency key ... must be a string argument` | The key names no argument, or a non-string one | Use a string argument that identifies the real-world action, or a callable selector. |
| The model receives "was not executed ... Invalid arguments" | The call did not match the tool's schema, or its idempotency key was blank | Nothing ran; the model can correct the call. Watch `ToolCallDenied` events with reason `InvalidArguments`. |
| `ActionInProgress` (agentbarrier) from `execute_capability` or `policy.execute` | The same action is already running | Report the request as in progress, or retry after it finishes to get the replayed result. |
| An agent inside another graph fails on approval with `ApprovalRequired` | The agent has no checkpointer | Give the agent the enclosing graph's checkpointer and pass the node's `config` to `agent.invoke`. |
| Loop evaluations run again after resuming an agent nested in another graph | LangGraph re-runs finished work of a nested graph's interrupted step (langchain-ai/langgraph#9106) | Plans, topology results, and consequential actions stay stable; run multi-iteration loops that pause as the top-level graph. |
| `ActionOutcomeUnknown` (agentbarrier) | A consequential tool raised and the agent has no checkpointer, or a direct `execute_capability` call failed | Check the external system, then `store.reconcile(action.action_id, ...)`; add a checkpointer so runs pause instead. |
| A tool is never offered | The policy denies it for every argument, or capability routing filtered it out | Check the rules; call `agent.route_capabilities(task)` to see the selection. |
| No MCP tools appear | The server was registered without discovery | Register with `refresh=RefreshPolicy(on_register=True)`. |
| `RecursionLimitError` | `ExecutionLimits` depth, child, or time budget reached | Raise the limit or simplify the plan; the budget covers the whole invocation. |
| `GraphRecursionError` inside `ExecutionError` | A `recursion_limit` was set in the config or inherited from an enclosing LCEL chain | Raise `recursion_limit` in the config you pass. |
| `TopologySelectionError` | No permitted topology can run the task, for example Supervisor without workers | Provide workers or stages, or widen the allow-list. |
| The loop stops early with `policy_violation` | A result requested an adaptation the policy forbids (refresh, topology change, human intervention) | Enable it in `LoopPolicy` or stop requesting it. |
| The loop ran past `max_duration` | The duration is checked when each iteration ends | Set timeouts on the chat model and tools. |
| A provider rejects `name` on assistant messages | Messages from another framework carry agent names | Agloom never sets message names; check the application's own messages. |

## Debugging steps

1. Add `observers=[LoggingObserver()]`, or a callable that prints each
   `AgentEvent`, to see routing, selection, policy outcomes, and loop
   decisions.
2. Stream with `stream_mode="custom"` to watch Agloom events live, or
   `"messages"` for model tokens.
3. Reproduce with a scripted LangChain chat model and an explicit `pattern` to
   rule out selection.
4. With a checkpointer, `agent.get_state(config)` shows the thread's loop
   state.
