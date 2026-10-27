# Errors

Every Agloom exception derives from [`AgloomError`][agloom.AgloomError].
LangGraph interrupts and agentbarrier runtime exceptions are never wrapped, so
you handle them with their own types.

## Hierarchy

```text
AgloomError
├── ConfigurationError            (also a ValueError)
│   └── PolicyConfigurationError
├── ExecutionPaused
├── PolicyViolationError
├── HarnessError
│   ├── ExecutionError
│   └── CancellationError
├── RecursionLimitError
├── TopologyError
│   ├── TopologySelectionError
│   └── CompilationError
├── CapabilityError
│   └── CapabilityResolutionError
└── LoopError
    ├── LoopBudgetExceededError
    ├── LoopDecisionError
    ├── LoopEvaluationError
    └── LoopStateError
```

## What each error means

| Error | Raised when | Typical response |
|---|---|---|
| [`ConfigurationError`][agloom.ConfigurationError] | `create_agent` arguments or an invocation input are invalid, for example a checkpointer without `checkpoint_authorizer`, a governed tool whose arguments the policy cannot judge, an unauthorized thread, or system messages in the input | Fix the configuration or reject the request |
| [`PolicyConfigurationError`][agloom.PolicyConfigurationError] | A policy decision cannot be enforced, for example an approval rule for a tool that is not declared consequential, or an idempotency key that a direct `execute` call or a key selector cannot provide | Fix the policy or the `consequential=` mapping |
| [`ExecutionPaused`][agloom.ExecutionPaused] | `invoke` stopped at an interrupt: an approval, a reconciliation, a tool-call review, a breakpoint, or a loop pause | Inspect `.interrupts`, act, then `agent.resume(config)` |
| [`PolicyViolationError`][agloom.PolicyViolationError] | The policy denied a tool call (`.tool_name`, `.rule`) | Report the denial; do not retry blindly |
| [`ExecutionError`][agloom.ExecutionError] | An execution failed, for example because a model or tool raised or a Planner plan was still invalid after its correction round; an original non-Agloom exception is chained as `__cause__` | Inspect the cause; retry transient failures |
| [`CancellationError`][agloom.CancellationError] | `agent.stop()` cancelled the run | Treat as an intentional stop |
| [`RecursionLimitError`][agloom.RecursionLimitError] | A recursion depth, child, or time limit was exhausted | Raise the limit or simplify the task |
| [`TopologySelectionError`][agloom.TopologySelectionError] | Task analysis was invalid, or the selected topology cannot run (for example, no workers) | Configure workers or stages, or pin a topology |
| [`CompilationError`][agloom.CompilationError] | A topology graph could not be built, for example an unknown breakpoint node | Fix the topology configuration |
| [`CapabilityResolutionError`][agloom.CapabilityResolutionError] | A capability executed from code is not registered or cannot be executed, for example an asynchronous capability called with `execute_capability`, or, under a policy, a tool called with arguments that do not match its schema. Requests for actions that no tool can perform do not raise it: the answer says what cannot be done | Check the name and arguments, or use `aexecute_capability` |
| [`LoopBudgetExceededError`][agloom.LoopBudgetExceededError] | The loop's tool-call budget was exhausted before a tool ran | Raise `max_tool_calls` or narrow the task |
| [`LoopDecisionError`][agloom.LoopDecisionError], [`LoopEvaluationError`][agloom.LoopEvaluationError], [`LoopStateError`][agloom.LoopStateError] | A custom loop controller, evaluator, or state transition failed, or a result carried malformed loop signals (`LoopEvaluationError`) | Fix the custom loop component or the signals your topology reports |

## How `invoke` reports outcomes

- A pause raises [`ExecutionPaused`][agloom.ExecutionPaused] with the pending
  `interrupts` and the `thread_id` to resume.
- A failure raises the original exception when it is an Agloom error or an
  agentbarrier runtime error, such as `ApprovalRequired` (an approval without a
  checkpointer), `ApprovalRejected`, or `ActionOutcomeUnknown`.
- Any other failure raises [`ExecutionError`][agloom.ExecutionError] chained to
  the original exception.

A consequential tool that raises leaves its action with an unknown outcome.
With a checkpointer, the run pauses with a
[`RECONCILIATION_INTERRUPT`][agloom.RECONCILIATION_INTERRUPT] until an
operator reconciles the action. Without one, the run raises
`ActionOutcomeUnknown`, chained from the tool's exception, and later calls with
the same idempotency key raise it too until the action is reconciled. See
[Failures and reconciliation](policies.md#failures-and-reconciliation).

`invoke_loop` and `ainvoke_loop` do not raise for pauses or failed
iterations. They return a [`LoopResult`][agloom.LoopResult] with `interrupts`,
`error`, and the full loop state, so you can handle those outcomes as data.
Invalid input, unauthorized threads, cancellation, and failures of custom loop
components still raise.

```python
result = agent.invoke_loop(task, config)
if result.paused:
    notify_reviewers(result.interrupts)
elif result.error is not None:
    log_failure(result.error, result.state.termination_reason)
else:
    reply(result.result)
```

## Handling errors

```python
from agentbarrier.errors import RuntimeBarrierError

from agloom import ExecutionPaused, PolicyViolationError

try:
    answer = agent.invoke(task, config)
except ExecutionPaused as paused:
    record_pending(paused.thread_id, paused.interrupts)
except PolicyViolationError as denied:
    reply(f"That action is not permitted ({denied.tool_name}).")
except RuntimeBarrierError as barrier_error:
    escalate(barrier_error)
```

Catch the narrowest type you can act on. Configuration errors indicate a bug
in your setup and should surface during development.
