# Loop Engineering

Every invocation runs inside a loop: execute, observe, evaluate, decide. The
default `LoopPolicy()` is one iteration with no extra model calls. Raise
`max_iterations` to let the agent iterate until the objective is met or a
budget runs out.

```python
from agloom import LoopPolicy, ModelLoopEvaluator

agent = create_agent(
    model=model, tools=tools,
    loop_policy=LoopPolicy(max_iterations=5, allow_retry=True, max_retries=2,
                           allow_replan=True, max_replans=1),
    loop_evaluator=ModelLoopEvaluator(model, criteria="Cite a source for every claim."),
)
result = await agent.ainvoke_loop("Summarize the incident with sources")
result.result                       # final AIMessage, or None when paused
result.state.termination_reason     # for example objective_satisfied
result.state.history                # one entry per iteration
```

`invoke` and `ainvoke` run the same loop and return only the answer.

## `LoopPolicy`

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | `1` | Iterations before the loop stops. |
| `max_duration` | `300.0` | Active seconds, checked when each iteration ends. Waiting for people is not counted. |
| `max_tool_calls` | `1000` | Tool calls across all iterations. |
| `max_retries`, `max_replans` | `0` | Retry and replan budgets. |
| `allow_retry`, `allow_replan`, `allow_refresh`, `allow_topology_change`, `allow_human_intervention` | `False` | Permitted adaptations. |
| `stagnation_threshold` | `3` | Iterations without progress that count as stagnation. |
| `oscillation_window` | `4` | Recent decisions inspected for alternation (even number). |
| `progress_threshold` | `0.01` | Minimum progress change that counts. |
| `quality_threshold` | `0.8` | Minimum quality for convergence. |
| `complete_on_convergence` | `True` | Whether repeated high-quality results complete the loop. |
| `history_limit` | `20` | History entries kept in the state. |
| `max_context_chars` | `4000` | Previous-result characters carried into the next iteration. |

An iteration in progress is not interrupted when `max_duration` passes; set
timeouts on the chat model and tools for per-call deadlines.

## Evaluating results

- With `max_iterations > 1` and no custom observer or evaluator,
  `ModelLoopEvaluator(model)` judges each result against the objective. It only
  recommends steps the policy allows, and its feedback is carried into the next
  iteration.
- `DefaultLoopEvaluator()` makes no model calls. It reads explicit signals
  from the result, so use it when the application can judge results itself.
- Custom evaluators implement `evaluate(state, observation, policy)`. Add
  `aevaluate` to evaluate natively in async loops.

Results carry signals in `AIMessage.additional_kwargs["loop"]` or in a mapping
result: `objective_progress`, `objective_complete`, `quality`,
`validation_passed`, `retryable`, `refresh_required`, `replan_required`,
`human_intervention_required`, `requested_topology`, `tool_calls`, and
`feedback`. Explicit completion signals win: `objective_complete`, then
`validation_passed`. Progress of 1.0 completes the objective only when neither
is reported, and the model evaluator never overrides them.

## Deciding the next step

`DefaultLoopController` decides in this order:

1. A failed iteration: an exhausted tool budget stops the loop, a policy
   denial stops it with `policy_violation`, an unknown action outcome or a
   store error fails it, and other errors are retried when `allow_retry` is
   set and retries remain.
2. A completed objective, or converged results, completes the loop.
3. Exhausted duration or tool-call budgets stop it.
4. Behavior signals, then the evaluator's recommendation (retry, replan,
   refresh, topology change, human intervention), apply when the policy
   permits them. Retry and replan hints that are not permitted continue
   instead; other unpermitted adaptations stop the loop.
5. Stagnation, oscillation, and the iteration budget end the loop.

`state.termination_reason` reports the outcome: `objective_satisfied`,
`validated`, `converged`, `one_shot_complete`, `explicit_stop`,
`iteration_limit`, `timeout`, `tool_call_limit`, `retry_limit`,
`replan_limit`, `stagnation`, `oscillation`, `policy_violation`,
`human_intervention`, `behavior_intervention`, or `execution_failed`.

## People in the loop

- With `allow_human_intervention=True` and a checkpointer, the loop can pause
  for a person (for example when a result asks for help, an evaluator
  recommends it, or progress stagnates) with a `LOOP_INTERRUPT` payload. Resume
  with `agent.resume(config, "feedback text")`.
- With `feedback=FeedbackManager(...)` (async only), notes targeted at the
  thread apply from the next iteration on.
- Human feedback is standing: it applies to every later iteration (the last
  10 notes, up to 2,000 characters each).

## Refresh and adaptation

- `refresh=RefreshEngine(source, operation)` with `allow_refresh=True` lets the
  loop reload stale sources before continuing (async only).
- `allow_topology_change=True` lets the loop switch topology; it requires
  automatic selection or an allow-list.
- `behavior_signals` maps BehaviorWeave decisions into the loop: stop ends it,
  pause waits, human review asks a person, redirect replans, and retry retries.

## Budgets across pauses

Usage is recorded when an iteration finishes. An iteration resumed after an
approval, a breakpoint, or a crash counts its own tool calls and active time,
and continues with the tool-call budget that remained when the iteration
started.
