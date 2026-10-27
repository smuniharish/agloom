# Loop Engineering

**A successful execution is not the same as an achieved objective.** A run can
finish without errors and still miss a requirement, rely on stale data, or
need a different approach. Loop Engineering is Agloom's answer: a
deterministic **outer control loop** around any execution, DIRECT or any of the
eight topologies, that evaluates the result against the objective and decides,
within explicit bounds, whether to complete, try again, adapt, wait for a
human, or stop.

It is not a ninth topology, and it is not recursion. Every invocation runs
through it, and the default policy is **one-shot**: one iteration, no
evaluation call, no extra cost.

## Two loops, two jobs

![The inner model/tool loop inside the outer objective loop](../assets/diagrams/inner-outer-loop-light.png#only-light)
![The inner model/tool loop inside the outer objective loop](../assets/diagrams/inner-outer-loop-dark.png#only-dark)

| | Inner loop | Outer loop (Loop Engineering) |
|---|---|---|
| **Runs** | Inside one execution: model, tool, model, tool | Around whole executions: iteration, assessment, decision |
| **Decided by** | The model, call by call | A deterministic controller under a `LoopPolicy` |
| **Stops when** | The model stops calling tools, or a topology finishes | The objective is met, a budget is exhausted, or the policy says stop |
| **Remembers** | The conversation of one execution | Typed history of every iteration: observation, evaluation, decision |

The inner loop is good at doing work. The outer loop is good at judging whether
the work is done, and at keeping the process bounded and explainable.

## Why it matters

- **Objectives, not runs.** Each iteration is judged against the objective,
  with progress, quality, and completion, rather than on "no exception was
  raised".
- **Explicit budgets, auditable stops.** Iterations, active duration, tool
  calls, retries, and replans are budgeted. Every loop ends with a typed
  [termination reason](#termination-reasons), so you can answer "why did it
  stop?" from data.
- **Progress, stagnation, and oscillation detection.** Repeated identical
  results, flat progress, and alternating approaches are detected
  deterministically and ended or redirected, instead of burning budget.
- **Policy-bounded adaptation.** Retry, replan, refresh, topology change, and
  human intervention happen only when the `LoopPolicy` permits them, and only
  within their budgets. Requests to do something the policy forbids stop the
  loop instead of being silently honored.
- **Checkpointed, resumable iterations.** With a checkpointer, each iteration
  is a checkpointed step: a paused or interrupted run resumes without
  repeating completed iterations or duplicating their events.
- **Signals from the whole ecosystem.** Evaluator feedback, human feedback,
  behavioral interventions, refreshed sources, and pending approvals all feed
  the same decision.
- **Zero cost by default.** The one-shot policy adds no model calls. You opt
  into iteration where it pays off.

## Anatomy of an iteration

![Loop state machine](../assets/diagrams/loop-state-machine-light.png#only-light)
![Loop state machine](../assets/diagrams/loop-state-machine-dark.png#only-dark)

1. **Begin.** The iteration is counted against the budget; a retry or replan
   also consumes its own budget.
2. **Execute.** The request runs exactly as a one-shot request would: routing,
   selection, then DIRECT or a topology. From the second iteration on, the
   execution also receives the [iteration context](#what-the-next-iteration-sees).
3. **Observe.** A [`LoopObserver`][agloom.LoopObserver] turns the result or
   failure into a [`LoopObservation`][agloom.LoopObservation]: success, error
   type, duration, tool calls, explicit signals, and a fingerprint of the
   result.
4. **Evaluate.** A [`LoopEvaluator`][agloom.LoopEvaluator] interprets the
   observation against the objective: progress, regression, stagnation,
   convergence, objective status, feedback, and a recommended action.
5. **Decide.** A [`LoopController`][agloom.LoopController] chooses the next
   [`LoopAction`][agloom.LoopAction] within the policy and budgets, and the
   iteration is recorded in the loop history.

## Evaluators

| Evaluator | Model calls | How it judges |
|---|---|---|
| [`DefaultLoopEvaluator`][agloom.DefaultLoopEvaluator] | None | Deterministic: explicit signals only. `objective_complete` decides completion when reported, otherwise `validation_passed`, otherwise progress of 1.0; repeating an identical result counts as no progress |
| [`ModelLoopEvaluator`][agloom.ModelLoopEvaluator] | One structured call per successful iteration | A model returns a [`LoopAssessment`][agloom.LoopAssessment] (completion, progress, quality, feedback, next step); stagnation, regression, convergence, and streaks are then computed deterministically. Explicit signals from your application always win, and the judge only chooses next steps the `LoopPolicy` allows |

When `max_iterations` is greater than 1 and you configure neither an observer
nor an evaluator, Agloom uses a `ModelLoopEvaluator` with the agent's model.
Pass `ModelLoopEvaluator(model, criteria="...")` to add trusted success
criteria, or your own evaluator for domain checks such as tests, validators,
or coverage metrics.

## Decisions

| Action | Effect |
|---|---|
| `complete` | Stop: the objective is met |
| `continue` | Run another iteration that builds on the previous result |
| `retry` | Run again, avoiding the previous failure (needs `allow_retry` and retry budget) |
| `replan` | Run again with guidance to take a different approach (needs `allow_replan` and replan budget) |
| `refresh` | Refresh sources with the configured refresh engine, then run again (needs `allow_refresh`) |
| `change_topology` | Run again with a requested topology, within your `pattern` constraints (needs `allow_topology_change`) |
| `wait`, `human_intervention` | Pause for a human (needs `allow_human_intervention`, or a pending approval) |
| `stop` | Stop with a termination reason |
| `fail` | Stop as a failure |

The [`DefaultLoopController`][agloom.DefaultLoopController] decides in this
order:

1. **Failures first.** A pending approval or an unknown outcome to reconcile
   pauses a checkpointed run for a human; a policy denial stops with
   `policy_violation`; unrecoverable errors (such as an unknown outcome in a
   run without a checkpointer) fail; any other error is retried while
   `allow_retry` and the retry budget allow, and fails otherwise.
2. **Completion.** An achieved objective completes (`validated` when
   validation passed, otherwise `objective_satisfied`); converged results
   complete (`converged`) unless `complete_on_convergence=False`. Completion
   is decided before budgets, so an iteration that achieves the objective
   completes even if it exhausted the duration budget.
3. **Budgets.** Exhausted duration or tool-call budgets stop the loop.
4. **Signals.** Behavioral interventions, then evaluator recommendations, each
   applied only if the policy permits them.
5. **One-shot.** With `max_iterations=1`, a successful iteration completes
   (`one_shot_complete`).
6. **Progress rules.** Stagnation triggers a replan, a refresh, or a human, in
   that order of what the policy permits, and otherwise stops; oscillation
   stops; an exhausted iteration budget stops.
7. Otherwise, **continue**.

Retry and replan are quality hints: when the policy does not permit them, the
loop simply continues within its budget. Human intervention, refresh, and
topology change alter the safety posture: when a custom evaluator or a result
signal requests one that the policy does not permit, the loop stops with
`policy_violation`, and a behavioral pause or review that the policy does not
permit stops it with `behavior_intervention`. `ModelLoopEvaluator` offers its
judge only the next steps the policy allows (`retry`, `replan`, and `refresh`
need `allow_retry`, `allow_replan`, and `allow_refresh`) and ignores any other
suggestion, so the loop continues instead of failing on a judge's hint.

## Budget accounting

- **Tool calls** are checked before every tool runs, so a runaway execution
  stops at the limit rather than after it.
- **Duration** counts active execution and evaluation time, not time spent
  waiting for a human or an approval. It is checked when each iteration ends:
  an iteration in progress is not interrupted, so set timeouts on your chat
  model and tools for per-call deadlines.
- **Resumed iterations.** Usage is recorded when an iteration finishes. An
  iteration that pauses (for an approval or a breakpoint) or crashes and is
  later resumed counts only its own tool calls and active time from the
  resume, against the tool-call budget that remained when the iteration
  started; see [`LoopBudget`][agloom.LoopBudget].
- **Recursion** is bounded separately by
  [`ExecutionLimits`](limits.md): one recursion budget covers the whole
  invocation, including every iteration.

## Termination reasons

| Reason | Meaning |
|---|---|
| `objective_satisfied` | The evaluator judged the objective complete |
| `validated` | The result passed explicit validation |
| `converged` | Results stopped changing above the quality threshold |
| `one_shot_complete` | The one-shot execution succeeded |
| `explicit_stop` | The evaluator recommended stopping |
| `iteration_limit` | The iteration budget is exhausted |
| `timeout` | The active-duration budget is exhausted |
| `tool_call_limit` | The tool-call budget is exhausted |
| `retry_limit` | The retry budget is exhausted |
| `replan_limit` | The replan budget is exhausted |
| `stagnation` | Progress stagnated and no permitted adaptation remained |
| `oscillation` | Actions or topologies alternated without progress |
| `policy_violation` | A policy blocked a tool call, or the loop was asked to do something its policy forbids |
| `human_intervention` | The loop is waiting for, or ended waiting for, a human |
| `behavior_intervention` | A behavioral policy stopped the work |
| `execution_failed` | The iteration failed and was not retried |

Cancellation is not a termination reason: `agent.stop()` ends the run with
[`CancellationError`][agloom.CancellationError] and an `ExecutionCancelled`
event, and a checkpointed thread can continue with `agent.resume(config)`.

`result.state.history` keeps each iteration's observation, evaluation, and
decision (bounded by `history_limit`), and every decision is also emitted as a
`LoopIterationCompleted` event with its action and reason.

## What the next iteration sees

From the second iteration on, the execution receives a context block appended
to the user's request. It is explicitly labelled as data from earlier
attempts, not instructions, and contains:

- the next action and its reason, with guidance (for example "Take a
  different approach than the previous attempt");
- evaluator feedback;
- standing human feedback: the latest 10 notes (up to 2,000 characters each),
  applied to this and every later iteration;
- the previous error, if the iteration failed;
- the previous result, truncated to `max_context_chars`.

## Signals

| Source | How it reaches the loop |
|---|---|
| Evaluators | The evaluation's feedback and recommended action |
| Results | A `loop` mapping in a result (in a custom topology's output state, or `AIMessage.additional_kwargs["loop"]`) with keys such as `objective_progress`, `objective_complete`, `quality`, `validation_passed`, `retryable`, `refresh_required`, `replan_required`, `human_intervention_required`, `requested_topology`, `tool_calls`, and `feedback`; malformed signals raise [`LoopEvaluationError`][agloom.LoopEvaluationError] |
| Behavioral policies | BehaviorWeave decisions bridged by [`behavior_signals`][agloom.behavior_signals]: `stop` stops, `pause` waits, `human_review` and `escalate` request a human, `redirect` replans, `retry` retries; see [Integrations](middleware.md#behaviorweave) |
| Policy approvals and reconciliations | A pending approval, or an unknown outcome waiting for reconciliation, pauses a checkpointed loop for a human |
| Human feedback | With `feedback=FeedbackManager(...)`, feedback submitted for the thread is acknowledged and applied to every later iteration |
| Sources | With `refresh=RefreshEngine(...)` and `allow_refresh=True`, the `refresh` action refreshes sources before the next iteration |

## Pauses and resume

With a checkpointer, `wait` and `human_intervention` pause the run at a
LangGraph interrupt whose payload `type` is
[`LOOP_INTERRUPT`][agloom.LOOP_INTERRUPT]. Resume with human feedback, which
stays part of the context of every later iteration:

```python
from agloom import ExecutionPaused

try:
    agent.invoke(task, config)
except ExecutionPaused:
    agent.resume(config, "Use the Q3 figures, not Q2.")
```

Without a checkpointer the loop cannot pause, so it ends with the latest
result and the `human_intervention` termination reason. Time spent waiting
does not count against `max_duration`.

## When not to use multi-iteration loops

Keep the one-shot default when:

- requests are simple questions, lookups, or single transformations;
- latency matters more than marginal quality;
- there is no meaningful objective signal: without a validator or a clear
  rubric, a judge mostly adds cost;
- one answer only needs polishing; the [Reflection](topologies.md#reflection)
  topology improves an answer inside one execution, more cheaply.

Reach for iterations when the objective is checkable (tests, validators,
coverage, a rubric), when sources change during the work, or when a failed
approach should be replaced rather than repeated.

## Loop Engineering compared

| | Scope | Decides | Bounded by |
|---|---|---|---|
| Loop Engineering | Whole executions | A deterministic controller under a `LoopPolicy` | Iteration, duration, tool-call, retry, and replan budgets |
| [Reflection](topologies.md#reflection) | One answer inside one execution | A structured review | `max_reflections` |
| [Recursion](limits.md) | Child tasks of a Planner | Task analysis of each child | `ExecutionLimits` |
| Deep Agents `RubricMiddleware` (beta) | One agent's loop | A grader sub-agent | Its `max_iterations` |

Configure loops in practice with the [Loops guide](../guides/loops.md).
