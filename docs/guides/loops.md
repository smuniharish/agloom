# Loops

This guide configures [Loop Engineering](../concepts/loop-engineering.md) for
real workloads: budgets, evaluators, adaptation, refresh, human feedback, and
troubleshooting.

## Turn on iterations

```python
from agloom import LoopPolicy, create_agent

agent = create_agent(
    model=model,
    tools=[order_status],
    loop_policy=LoopPolicy(
        max_iterations=4,
        max_duration=120,
        max_tool_calls=40,
        allow_retry=True,
        max_retries=1,
        allow_replan=True,
        max_replans=1,
    ),
)
result = agent.invoke_loop(
    "Summarize the delivery status of orders A-1001 and A-1002 for the customer."
)
print(result.result.content)
print(result.state.iteration, result.state.termination_reason)
for entry in result.state.history:
    print(entry.observation.iteration, entry.decision.action, entry.decision.reason)
```

`invoke` and `ainvoke` return only the answer; `invoke_loop` and
`ainvoke_loop` return a [`LoopResult`][agloom.LoopResult] with the answer and
the typed [`LoopState`][agloom.LoopState]: phase, iteration, budget usage,
objective and progress status, and bounded history.

## Choose how iterations are judged

**Model judge (default for multi-iteration policies).** Add trusted success
criteria that the judge applies to every result. The judge only recommends
next steps your `LoopPolicy` allows, and explicit signals from your
application always take precedence over its assessment:

```python
from agloom import ModelLoopEvaluator

agent = create_agent(
    model=model,
    tools=[order_status],
    loop_policy=LoopPolicy(max_iterations=3),
    loop_evaluator=ModelLoopEvaluator(
        model, criteria="Mention every order id and its delivery date."
    ),
)
```

**Domain evaluator (deterministic).** When the objective is checkable, check
it in code. Report progress and feedback, and let the deterministic rules do
the rest:

```python
from agloom import DefaultLoopEvaluator


class RequiredFacts:
    """Complete when the answer mentions every required fact."""

    def __init__(self, *facts: str) -> None:
        self.facts = facts
        self.rules = DefaultLoopEvaluator()

    def evaluate(self, state, observation, policy):
        text = observation.result.text if observation.result is not None else ""
        missing = [fact for fact in self.facts if fact.casefold() not in text.casefold()]
        enriched = observation.model_copy(
            update={
                "objective_progress": 1 - len(missing) / len(self.facts),
                "feedback": tuple(f"Include {fact}." for fact in missing),
            }
        )
        return self.rules.evaluate(state, enriched, policy)


agent = create_agent(
    model=model,
    tools=[order_status],
    loop_policy=LoopPolicy(max_iterations=3),
    loop_evaluator=RequiredFacts("A-1001", "A-1002", "8 October"),
)
result = agent.invoke_loop("Where are my orders A-1001 and A-1002?")
print(result.state.iteration, result.state.termination_reason)
```

Progress of 1.0 completes the loop with `objective_satisfied` (an explicit
`objective_complete` or `validation_passed` signal, when reported, decides
instead); the feedback of an incomplete iteration reaches the next one as
data. Implement `aevaluate` as
well (see [`AsyncLoopEvaluator`][agloom.AsyncLoopEvaluator]) when your check
does I/O.

## Adapt within the policy

| Adaptation | Enable with | Triggered by |
|---|---|---|
| Retry | `allow_retry=True`, `max_retries=n` | A failed iteration, or an evaluator recommending `retry` |
| Replan | `allow_replan=True`, `max_replans=n` | An evaluator recommending `replan`, a `replan_required` signal, or stagnation |
| Refresh | `allow_refresh=True` and `refresh=RefreshEngine(...)` | An evaluator recommending `refresh`, a `refresh_required` signal, or stagnation |
| Topology change | `allow_topology_change=True` | A `requested_topology` signal |
| Human intervention | `allow_human_intervention=True` and a checkpointer | An evaluator, a signal, a behavioral intervention, or stagnation |

Topology change needs automatic selection or an allow-list: with an
allow-list, only listed topologies can be requested, and `create_agent` rejects
`allow_topology_change=True` together with a single explicit `pattern`
([`ConfigurationError`][agloom.ConfigurationError]).

## Refresh stale sources

Give the agent a [refresh-engine](https://pypi.org/project/refresh-engine/)
`RefreshEngine` and allow refresh. When the loop decides to refresh, it runs
the engine before the next iteration and emits `LoopRefreshed`:

```python
from refresh_engine import RefreshEngine

agent = create_agent(
    model=model,
    tools=[search_docs],
    refresh=RefreshEngine(source, operation),
    loop_policy=LoopPolicy(max_iterations=3, allow_refresh=True),
)
result = await agent.ainvoke_loop("Answer from the latest pricing documents.")
```

Agents with `refresh=` use the async API.

??? example "examples/30_loop_refresh.py"

    ```python
    --8<-- "examples/30_loop_refresh.py"
    ```

## Bring humans in

With a checkpointer and `allow_human_intervention=True`, the loop can pause
for a person and continue with their feedback:

```python
from agloom import ExecutionPaused

config = {"configurable": {"thread_id": "report-7"}}
try:
    agent.invoke("Draft the quarterly risk report.", config)
except ExecutionPaused:
    answer = agent.resume(config, "Use the Q3 figures and add a mitigation table.")
```

For asynchronous feedback that arrives while work continues, pass a
[feedback-manager](https://pypi.org/project/feedback-manager/) `FeedbackManager`
as `feedback=`. Feedback submitted for the thread is acknowledged and, like
resume feedback, applied to every later iteration (the latest 10 notes, up to
2,000 characters each); evaluator feedback is recorded for analysis.

??? example "examples/37_loop_feedback.py"

    ```python
    --8<-- "examples/37_loop_feedback.py"
    ```

## Persist long-running loops

With a checkpointer, every iteration is a checkpointed step. A paused run
resumes where it stopped, and a run interrupted by a crash continues with
`agent.resume(config)` (or `agent.invoke_loop(None, config)`) without
repeating completed iterations. Conversation memory per `thread_id` is kept
across invocations.

??? example "examples/34_loop_checkpoint_resume.py"

    ```python
    --8<-- "examples/34_loop_checkpoint_resume.py"
    ```

## Observe loops

- `LoopStarted`, `LoopIterationStarted`, `LoopIterationCompleted` (with the
  action, reason, and progress), `LoopRefreshed`, `LoopPaused`,
  `LoopCompleted`, and `LoopTerminated` events.
- Prometheus counters `loop_iterations_total{progress,topology}`,
  `loop_decisions_total{action}`, and `loop_terminations_total{reason}`.

## Troubleshooting

| Symptom | What to do |
|---|---|
| The loop never completes | Give the evaluator a checkable objective: `criteria=`, a domain evaluator, or explicit signals |
| Iterations repeat the same result | Stagnation ends it after `stagnation_threshold` iterations; enable replan or improve feedback |
| Approaches alternate without progress | Oscillation detection stops it; keep topology change disabled unless needed |
| Too many retries | Retry only transient failures; lower `max_retries` |
| An unexpected stop | Read `result.state.termination_reason` and the last `decision.reason` |
| A long history | `history_limit` bounds the retained entries |

## Example catalog

| Example | Shows |
|---|---|
| [25_loop_automatic_completion.py](https://github.com/smuniharish/agloom/blob/master/examples/25_loop_automatic_completion.py) | The one-shot default |
| [26_loop_multi_iteration.py](https://github.com/smuniharish/agloom/blob/master/examples/26_loop_multi_iteration.py) | Several evaluated iterations |
| [27_loop_progress.py](https://github.com/smuniharish/agloom/blob/master/examples/27_loop_progress.py) | Objective progress signals |
| [28_loop_retry.py](https://github.com/smuniharish/agloom/blob/master/examples/28_loop_retry.py) | Retrying failed iterations |
| [29_loop_replan.py](https://github.com/smuniharish/agloom/blob/master/examples/29_loop_replan.py) | Replanning |
| [30_loop_refresh.py](https://github.com/smuniharish/agloom/blob/master/examples/30_loop_refresh.py) | Refreshing sources |
| [31_loop_stagnation.py](https://github.com/smuniharish/agloom/blob/master/examples/31_loop_stagnation.py) | Stagnation detection |
| [32_loop_budget.py](https://github.com/smuniharish/agloom/blob/master/examples/32_loop_budget.py) | Budgets |
| [33_loop_human_intervention.py](https://github.com/smuniharish/agloom/blob/master/examples/33_loop_human_intervention.py) | Pausing for a human |
| [34_loop_checkpoint_resume.py](https://github.com/smuniharish/agloom/blob/master/examples/34_loop_checkpoint_resume.py) | Checkpointed iterations |
| [35_loop_recursive.py](https://github.com/smuniharish/agloom/blob/master/examples/35_loop_recursive.py) | Loops around recursive execution |
| [36_loop_topology_adaptation.py](https://github.com/smuniharish/agloom/blob/master/examples/36_loop_topology_adaptation.py) | Changing topology between iterations |
| [37_loop_feedback.py](https://github.com/smuniharish/agloom/blob/master/examples/37_loop_feedback.py) | Human and evaluator feedback |
| [38_loop_context.py](https://github.com/smuniharish/agloom/blob/master/examples/38_loop_context.py) | The context carried between iterations |
| [39_loop_long_running.py](https://github.com/smuniharish/agloom/blob/master/examples/39_loop_long_running.py) | A long-running objective |
