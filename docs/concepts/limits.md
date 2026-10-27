# Execution limits and recursion

Every execution runs under explicit, validated bounds. [`ExecutionLimits`][agloom.ExecutionLimits]
bounds the shape of one execution: recursion, fan-out, plans, and handoffs.
[`LoopPolicy`][agloom.LoopPolicy] bounds the outer loop: iterations, active
duration, tool calls, retries, and replans. Both are immutable and validated
when created.

![Recursion bounds](../assets/diagrams/recursion-bounds-light.png#only-light)
![Recursion bounds](../assets/diagrams/recursion-bounds-dark.png#only-dark)

## ExecutionLimits

| Field | Default | Range | Bounds |
|---|---|---|---|
| `max_depth` | `0` | 0 to 8 | Recursive child depth; `0` disables recursion |
| `max_children` | `16` | 1 to 256 | Recursive child executions in one invocation, across all loop iterations |
| `max_workers` | `4` | 1 to 32 | Branches running in parallel in one fan-out |
| `max_plan_steps` | `8` | 1 to 64 | Steps in one Planner plan |
| `max_handoffs` | `8` | 1 to 64 | Supervisor delegations or Swarm turns in one run |
| `max_execution_time` | `120.0` | up to 3600 s | Wall-clock time of recursive work in one invocation, from the start of the first child |
| `allowed_nested_topologies` | all eight | non-empty, unique | Topologies that recursive children may select |

```python
from agloom import ExecutionLimits, create_agent

agent = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
    limits=ExecutionLimits(
        max_depth=2,
        max_children=8,
        max_workers=4,
        max_execution_time=120,
        allowed_nested_topologies=("react", "reflection", "planner"),
    ),
)
```

## Parallelism

Planner waves and Blackboard contributors run in parallel, at most
`max_workers` at a time: a Planner wave starts at most `max_workers` ready
steps, and the rest start in the next wave; Blackboard contributors run in
groups of `max_workers`. Each fan-out is bounded on its own, so nested
fan-outs never wait on a shared pool.

A `max_concurrency` in the run configuration limits only how many inputs
`batch` runs at once; inside a run, `max_workers` alone bounds the parallel
branches.

## Plans and handoffs

- A Planner plan with more than `max_plan_steps` steps, duplicate ids, unknown
  dependencies, or a cycle gets one correction round with the validation
  error; if the corrected plan is still invalid, the execution fails with
  [`ExecutionError`][agloom.ExecutionError].
- Supervisor delegations stop at `max_handoffs`; the answer is then synthesized
  from the work completed so far. Swarm turns stop at `max_handoffs` the same
  way.

## Recursion

With `max_depth` greater than 0, **every Planner step runs as a recursive child
execution**: the step's instruction is routed, analyzed, and executed by the
same agent, DIRECT or in any topology from `allowed_nested_topologies`. A child
that selects Planner can recurse again until `max_depth` is reached.

- Children always use automatic selection within `allowed_nested_topologies`,
  even when the parent pins a topology; children may run DIRECT.
- One recursion budget covers the whole invocation, including every Loop
  Engineering iteration: at most `max_children` children, within
  `max_execution_time` seconds of the first child. Resuming a paused thread
  starts a new invocation with a fresh budget.
- With a checkpointer, every execution records its plan (routed capabilities
  and selected topology) in the checkpoint. A resumed run continues the plan it
  paused in, at every depth, instead of analyzing the task again.
- Children share the agent's capability catalog (each child routes the
  capabilities for its own step), middleware, policy, observers, and the
  loop's tool-call budget.
- When a depth, child, or time limit is exhausted, the execution fails with
  [`RecursionLimitError`][agloom.RecursionLimitError]. Loop Engineering then
  retries it if the `LoopPolicy` allows retries, and fails otherwise.

Each recursive child makes one task-analysis call, so recursion trades cost for
adaptivity: enable it for tasks whose steps genuinely differ in structure.

??? example "examples/11_recursion.py"

    ```python
    --8<-- "examples/11_recursion.py"
    ```

??? example "examples/12_no_recursion.py"

    ```python
    --8<-- "examples/12_no_recursion.py"
    ```

## Loop budgets

| `LoopPolicy` field | Default | Bounds |
|---|---|---|
| `max_iterations` | `1` | Iterations of the outer loop |
| `max_duration` | `300.0` s | Active execution and evaluation time, checked when each iteration ends; waiting for a human is not counted |
| `max_tool_calls` | `1000` | Tool calls across all iterations, workers, and children |
| `max_retries` | `0` | Retries after failures or evaluator recommendations |
| `max_replans` | `0` | Replans |

The tool-call budget is checked before every tool runs, so a runaway agent is
stopped at the limit rather than after it. The duration budget does not
interrupt an iteration in progress: set timeouts on your model client and
tools for hard per-call deadlines, and call `agent.stop()` to cancel the
agent's active runs.
Usage is recorded when an iteration
finishes. An iteration that pauses (for an approval or a breakpoint) or
crashes, and is then resumed, counts its own tool calls and active time from
the resume and continues with the tool-call budget that remained when the
iteration started; calls and time from before the pause are not added. See
[Loop Engineering](loop-engineering.md).

## LangGraph's recursion limit

LangGraph's `recursion_limit` caps the supersteps of one graph run. Agloom
does not impose a limit of its own: unless you (or an enclosing LangChain run)
set `recursion_limit` in the run configuration, LangGraph applies its own
default, exactly as for LangChain agents. Treat it as a last-resort backstop.
Agloom's limits are the ones to tune: they bound the work in domain terms and
fail with precise errors.
