# Execution modes and topologies

## How Agloom chooses

- **Automatic** (no `pattern`): one structured `TaskAnalysis` call describes
  the task. The strategy answers simple tasks directly (DIRECT) and otherwise
  picks the first topology whose needs the analysis matches.
- **Allow-list** (`pattern=["planner", "supervisor"]`): analysis runs, and one
  of the listed topologies always runs; DIRECT is not used.
- **Explicit** (`pattern="planner"`): no analysis call; that topology always
  runs. `TopologySelected` still reports it.
- **Hybrid** (`pattern="hybrid", composition=[...]`): the children run in
  order, each receiving the previous result.

When analysis lists requested actions that no tool can perform, the run still
proceeds. The model is told not to claim them and to say what cannot be done.
`TopologySelected.payload["missing_capabilities"]` reports how many there were.

## Topologies

| Topology | Use for | Configuration | Breakpoint node names |
|---|---|---|---|
| `react` | Tool use in a model-tool loop | `tools` | `model`, `tools` |
| `supervisor` | A coordinator delegating to specialists | `workers` (or analysis suggestions) | `supervisor`, `worker_<i>_<name>`, `synthesize` |
| `pipeline` | Fixed, ordered stages | `stages` (or analysis suggestions) | `stage_<i>_<name>` |
| `planner` | Decomposable work with dependencies | `limits.max_plan_steps`, `limits.max_workers` | `plan`, `step`, `join`, `synthesize` |
| `reflection` | Drafts that benefit from review | `max_reflections` (1-5) | `draft`, `review`, `revise` |
| `swarm` | Peers handing work to each other | `workers` | `peer_<i>_<name>`, `finish` |
| `blackboard` | Independent contributions to a shared record | `workers` | `contributor_<i>_<name>`, `synthesize` |
| `hybrid` | Two to four topologies in sequence | `composition` | `child_<i>_<topology>` |

- Supervisor and Swarm stop after `limits.max_handoffs` delegations.
- Planner runs independent steps in parallel, at most `limits.max_workers` at
  a time, and gives the model one correction round when a plan is invalid.
  Blackboard runs contributors in groups of `limits.max_workers`. A
  `max_concurrency` in the run configuration only limits how many inputs
  `batch` runs at once; it does not apply inside a run.
- Reflection stops early when the reviewer approves.
- With application `middleware`, final answers of Supervisor, Planner,
  Blackboard, and Swarm are written by a middleware-wrapped agent turn, so
  output middleware such as PII redaction covers them.

## Recursion

`limits=ExecutionLimits(max_depth=1)` enables recursion: each Planner step runs
as a child execution that analyzes its own task and selects from
`limits.allowed_nested_topologies`. The budget covers the whole invocation,
including every loop iteration:

- `max_depth`: how deep children may nest (0 disables recursion).
- `max_children`: child executions per invocation.
- `max_execution_time`: seconds of recursive work, from the first child.

With a checkpointer, each execution records its plan (routed capabilities and
topology), so a resumed run continues the same plan at every depth instead of
analyzing the task again.

Exceeding any limit raises `RecursionLimitError`. Breakpoints never apply to
children.

## Choosing a configuration

- Prefer automatic selection for open-ended assistants. It costs one analysis
  call per iteration.
- Pin `pattern` when the workflow is known. It is cheaper and predictable.
- Use an allow-list to keep automatic selection inside approved structures.
- Supervisor, Swarm, Planner, and Reflection make structured routing
  decisions, and automatic selection makes a structured analysis call, so they
  need a model that supports tool calling or structured output.
- To pause before a tool runs and let a person approve, edit, or reject it,
  use LangChain's `HumanInTheLoopMiddleware` or Policy Manager approvals.
  Static breakpoints are debugging stops.
