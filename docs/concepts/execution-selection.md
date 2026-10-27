# Execution selection

For every request, Agloom decides **how** to execute it: DIRECT (one model
call) or one of eight [topologies](topologies.md). You control how much of
that decision is automatic with `pattern=`.

![Execution selection](../assets/diagrams/execution-selection-light.png#only-light)
![Execution selection](../assets/diagrams/execution-selection-dark.png#only-dark)

## Three modes

| `pattern=` | Behavior | Analysis call |
|---|---|---|
| `None` (default) | **Automatic.** DIRECT or any of the eight topologies | One per request |
| `["planner", "supervisor"]` | **Allow-list.** Exactly one of the listed topologies; never DIRECT | One per request |
| `"planner"` | **Explicit.** Always this topology | None |

```python
from agloom import create_agent

automatic = create_agent(model=model, tools=tools)
constrained = create_agent(model=model, tools=tools, pattern=["planner", "supervisor"])
pinned = create_agent(model=model, tools=tools, pattern="planner")
```

An allow-list chooses one topology; it never runs several. To run topologies
in sequence, use [Hybrid](topologies.md#hybrid) with an explicit
`composition=`.

## Task analysis

In automatic and allow-list modes, the model produces a structured
[`TaskAnalysis`][agloom.TaskAnalysis] using provider-native structured output.
The analysis describes the task; it does not answer it and does not choose a
topology:

| Field | Meaning |
|---|---|
| `complexity`, `estimated_steps` | Difficulty (0 to 10) and the estimated number of steps |
| `requires_tools` | An available tool or capability is needed |
| `requires_decomposition` | The task must be split into dependent subtasks |
| `requires_parallelism` | Independent subtasks can run at the same time |
| `requires_delegation` | Specialists should be coordinated by a supervisor |
| `requires_handoffs` | Peers should hand control to each other |
| `requires_shared_workspace` | Several workers should contribute to one shared board |
| `requires_ordering` | Work must pass through fixed, ordered stages |
| `requires_validation` | The answer should be reviewed and revised |
| `requires_composition` | Two or more topologies must run in sequence |
| `missing_capabilities` | Needed external actions that no routed tool can perform |
| `blueprint` | Suggested workers, stages, composition, and task-specific notes |

The analysis sees the **routed capability inventory**: names, descriptions,
and kinds of the tools and services selected for the request, plus your
configured workers and stages. The task and the inventory are passed as
untrusted data.

!!! tip "Missing capabilities are never improvised"

    When the analysis lists `missing_capabilities`, the selected execution
    receives them as data, with the instruction not to claim those actions and
    to say what cannot be done. The answer explains the gap instead of
    pretending, and the `TopologySelected` event reports how many requested
    actions were missing.

## The strategy

The [`StrategyEngine`][agloom.StrategyEngine] maps the analysis to a decision
deterministically. The preferred topology follows the dominant requirement, in
this priority order:

| Priority | Requirement | Topology |
|---|---|---|
| 1 | `requires_composition` | Hybrid |
| 2 | `requires_shared_workspace` | Blackboard |
| 3 | `requires_handoffs` | Swarm |
| 4 | `requires_parallelism` | Planner |
| 5 | `requires_delegation` | Supervisor |
| 6 | `requires_decomposition` | Planner |
| 7 | `requires_validation` and `complexity >= 4` | Reflection |
| 8 | `requires_ordering` and `estimated_steps >= 3` | Pipeline |
| 9 | `requires_tools` | ReAct |
| 10 | none of the above | DIRECT |

With an allow-list that does not contain the preferred topology, the candidate
that satisfies the most requirements wins, with ties going to the earlier entry
in your list. The decision is reported in the `TopologySelected` event with its
`mode` and `reason`.

## Workers, stages, and blueprints

Supervisor, Swarm, and Blackboard need workers; Pipeline uses stages; Hybrid
needs a composition. Configured values always win:

- `workers=[WorkerSpec(...)]` and `stages=[PipelineStage(...)]` are used for
  every request.
- Without configured workers or stages, the analysis may propose them in its
  `blueprint`. Proposals are validated (unique names, bounded lengths) before
  use.
- Pipeline falls back to three stages (analyze, draft, finalize) when nothing
  is configured or proposed.
- Hybrid uses `composition=` when given, otherwise the analysis proposal.

If the selected topology still cannot run (for example, Supervisor without any
workers), Agloom raises [`TopologySelectionError`][agloom.TopologySelectionError].

The blueprint's `instructions` (objective, constraints, success criteria) reach
the selected execution as clearly labelled data in the user turn. They never
grant capabilities and never become system instructions.

## Requirements and costs

- Automatic and allow-list modes need a model with structured output. Without
  it, `create_agent` raises [`ConfigurationError`][agloom.ConfigurationError]
  and suggests pinning a topology.
- The analysis adds one model call per request (one per iteration when Loop
  Engineering iterates, and one per recursive child). Pin a topology for
  latency-critical paths where the right structure is known.
- The one-shot [Loop Engineering](loop-engineering.md) default adds no calls.

## Custom strategies

Implement the [`Strategy`][agloom.Strategy] protocol to change the mapping,
for example to prefer Reflection for every validated task:

```python
from agloom import ExecutionDecision, ExecutionMode, StrategyEngine, TopologyName


class ReviewFirst(StrategyEngine):
    def decide(self, analysis, candidates=None, *, allow_direct=True):
        if analysis.requires_validation and (
            candidates is None or TopologyName.REFLECTION in candidates
        ):
            return ExecutionDecision(
                mode=ExecutionMode.TOPOLOGY,
                topology=TopologyName.REFLECTION,
                reason="validated tasks are always reviewed",
            )
        return super().decide(analysis, candidates, allow_direct=allow_direct)


agent = create_agent(model=model, tools=tools, strategy=ReviewFirst())
```
