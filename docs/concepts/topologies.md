# Execution topologies

Agloom has **eight execution topologies plus a direct execution mode**.
DIRECT is not a topology. The topologies share LangGraph, execution state,
runtime policies, and lifecycle handling.

| Topology | Use |
|---|---|
| ReAct | Tool-aware reasoning and action loop. |
| Supervisor | Select from a finite configured worker set. |
| Pipeline | Run ordered stages and propagate results. |
| Planner | Decompose work into bounded subtasks and aggregate results. |
| Reflection | Draft, evaluate, and revise a response. |
| Swarm | Hand off between a bounded set of peer workers. |
| Blackboard | Let workers publish conflict-checked shared findings. |
| Hybrid | Compose two to four non-Hybrid topologies. |

Use `pattern="planner"` to request a topology directly. A list such as
`["planner", "supervisor"]` is an allow-list, not a composition. See the
[topology specification](../architecture/TOPOLOGY_SPECIFICATION.md).
