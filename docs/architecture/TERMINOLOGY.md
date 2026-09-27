# Terminology

| Term | Meaning |
|---|---|
| DIRECT | A lightweight execution mode, not a topology. |
| Topology | One of exactly eight composable graph execution structures. |
| Pattern constraint | Developer allow-list of topology candidates. |
| Analysis | Capability-aware assessment used for automatic or constrained selection. |
| Execution recommendations | Bounded recommendations for instructions, workers, stages, and Hybrid composition; they cannot grant capabilities. |
| Strategy | Policy that maps analysis to one permissible execution mode. |
| Capability | Optional, topology-independent service or tool provider. |
| Recursion | Bounded child-task resolution, orthogonal to topology. |
| Runtime / harness | Isolated lifecycle and execution environment owned by one agent. |
| Hybrid | An actual topology composing two or more allowed child topologies. |

Agloom provides **eight execution topologies plus a direct execution mode**.
There are never nine topologies. A list such as `["planner", "supervisor"]`
means two allowed candidates, not a composition; use an explicit Hybrid
specification to compose topologies.
