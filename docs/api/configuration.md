# Configuration types

These public types make topology and runtime configuration explicit.

## `TopologyName`

String enum with exactly eight values:

`REACT`, `SUPERVISOR`, `PIPELINE`, `PLANNER`, `REFLECTION`, `SWARM`,
`BLACKBOARD`, and `HYBRID`.

DIRECT belongs to `ExecutionMode`; it is not a `TopologyName`.

## `ExecutionMode`

| Value | Meaning |
|---|---|
| `DIRECT` | Execute with the model without constructing a topology. |
| `TOPOLOGY` | Execute through one selected topology. |

## `RecursionPolicy`

| Field | Type | Requirement | Default | Validation |
|---|---|---:|---|---|
| `max_depth` | `int` | Optional | `3` | 1–8 |
| `max_workers` | `int` | Optional | `4` | 1–32 |
| `max_subtasks` | `int` | Optional | `16` | 1–128 |
| `max_execution_time` | `float` | Optional | `120.0` | Greater than 0 and at most 3600 seconds |
| `allowed_nested_topologies` | `tuple[TopologyName, ...]` | Optional | All eight | Topologies allowed for recursive children |

The model is immutable after construction.

## `WorkerSpec`

| Field | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | **Required** | — | Unique, non-empty worker name. |
| `description` | `str` | **Required** | — | Non-empty summary used when selecting a worker. |
| `instructions` | `str` | Optional | `""` | Worker-specific execution instructions. |

```python
from agloom import WorkerSpec

researcher = WorkerSpec(
    name="researcher",
    description="Finds evidence in approved sources.",
    instructions="Return concise findings with source identifiers.",
)
```

## `PipelineStage`

| Field | Type | Requirement | Default | Description |
|---|---|---:|---|---|
| `name` | `str` | **Required** | — | Unique, non-empty stage name. |
| `instruction` | `str` | **Required** | — | Instruction applied at this stage. |
| `transform` | `Callable[[Any], Any] \| None` | Optional | `None` | Optional application transform for the stage value. |

```python
from agloom import PipelineStage

validate = PipelineStage(
    name="validate",
    instruction="Check the draft against the supplied requirements.",
)
```
