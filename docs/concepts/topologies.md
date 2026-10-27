# Topologies

Agloom provides **DIRECT execution plus eight topologies**. DIRECT is an
execution mode, not a topology. Every topology is a compiled LangGraph graph,
and every worker, stage, step, and peer inside it is a tool-capable LangChain
`create_agent` agent that uses the routed tools and your
[middleware stack](middleware.md).

| Mode | Graph nodes (for `interrupt_before` and `interrupt_after`) | Configuration | Limits |
|---|---|---|---|
| [DIRECT](#direct) | none | `system_prompt` | none |
| [ReAct](#react) | `model`, `tools` | `system_prompt` | loop tool-call budget |
| [Supervisor](#supervisor) | `supervisor`, `worker_<i>_<name>`, `synthesize` | `workers` | `max_handoffs` |
| [Pipeline](#pipeline) | `stage_<i>_<name>` | `stages` | none |
| [Planner](#planner) | `plan`, `step`, `join`, `synthesize` | none | `max_plan_steps`, `max_workers`, recursion |
| [Reflection](#reflection) | `draft`, `review`, `revise` | `max_reflections` | 1 to 5 rounds |
| [Swarm](#swarm) | `peer_<i>_<name>`, `finish` | `workers` | `max_handoffs` |
| [Blackboard](#blackboard) | `contributor_<i>_<name>`, `synthesize` | `workers` | `max_workers` |
| [Hybrid](#hybrid) | `child_<i>_<topology>` | `composition` | per child |

Node names are derived from your configuration: `<i>` is the zero-based
position and `<name>` is the worker or stage name lowercased, with every run of
characters other than letters, digits, and underscores replaced by `_` and
leading or trailing underscores removed. For example, worker `"Risk Analyst"`
at position 1 becomes `worker_1_risk_analyst`.

Supervisor, Swarm, Planner, and Reflection make structured routing decisions,
so they need a model with structured output.

## DIRECT

One model call with your `system_prompt` and the conversation. DIRECT offers
the model no tools. When you configure middleware, the call is a tool-less
LangChain agent turn through your full middleware stack, with the policy still
in place, so guardrails, redaction, and summarization apply to direct answers
too. Automatic selection chooses DIRECT when the analysis finds no requirement
for tools or orchestration.

??? example "examples/01_direct.py"

    ```python
    --8<-- "examples/01_direct.py"
    ```

## ReAct

![ReAct topology](../assets/diagrams/topology-react-light.png#only-light)
![ReAct topology](../assets/diagrams/topology-react-dark.png#only-dark)

Reasoning interleaved with tool calls: LangChain's `create_agent` loop of
`model` and `tools`, with the routed tools and your middleware. Use it for
requests that need tools but no further structure.

```python
agent = create_agent(model=model, tools=[order_status], pattern="react")
```

??? example "examples/04_react_tools.py"

    ```python
    --8<-- "examples/04_react_tools.py"
    ```

## Supervisor

![Supervisor topology](../assets/diagrams/topology-supervisor-light.png#only-light)
![Supervisor topology](../assets/diagrams/topology-supervisor-dark.png#only-dark)

A coordinator delegates to specialist workers in a bounded loop. On each turn
the supervisor makes a structured decision (`SupervisorRoute`): the next worker
and a precise assignment, or `FINISH` with the complete answer. Workers see the
assignment and the completed work. The loop ends at `FINISH` or after
`ExecutionLimits.max_handoffs` delegations; `synthesize` then writes the final
answer from the workers' results when the supervisor did not.

```python
from agloom import WorkerSpec, create_agent

agent = create_agent(
    model=model,
    tools=[search_logs, query_metrics],
    pattern="supervisor",
    workers=[
        WorkerSpec(name="logs", description="Investigates application logs."),
        WorkerSpec(name="metrics", description="Analyzes service metrics."),
    ],
)
```

Use it when distinct specialists should be coordinated, and the order of work
depends on intermediate results. `FINISH` is reserved and cannot name a worker.

??? example "examples/05_supervisor.py"

    ```python
    --8<-- "examples/05_supervisor.py"
    ```

## Pipeline

![Pipeline topology](../assets/diagrams/topology-pipeline-light.png#only-light)
![Pipeline topology](../assets/diagrams/topology-pipeline-dark.png#only-dark)

Ordered, tool-capable stages; each stage receives the previous stage's output.
A stage may apply a `transform` function to its text output before the next
stage runs. Without configured or proposed stages, the stages are analyze,
draft, and finalize.

```python
from agloom import PipelineStage, create_agent

agent = create_agent(
    model=model,
    pattern="pipeline",
    stages=[
        PipelineStage(name="extract", instruction="Extract every requirement."),
        PipelineStage(name="draft", instruction="Draft a response meeting them."),
        PipelineStage(name="polish", instruction="Tighten the wording.", transform=str.strip),
    ],
)
```

Use it when work must always pass through the same ordered stages.

??? example "examples/06_pipeline.py"

    ```python
    --8<-- "examples/06_pipeline.py"
    ```

## Planner

![Planner topology](../assets/diagrams/topology-planner-light.png#only-light)
![Planner topology](../assets/diagrams/topology-planner-dark.png#only-dark)

The `plan` node produces a structured `Plan`: self-contained steps with ids and
dependencies. Agloom validates it (at most `max_plan_steps` steps, unique ids,
known dependencies, no cycles); an invalid plan gets one correction round in
which the planner sees the validation error, and the run fails with
[`ExecutionError`][agloom.ExecutionError] only if the corrected plan is still
invalid. Agloom then dispatches every step whose dependencies
are complete as one **wave**. Steps of a wave run in parallel, bounded by
`ExecutionLimits.max_workers`; `join` collects the results and dispatches the
next wave; `synthesize` combines all step results into the answer.

```python
from agloom import ExecutionLimits, create_agent

agent = create_agent(
    model=model,
    tools=tools,
    pattern="planner",
    limits=ExecutionLimits(max_plan_steps=6, max_workers=3),
)
```

With recursion enabled (`ExecutionLimits(max_depth=...)`), every step runs as
a recursive child execution that can itself be analyzed and orchestrated. See
[Execution limits and recursion](limits.md).

??? example "examples/03_planner.py"

    ```python
    --8<-- "examples/03_planner.py"
    ```

## Reflection

![Reflection topology](../assets/diagrams/topology-reflection-light.png#only-light)
![Reflection topology](../assets/diagrams/topology-reflection-dark.png#only-dark)

`draft` writes a complete answer; `review` returns a structured verdict
(`approved` plus concrete `issues`); `revise` resolves the issues. The run ends
as soon as a review approves the answer, or after `max_reflections`
review-and-revise rounds (1 to 5).

```python
agent = create_agent(model=model, pattern="reflection", max_reflections=2)
```

Reflection improves one answer within one execution. To judge whether a whole
execution met its objective, and to retry or change approach, use
[Loop Engineering](loop-engineering.md).

??? example "examples/07_reflection.py"

    ```python
    --8<-- "examples/07_reflection.py"
    ```

## Swarm

![Swarm topology](../assets/diagrams/topology-swarm-light.png#only-light)
![Swarm topology](../assets/diagrams/topology-swarm-dark.png#only-dark)

Peers work in turn and hand control to each other. The first worker starts;
each turn returns a structured `SwarmTurn` with its contribution and the next
peer, or `STOP` with the complete answer. Turns are bounded by
`ExecutionLimits.max_handoffs`; `finish` writes the final answer when no peer
did. `STOP` is reserved and cannot name a worker.

```python
agent = create_agent(
    model=model,
    pattern="swarm",
    workers=[
        WorkerSpec(name="triage", description="Classifies the request."),
        WorkerSpec(name="billing", description="Resolves billing questions."),
        WorkerSpec(name="technical", description="Resolves technical issues."),
    ],
)
```

Use it when the right specialist emerges during the conversation rather than
from a central plan.

??? example "examples/08_swarm.py"

    ```python
    --8<-- "examples/08_swarm.py"
    ```

## Blackboard

![Blackboard topology](../assets/diagrams/topology-blackboard-light.png#only-light)
![Blackboard topology](../assets/diagrams/topology-blackboard-dark.png#only-dark)

Every worker contributes to a shared board in parallel (bounded by
`ExecutionLimits.max_workers`); `synthesize` combines the contributions and
resolves conflicts explicitly. Use it for independent perspectives on one
question, such as security, cost, and reliability reviews of a design.

```python
agent = create_agent(
    model=model,
    pattern="blackboard",
    workers=[
        WorkerSpec(name="security", description="Reviews security risks."),
        WorkerSpec(name="cost", description="Estimates cost impact."),
        WorkerSpec(name="reliability", description="Assesses reliability risks."),
    ],
)
```

??? example "examples/09_blackboard.py"

    ```python
    --8<-- "examples/09_blackboard.py"
    ```

## Hybrid

![Hybrid topology](../assets/diagrams/topology-hybrid-light.png#only-light)
![Hybrid topology](../assets/diagrams/topology-hybrid-dark.png#only-dark)

Runs two to four distinct child topologies in sequence; each child receives the
previous child's result as context. Hybrid cannot contain Hybrid. Children
share the agent's workers, stages, and system prompt.

```python
agent = create_agent(
    model=model,
    tools=tools,
    pattern="hybrid",
    composition=["react", "reflection"],
)
```

`composition=` is valid only when Hybrid can be selected. Breakpoints apply to
the `child_<i>_<topology>` nodes, not to nodes inside the children.

??? example "examples/10_hybrid.py"

    ```python
    --8<-- "examples/10_hybrid.py"
    ```

## Choosing a topology

| If the work... | Prefer |
|---|---|
| needs no tools or structure | DIRECT |
| needs tools, step by step | ReAct |
| needs specialists, coordinated centrally | Supervisor |
| always follows the same ordered stages | Pipeline |
| decomposes into dependent steps, some parallel | Planner |
| must be reviewed before it is returned | Reflection |
| moves between specialists as it unfolds | Swarm |
| benefits from independent parallel perspectives | Blackboard |
| needs two structures in sequence | Hybrid |

Let [automatic selection](execution-selection.md) make this choice per request,
or constrain it with an allow-list.
