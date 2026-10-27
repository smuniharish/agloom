# Observability

Every decision an agent makes is an **event**: which capabilities were routed,
which execution mode was chosen and why, which tool calls were denied or
approved, how each loop iteration was judged, and how the run ended. Observers
turn those events into logs, metrics, and traces.

![Observability flow](../assets/diagrams/observability-flow-light.png#only-light)
![Observability flow](../assets/diagrams/observability-flow-dark.png#only-dark)

## Observers

Pass observers with `observers=`. An observer is any callable that accepts an
[`AgentEvent`][agloom.AgentEvent], or an [`Observer`][agloom.Observer]
subclass, which can also contribute LangChain callback handlers.

```python
import os

from agloom import LangfuseObserver, LoggingObserver, PrometheusObserver, create_agent

metrics = PrometheusObserver()
metrics.start_http_server(port=9464)

agent = create_agent(
    model=model,
    tools=tools,
    observers=[
        LoggingObserver(),
        metrics,
        LangfuseObserver(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_BASE_URL"],
        ),
        lambda event: print(event.name, dict(event.payload)),
    ],
)
try:
    agent.invoke("Where is order A-1001?")
finally:
    agent.close()  # flush telemetry and stop observer resources
```

| Observer | Output |
|---|---|
| [`LoggingObserver`][agloom.LoggingObserver] | One structured log record per event with structlog; failures and denials at warning level |
| [`PrometheusObserver`][agloom.PrometheusObserver] | Counters, a gauge, and a histogram for executions, policy actions, and loop decisions; optional built-in HTTP endpoint |
| [`LangfuseObserver`][agloom.LangfuseObserver] | Events grouped into one Langfuse trace per run, plus LangChain callback tracing of every model, tool, and graph call |
| Any callable | Whatever you need: forward to your telemetry, assert in tests |

Observers are isolated: an observer that raises is logged on the
`agloom.observability` logger and skipped, and never changes the outcome of a
run. Call `agent.flush()` to flush telemetry and `agent.close()` on shutdown.

??? example "examples/23_observability.py"

    ```python
    --8<-- "examples/23_observability.py"
    ```

## Event catalog

Each [`AgentEvent`][agloom.AgentEvent] carries `name` (an
[`EventName`][agloom.EventName]), `agent_id`, `run_id`, a UTC `timestamp`,
`topology` when known, and a read-only, JSON-compatible `payload`. Payloads
never contain prompts, tool arguments, or credentials.

| Event | Emitted when | Payload |
|---|---|---|
| `AgentCreated` | `create_agent` returned the agent | none |
| `ExecutionStarted` | An invocation, stream, or resume started | `resume` |
| `CapabilitiesRouted` | Capabilities were routed for an iteration | `selected`, `available` |
| `TopologySelectionStarted` | Task analysis started | `depth` |
| `TopologySelected` | DIRECT or a topology was chosen, by analysis or by an explicit `pattern`; the event's `topology` is set for topologies | `mode`, `reason`, `missing_capabilities` (count) |
| `DirectExecutionStarted` | DIRECT execution started | none |
| `ExecutionCompleted` | The invocation finished | `iterations` (not reported for streamed runs) |
| `ExecutionPaused` | The invocation paused at an interrupt, also inside an enclosing graph, or LangGraph drained it at a step boundary | `interrupts` (`0` for a drain) |
| `ExecutionFailed` | The invocation failed | `error` (exception type) |
| `ExecutionCancelled` | The invocation was cancelled with `agent.stop()`, by a stream consumer that stopped early, or by `asyncio` cancellation such as a timeout | none |
| `ToolCallDenied` | A tool call was blocked before it ran: by the policy, because its arguments did not match the tool's schema (`reason` is `InvalidArguments`), or because the tool was not offered for the request (`reason` is `ToolNotOffered`) | `tool`, `reason` |
| `ApprovalRequired` | A consequential tool call is waiting for a reviewer (emitted once per pause) | `tool`, `action_id` |
| `ActionExecuted` | A consequential tool call ran | `tool` |
| `ActionReplayed` | A consequential tool call returned its recorded result | `tool` |
| `ActionFailed` | A consequential tool call raised, so its outcome is unknown until reconciled; calls refused while it awaits reconciliation are reported too when they cannot pause (without a checkpointer, or made by your code) | `tool`, `error`, and `action_id` when known |
| `LoopStarted` | The outer loop started | `loop_id`, `max_iterations` |
| `LoopIterationStarted` | An iteration started | `iteration`, `action` |
| `LoopIterationCompleted` | An iteration was assessed and decided | `loop_id`, `iteration`, `action`, `reason`, `progress_status`, `objective_status` |
| `LoopRefreshed` | Sources were refreshed before the next iteration | `status`, `refreshed` |
| `LoopPaused` | The loop paused for a human | `loop_id`, `iteration`, `reason` |
| `LoopCompleted` | The loop completed its objective | `loop_id`, `iterations`, `termination_reason` |
| `LoopTerminated` | The loop stopped for any other reason | `loop_id`, `iterations`, `termination_reason` |

A typical automatic request that runs DIRECT emits `ExecutionStarted`,
`LoopStarted`, `LoopIterationStarted`, `CapabilitiesRouted`,
`TopologySelectionStarted`, `TopologySelected`, `DirectExecutionStarted`,
`LoopIterationCompleted`, `LoopCompleted`, and `ExecutionCompleted`, in that
order. With an explicit `pattern` there is no analysis, so
`TopologySelectionStarted` is skipped and `TopologySelected` (mode `topology`)
follows `CapabilitiesRouted`; metrics are labelled with the topology either
way. The same events are available as a live stream; see
[Streaming](../guides/streaming.md#agloom-events).

## Prometheus metrics

| Metric | Labels |
|---|---|
| `events_total` | `event`, `topology` |
| `executions_total` | `status`, `topology` |
| `execution_duration_seconds` (histogram) | `status`, `topology` |
| `active_executions` (gauge) | none |
| `policy_actions_total` | `tool`, `outcome` |
| `loop_iterations_total` | `progress`, `topology` |
| `loop_decisions_total` | `action` |
| `loop_terminations_total` | `reason` |

Metric names are prefixed with the observer's `namespace` (`agloom` by
default). `status` is `completed`, `failed`, `cancelled`, or `paused`;
`topology` is the selected topology, `direct`, or `pending`. Pass
`registry=` to use your own `CollectorRegistry`, and expose metrics either
through your application's metrics endpoint or with
`start_http_server(port, address="127.0.0.1")`.

## Langfuse

`LangfuseObserver()` creates a Langfuse client from the standard environment
variables `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`,
or from `public_key`, `secret_key`, and `base_url`, or
uses a `client=` you pass. Events of one invocation share a deterministic trace
derived from the run id; failures are recorded at error level and denials and
cancellations at warning level. Add tenant or application metadata to every
event with `event_metadata=`.

## Explainability

For decision provenance and explanations, pass a
[langgraph-xai](https://pypi.org/project/langgraph-xai/) runtime with
`explainability=XAIRuntime(application_id="support")`. Agloom instruments task
analysis, DIRECT execution, every compiled topology, and the loop evaluator
with it, and exposes the runtime as `agent.xai`. Explainability is opt-in:
without `explainability=`, nothing is captured.

??? example "examples/18_explainability.py"

    ```python
    --8<-- "examples/18_explainability.py"
    ```

## Production stack

The repository's `examples/observability/` directory contains a Docker
Compose stack used by the reference applications:

- **Prometheus** with alert rules for unavailable metrics targets, a high
  execution failure ratio, failed consequential actions (their outcome is
  unknown and must be reconciled), repeated policy denials, and loops that
  stop before reaching their objective;
- **Grafana** with a provisioned *Agloom Runtime* dashboard (runtime, Policy
  Manager, and Loop Engineering panels) that you can filter by Prometheus
  `job`;
- **Langfuse** for traces.

See [Production deployment](../guides/production.md#observability).
