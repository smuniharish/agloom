# Production deployment

Agloom runs inside your application. This guide covers the production setup:
durable state in PostgreSQL, thread authorization, exactly-once side effects,
background execution with a job queue, MCP lifecycle, observability, and a
security checklist.

## Deployment shape

A typical deployment has:

- **API processes** that accept requests, authenticate users, and enqueue runs;
- **worker processes** that build agents once and execute runs from a durable
  job queue;
- **PostgreSQL** for LangGraph checkpoints, the agentbarrier policy store, and
  the job queue;
- **an observability stack**: Prometheus and Grafana for metrics, Langfuse for
  traces, and structured logs.

Install the production extras (use `agloom[postgres-binary]` on hosts without
the PostgreSQL client library):

```bash
pip install "agloom[postgres]"
```

## Agents in a process

- **Create agents once** per process and share them across requests: they are
  thread-safe, and compiled topology graphs are cached per configuration.
- Create **separate agents** when tools, policies, or system prompts differ,
  for example per tenant tier. Never share a tenant-specific registry,
  authorizer, or policy identity across tenants.
- Use the **async API** in async servers. Agents with `mcp=`, `refresh=`, or
  `feedback=` require it.
- On shutdown, call `agent.close()` to flush telemetry, and close the
  components you own (MCP runtime, stores, connection pools).

## Durable state in PostgreSQL

Create or upgrade the schemas once per deployment, before workers start:

```python
import os

from agentbarrier.runtime import PostgresRuntimeStore
from langgraph.checkpoint.postgres import PostgresSaver

database_url = os.environ["DATABASE_URL"]
with PostgresSaver.from_conn_string(database_url) as checkpointer:
    checkpointer.setup()
PostgresRuntimeStore(
    database_url, schema="agentbarrier", create_schema=True, migrate=True
).close()
```

At run time, give agents an async checkpointer, an authorizer, and a policy
backed by the same database:

```python
import os
from contextvars import ContextVar

from agentbarrier.runtime import PostgresRuntimeStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agloom import PolicyManager, create_agent

current_tenant: ContextVar[str | None] = ContextVar("current_tenant", default=None)


def authorize(thread_id: str) -> bool:
    """Admit only threads that belong to the tenant of the current request."""
    tenant = current_tenant.get()
    return tenant is not None and thread_id.startswith(f"{tenant}:")


async def serve() -> None:
    database_url = os.environ["DATABASE_URL"]
    store = PostgresRuntimeStore(database_url, schema="agentbarrier")
    async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
        agent = create_agent(
            model=model,
            tools=[lookup_order, refund_order],
            policy=PolicyManager(
                policy="policies/support.json",
                store=store,
                consequential={"refund_order": "request_id"},
                namespace="support-agent",
                organization_id="acme",
            ),
            checkpointer=checkpointer,
            checkpoint_authorizer=authorize,
        )
        current_tenant.set("acme")
        answer = await agent.ainvoke(
            "Where is order A-1001?", {"configurable": {"thread_id": "acme:run-42"}}
        )
        print(answer.content)
    store.close()
```

- **Thread ids** should encode ownership (for example
  `"<tenant>:<run id>"`), and the authorizer must check it against the
  authenticated principal of the current request. Every invocation, stream,
  resume, and state read is authorized.
- **Strict serialization.** Agloom checkpoints contain only JSON-compatible
  values and LangChain messages, so agents run with LangGraph's strict msgpack
  mode. Set `LANGGRAPH_STRICT_MSGPACK=true` in production and keep your own
  checkpointed state JSON-friendly too.
- **Windows.** psycopg's async connections need a selector event loop:
  `asyncio.run(serve(), loop_factory=asyncio.SelectorEventLoop)`.

??? example "examples/47_postgres_checkpointer.py"

    ```python
    --8<-- "examples/47_postgres_checkpointer.py"
    ```

## Exactly-once side effects

With `PostgresRuntimeStore`, any number of processes can share the policy
store safely: action binding, execution claims, limits, and receipt chains are
serialized per schema. Declare every tool with side effects in
`consequential=` with a business idempotency key.

Side effects fail closed. If such a tool raises, the action's outcome is
recorded as unknown and Agloom emits `ActionFailed` with the `action_id`; if a
worker dies while the tool runs, the outcome becomes unknown when the
execution lease expires. The tool is never run again for that key until an
operator checks the external system and reconciles the action. Checkpointed
runs pause with a `RECONCILIATION_INTERRUPT` meanwhile and continue with
`agent.resume(config)` after reconciliation; runs without a checkpointer, and
direct calls, raise `ActionOutcomeUnknown`.

```bash
agentbarrier approvals list --postgres-dsn-env DATABASE_URL --status unknown
agentbarrier approvals reconcile <action-id> --postgres-dsn-env DATABASE_URL \
  --outcome not_committed --resolved-by alice --reason "No refund in the payments console"
```

`not_committed` lets the next call execute the action exactly once. Use
`--outcome committed` with `--result-json` when the external effect is proven
to have happened; the next call replays that result to the model (its
`content` field when it has one, a string as is, and any other value as JSON
text).
Operators can also activate emergency pauses and atomic action limits with
`agentbarrier controls`, and verify audit receipts with `agentbarrier audit`.

??? example "examples/44_policy_postgres.py"

    ```python
    --8<-- "examples/44_policy_postgres.py"
    ```

## Background execution

Long-running and approval-gated runs belong in a durable job queue. The
reference applications use [procrastinate](https://procrastinate.readthedocs.io/)
on PostgreSQL:

- **One job per run.** The job's lock is the run id, so a run never executes
  twice at the same time. The thread id is derived from the run id.
- **Execute or continue.** If the run's thread has a checkpoint with pending
  work, the job calls `await agent.aresume(config)`; otherwise it starts the
  run with `await agent.ainvoke(task, config)`. Completed steps and executed
  consequential tools are never repeated.
- **Retries and stalled workers.** Use a `RetryStrategy` for transient errors
  and worker shutdowns, and periodically requeue jobs whose worker stopped
  sending heartbeats. The requeued job continues from the checkpoint.
- **Cancellation.** Abort the job (`cancel_job_by_id_async(job_id, abort=True)`)
  and, in the worker, call `agent.stop()`: the run stops cooperatively at the
  next model call, tool call, or graph step, never in the middle of a tool.
- **Approvals and reconciliations.** When a run raises `ExecutionPaused`,
  branch on the interrupt `type`: record an `APPROVAL_INTERRUPT` as awaiting
  approval and a `RECONCILIATION_INTERRUPT` as awaiting reconciliation (its
  outcome is unknown), and end the job. After the reviewer decides, or the
  operator reconciles the action, in agentbarrier, enqueue the run again; the
  job resumes it, and the approved or released call executes exactly once.

```python
from agloom import APPROVAL_INTERRUPT, RECONCILIATION_INTERRUPT, ExecutionPaused

PAUSED_STATUS = {
    APPROVAL_INTERRUPT: "awaiting_approval",
    RECONCILIATION_INTERRUPT: "awaiting_reconciliation",
}


async def execute_run(agent, run_id: str, task: str) -> str:
    config = {"configurable": {"thread_id": run_id}}
    snapshot = await agent.aget_state(config)
    try:
        if snapshot.next or snapshot.interrupts:
            answer = await agent.aresume(config)
        else:
            answer = await agent.ainvoke(task, config)
    except ExecutionPaused as paused:
        status = PAUSED_STATUS.get(paused.interrupts[0].value.get("type"), "paused")
        await record_pause(run_id, status, paused.interrupts)
        return status
    await record_answer(run_id, answer.content)
    return "completed"
```

The [Autonomous Product Studio](https://github.com/smuniharish/agloom/tree/master/examples/autonomous_product_studio)
reference application implements this pattern end to end with procrastinate,
including aborts, stalled-job recovery, and approval-driven resumes.

## MCP in production

The application owns MCP connections. Open the runtime once in your server's
lifespan, register servers, pass it to the agents, and close it on shutdown:

```python
from contextlib import asynccontextmanager

from mcp_capability_router import MCPRuntime, RefreshPolicy


@asynccontextmanager
async def lifespan(app):
    async with MCPRuntime() as runtime:
        await runtime.register_mcp(
            "inventory",
            "https://mcp.example.com/inventory/mcp",
            refresh=RefreshPolicy(on_register=True, interval=300),
        )
        app.state.agent = create_agent(model=model, mcp=runtime, policy=policy)
        yield
```

`on_register=True` discovers the server's tools, resources, and prompts before
the agent serves requests (without a refresh policy, or an explicit
`await runtime.refresh()`, the agent sees no MCP capabilities), and
`interval=300` keeps them current. Configure stdio servers with
`"keep_alive": False` so that shutdown stops their processes. Write policy
rules against the `<server_id>__<tool>` names, and treat MCP servers and their
outputs as untrusted.

## Observability

- **Metrics.** Register a [`PrometheusObserver`][agloom.PrometheusObserver] on
  your application's registry and serve it from your metrics endpoint, or call
  `start_http_server(port)` in workers. Alert on the execution failure ratio,
  on failed consequential actions (`policy_actions_total{outcome="failed"}`:
  their outcome is unknown until reconciled), on repeated denials
  (`outcome="denied"`), on pending approvals, and on loops that end on
  budgets, stagnation, or oscillation (`loop_terminations_total` by `reason`).
- **Traces.** A [`LangfuseObserver`][agloom.LangfuseObserver] groups each run's
  events into one trace and traces every model and tool call. Traces contain
  prompts and tool arguments; apply your data-retention rules.
- **Logs.** [`LoggingObserver`][agloom.LoggingObserver] writes one structured
  record per event; event payloads never contain prompts, tool arguments, or
  credentials.
- **A ready-made stack.** The repository's `examples/observability/` directory
  provides Docker Compose files for Prometheus (with alert rules for execution
  failures, failed consequential actions, repeated denials, loops that stop
  before their objective, and unavailable metrics targets), Grafana (with a
  provisioned dashboard filterable by `job`), and Langfuse. Point Langfuse
  clients at it with `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
  `LANGFUSE_BASE_URL`.

## Security checklist

**Access and state**

- [x] A `checkpoint_authorizer` that checks thread ownership against the
  authenticated principal; thread ids that encode the tenant.
- [x] Separate `namespace` and `organization_id` per agent and tenant in every
  `PolicyManager`.
- [x] `LANGGRAPH_STRICT_MSGPACK=true`, and JSON-friendly application state.

**Tools and side effects**

- [x] A policy for every agent with tools: specific denials first, approvals
  for high-impact actions, explicit allows last; anything else is denied.
- [x] Every side-effecting tool declared `consequential` with a business
  idempotency key; approval TTLs on approval rules.
- [x] Capability filters per tenant to narrow what the model is offered; the
  policy, not the filter, decides what may run.
- [x] Code execution and file access isolated in a sandbox at the tool level;
  policies decide whether a call may run, not what the code inside it can do.

**Prompt-injection hygiene**

- [x] Trusted instructions only in `system_prompt` and topology configuration.
  Agloom rejects system and tool messages in input, so callers cannot inject
  instructions or forge tool results.
- [x] Treat tool results, retrieved documents, and MCP outputs as untrusted.
  Agloom passes task-analysis notes, loop feedback, previous results, and human
  feedback to the model as labelled data in the user turn, never as system
  instructions; do the same in your own prompts and tools.
- [x] Keep consequential actions behind policy: injected text can ask for a
  refund, but it cannot approve one.

**Secrets and operations**

- [x] Credentials only in environment variables or a secret manager; never in
  prompts, tool arguments, code, or logs. The agentbarrier CLI reads database
  connection strings from a named environment variable.
- [x] Budgets on every agent: `ExecutionLimits` and `LoopPolicy` limits sized
  for the workload.
- [x] Alerts on denials, failed actions, unknown outcomes, and pending
  approvals; an on-call procedure for reconciling actions.
