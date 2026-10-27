# Policies and approvals

Prompts are not controls. A model asked to "be careful with refunds" can still
issue one. Agloom enforces policy **in code, between the model and every
tool**, with [agentbarrier](https://pypi.org/project/agentbarrier/): a call
runs only if the policy allows it, waits for a reviewer when the policy
requires approval, and, for consequential tools, executes exactly once per
business key with an audit receipt.

![Policy enforcement](../assets/diagrams/policy-enforcement-light.png#only-light)
![Policy enforcement](../assets/diagrams/policy-enforcement-dark.png#only-dark)

## The policy model

A policy is an agentbarrier `RuntimePolicy`: a version and an ordered list of
rules. Each rule has a unique name, an effect (`allow`, `deny`, or
`require_approval`), a tool glob (`refund`, `lookup_*`, `crm__*`), and optional
argument conditions such as `amount > 100`. **The first matching rule wins,
and by default anything no rule matches is denied.**

```python
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langchain_core.tools import tool

from agloom import PolicyManager


@tool
def lookup_order(order_id: str) -> str:
    """Return the status and total of an order."""
    return f"order {order_id}: delivered damaged, total 240 EUR"


@tool
def refund_order(order_id: str, amount: float, request_id: str) -> str:
    """Refund an order. request_id uniquely identifies the refund request."""
    return f"refunded {amount} EUR for order {order_id}"


policy = PolicyManager(
    policy=RuntimePolicy(
        version="support-v1",
        rules=(
            PolicyRule(
                "review large refunds",
                PolicyEffect.REQUIRE_APPROVAL,
                tool="refund_order",
                conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
            ),
            PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="refund_order"),
            PolicyRule("allow lookups", PolicyEffect.ALLOW, tool="lookup_*"),
        ),
    ),
    store=SQLiteRuntimeStore("agentbarrier.db"),
    consequential={"refund_order": "request_id"},
    namespace="support-agent",
    organization_id="acme",
    requested_by="support-agent",
)
```

Policies can also be loaded from a strict JSON file or a mapping; see
[Writing policies](../guides/writing-policies.md).

## Two kinds of tools

**Consequential tools** change the world: refunds, payments, emails,
deployments. Declare each one in `consequential=` with a **business
idempotency key**, the name of a string argument (or a callable deriving the
key from the arguments) that identifies the real-world action, such as a
request or order id. agentbarrier then mediates the call durably:

- it binds the exact request (tool and canonical arguments) to an action;
- it runs the approval workflow when a rule requires approval;
- it claims execution so the action runs **at most once** per key: retries,
  replays, and resumes return the recorded result instead of running again;
- it enforces operator emergency pauses and atomic action limits;
- it writes tamper-evident audit receipts.

When the model requests the same consequential call twice at once, for
example two identical tool calls in one turn, the second waits for the first
and replays its result. A call made by your own code with
`execute_capability` or `PolicyManager.execute` while the same action is
running raises agentbarrier's `ActionInProgress` instead, so your application
can report the request as in progress or retry it later.

A consequential action whose outcome cannot be proven is never repeated
automatically; see [Failures and reconciliation](#failures-and-reconciliation).

**Every other tool** is checked statelessly against the same policy: `allow`
runs it, `deny` blocks it. A rule that requires approval for a tool that is not
declared consequential fails closed with
[`PolicyConfigurationError`][agloom.PolicyConfigurationError], because an
approval needs an idempotency key to bind to.

Tools contributed by middleware (for example `write_todos` from LangChain's
`TodoListMiddleware`, or the filesystem tools of Deep Agents) are governed
exactly like your own tools and can be declared consequential too. Because
anything no rule matches is denied, add allow rules for the middleware tools
the model should use.

## The arguments the policy judges

Every call is validated against the tool's argument schema before the policy
evaluates it, so rules see exactly the values the tool receives. A Pydantic
schema validates with the tool's own argument model, so its conversions,
validators, settings, and defaults apply exactly as when LangChain calls the
tool: a model that sends `"amount": "5000"` meets the rule `amount > 100` as
the number `5000`, and an idempotency key normalized by a validator is the key
agentbarrier records. JSON schemas, such as those of MCP tools, are checked
strictly and their top-level defaults filled in, so a server can never convert
a value the policy did not see.

A call whose arguments do not match the schema does not run and records no
action. Whatever `on_denied` is set to, the model receives an error tool
message naming the problem and can correct the call, and a `ToolCallDenied`
event with the reason `InvalidArguments` is emitted. The same applies to a
consequential call whose idempotency-key argument is missing or blank.
`execute_capability` and `aexecute_capability` raise
[`CapabilityResolutionError`][agloom.CapabilityResolutionError] for invalid
arguments.

`create_agent` checks the tools a policy governs:

- each tool declares its arguments with a Pydantic v2 model or a JSON schema
  that has a JSON form, because the policy evaluates and records JSON
  arguments. For a tool without an `args_schema`, the schema LangChain infers
  from its `_run` method counts only when that signature names every argument:
  no `*args`, `**kwargs`, or `RunnableConfig` parameter, so a plain
  single-input `Tool` needs an `args_schema`;
- the idempotency key of a consequential tool names one of its arguments, and
  that argument can be a string.

Capabilities registered after the agent is created are validated when they
run.

## What the model sees

Before the model is called, Agloom analyzes the policy statically. A tool that
the policy denies for **every** possible call (no rule can allow it or require
approval for it) is removed from the agent and never offered to the model or to
task analysis. Tools that are allowed only for some arguments stay visible, and
each call is checked at run time.

Capability routing narrows the offered tools further for each request, but it
is not authorization: whether a call may run is decided by the policy alone.

## Denials

| `on_denied=` | When a model's tool call is denied before it runs |
|---|---|
| `"raise"` (default) | The run fails closed with [`PolicyViolationError`][agloom.PolicyViolationError] (for a consequential tool, the agentbarrier denial, such as `PolicyDenied` or `ApprovalRejected`); Loop Engineering stops with `policy_violation` |
| `"inform_model"` | The model receives an error tool message explaining that the call was not executed, and can choose another approach |

Every denial is reported as a `ToolCallDenied` event whose `reason` names the
denial, such as `PolicyViolationError`, `PolicyDenied`, or `ApprovalRejected`.

## Approvals

When a rule requires approval for a consequential tool:

1. **With a checkpointer**, the run pauses at a LangGraph interrupt. `invoke`
   raises [`ExecutionPaused`][agloom.ExecutionPaused]; its `interrupts` carry a
   payload with `type` equal to [`APPROVAL_INTERRUPT`][agloom.APPROVAL_INTERRUPT],
   the `action_id`, the `tool`, its `arguments`, the matching `policy_rule`, the
   `namespace` and `organization_id`, and the approval expiry (`expires_at_ns`).
2. A reviewer decides **in agentbarrier**, never in the agent: in code with
   `store.decide(action_id, Decision.APPROVE, decided_by="alice")`, with the
   `agentbarrier approvals` CLI, or through agentbarrier's approval HTTP API,
   dashboard, or Slack integration.
3. `agent.resume(config)` continues the thread. An approved call executes
   exactly once; a rejected or expired one fails closed.

```python
from agentbarrier import Decision
from langgraph.checkpoint.memory import InMemorySaver

from agloom import ExecutionPaused, create_agent

agent = create_agent(
    model=model,
    tools=[lookup_order, refund_order],
    policy=policy,
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=lambda thread_id: thread_id.startswith("ticket-"),
)
config = {"configurable": {"thread_id": "ticket-4711"}}
try:
    agent.invoke("Refund order A-1042 in full. Use refund request id R-88.", config)
except ExecutionPaused as paused:
    action_id = paused.interrupts[0].value["action_id"]
    policy.store.decide(action_id, Decision.APPROVE, decided_by="alice")
    answer = agent.resume(config)
    print(answer.content)
```

```text
Full refund submitted successfully for order **A-1042**.

- **Refund amount:** €240
- **Refund request ID:** R-88
- **Order status:** Delivered damaged
```

**Without a checkpointer** there is nothing to resume, so agentbarrier's
`ApprovalRequired` propagates to your code with the pending action. Configure
a checkpointer for approval workflows.

??? example "examples/41_policy_approval.py"

    ```python
    --8<-- "examples/41_policy_approval.py"
    ```

## Failures and reconciliation

Agloom fails closed when a consequential tool raises. agentbarrier records the
action's outcome as **unknown**, because the external effect may or may not
have happened, and refuses to run the action again until an operator checks
the external system and reconciles it. Agloom emits an `ActionFailed` event
with the tool, the error, and the `action_id`.

**With a checkpointer, the run pauses**, exactly like an approval. `invoke`
raises [`ExecutionPaused`][agloom.ExecutionPaused]; the interrupt payload has
`type` equal to [`RECONCILIATION_INTERRUPT`][agloom.RECONCILIATION_INTERRUPT],
the `action_id`, the `tool`, its `arguments`, the `error` (the type of the
exception the tool raised), the `namespace`, and the `organization_id`. After
the operator reconciles the action, `agent.resume(config)` retries the call:

- **committed**: the operator's recorded result is replayed to the model, and
  the tool does not run;
- **not committed**: the call runs exactly once;
- **not yet reconciled**: the run pauses again.

```python
from agentbarrier.runtime import RuntimeReconciliation

from agloom import RECONCILIATION_INTERRUPT, ExecutionPaused

try:
    agent.invoke("Refund order A-1042 in full. Use refund request id R-88.", config)
except ExecutionPaused as paused:
    pending = paused.interrupts[0].value
    if pending["type"] == RECONCILIATION_INTERRUPT:
        # An operator checks the payment provider and records the outcome.
        policy.store.reconcile(
            pending["action_id"],
            RuntimeReconciliation.NOT_COMMITTED,
            resolved_by="alice",
            reason="The payment provider shows no refund for request R-88.",
        )
        answer = agent.resume(config)  # the refund runs exactly once
```

**Without a checkpointer**, and for calls your code makes with
`execute_capability` or `PolicyManager.execute`, agentbarrier's
`ActionOutcomeUnknown` is raised instead, chained from the tool's exception
(`__cause__`); its `action` identifies the action to reconcile. Every later
call with the same idempotency key raises it too until the action is
reconciled, and Loop Engineering ends the loop with `execution_failed` rather
than retrying.

A process that dies while a consequential call runs leaves the same unknown
outcome once its execution lease expires.

### Reconciling an action

Operators reconcile in agentbarrier, from the CLI or in code. `committed`
records the proven result, which the next call replays without running the
tool; `not_committed` releases the action, so the next call executes it, and a
rule that requires approval asks for approval again.

=== "CLI"

    ```bash
    agentbarrier approvals list --db agentbarrier.db --status unknown
    agentbarrier approvals reconcile <action-id> --db agentbarrier.db \
      --outcome not_committed --resolved-by alice \
      --reason "The payment provider shows no refund for request R-88."
    ```

=== "Python"

    ```python
    from agentbarrier.runtime import RuntimeReconciliation, RuntimeStatus

    for action in policy.store.list_actions(status=RuntimeStatus.UNKNOWN):
        print(action.action_id, action.tool_name, action.idempotency_key)

    policy.store.reconcile(
        action_id,
        RuntimeReconciliation.COMMITTED,
        resolved_by="alice",
        reason="The payment provider shows refund RF-9 for request R-88.",
        result={"content": "Refund RF-9 issued."},
    )
    ```

A committed reconciliation carries the proven result (`--result-json` on the
CLI). The next call replays it to the model: its `content` field when it has
one, a string as is, and any other value as JSON text. Every reconciliation is
written to the audit receipts.

## Stores and identity

The store holds actions, approvals, results, and receipts:

| Store | Use it for |
|---|---|
| `SQLiteRuntimeStore("agentbarrier.db")` | Development and single-node deployments |
| `PostgresRuntimeStore(dsn, schema="agentbarrier")` | Multi-process and multi-node production (`pip install "agloom[postgres]"`) |

`namespace`, `organization_id`, and `requested_by` bind every action to a
service, a tenant, and a requesting identity. Use a distinct namespace per
agent and the tenant's id as the organization.

## Policies everywhere

- **Every topology.** The policy middleware wraps every tool call of every
  tool-capable agent, always innermost, directly around the tool.
- **Middleware tools.** Tools contributed by middleware are matched by their
  tool names, like your own tools.
- **MCP tools.** Rules match MCP tools by their Agloom names,
  `<server_id>__<tool>`.
- **Direct execution.** `agent.execute_capability(...)` and
  `aexecute_capability(...)` apply the same policy to calls made by your code.
  A denial raises [`PolicyViolationError`][agloom.PolicyViolationError]
  (agentbarrier's `PolicyDenied` for a consequential capability) and emits
  `ToolCallDenied`.
- **Outside agents.** `policy.execute(tool_name, arguments, operation)` and
  `aexecute(...)` run any application operation, such as a step of a
  background job, under the same rules, approvals, and exactly-once
  guarantees as an agent tool of that name.
- **DIRECT answers.** DIRECT offers the model no tools. When middleware is
  configured, the direct answer runs through your middleware stack with the
  policy still in place.
- **Loop Engineering.** A pending approval or an unknown outcome pauses a
  checkpointed run for a human, a denial stops the loop with
  `policy_violation`, and an unknown outcome in a run without a checkpointer
  fails the loop.

## Observability

| Event | Meaning |
|---|---|
| `ToolCallDenied` | A call was blocked before it ran: by the policy, because its arguments were invalid (`reason` is `InvalidArguments`), or because the tool was not offered for the request |
| `ApprovalRequired` | A consequential call is waiting for a reviewer; emitted once per pause |
| `ActionExecuted` | A consequential call ran |
| `ActionReplayed` | A consequential call returned its recorded result without running again |
| `ActionFailed` | A consequential call raised; its outcome is unknown until an operator reconciles it |

The [`PrometheusObserver`][agloom.PrometheusObserver] counts them in
`policy_actions_total{tool,outcome}`.
