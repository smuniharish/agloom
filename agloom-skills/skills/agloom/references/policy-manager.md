# Policy Manager

`PolicyManager` puts every tool call of an agent behind an
[agentbarrier](https://pypi.org/project/agentbarrier/) runtime policy. The
application owns the rules, the store, and the identity that actions are bound
to.

```python
from agloom import PolicyManager

policy = PolicyManager(
    policy="policies/procurement.json",   # RuntimePolicy, JSON file path, or mapping
    store=store,                          # SQLiteRuntimeStore or PostgresRuntimeStore
    consequential={"create_purchase_order": "request_id"},
    namespace="procurement-agent",
    organization_id="acme",
    requested_by="procurement-agent",
    on_denied="raise",                    # or "inform_model"
)
```

## Writing rules

Rules are evaluated in order, the first match wins, and the default effect is
`deny`. `tool` matches names with shell-style patterns (`lookup_*`).
Conditions read argument paths with the operators `exists`, `eq`, `ne`, `lt`,
`le`, `gt`, `ge`, `in`, `not_in`, `contains`, `starts_with`, and `ends_with`.
Optional arguments the model left out reach conditions with their default,
usually `null`, so test them with `ne` and `null`, not `exists`.

The same policy as versioned JSON:

```json
{
  "version": "procurement-2026-10",
  "default": "deny",
  "rules": [
    {"name": "approved vendors only", "effect": "deny", "tool": "create_purchase_order",
     "conditions": [{"path": "vendor", "operator": "not_in", "value": ["Globex", "Initech"]}]},
    {"name": "review orders over 5000", "effect": "require_approval",
     "tool": "create_purchase_order",
     "conditions": [{"path": "amount", "operator": "gt", "value": 5000}],
     "approval_ttl_seconds": 86400},
    {"name": "allow purchase orders", "effect": "allow", "tool": "create_purchase_order"},
    {"name": "allow invoice reads", "effect": "allow", "tool": "lookup_*"}
  ]
}
```

Keep policies in the application, for example one file per tenant or
environment, and choose the file when you build the `PolicyManager`. Rule
names must be unique.

## Consequential tools

Declare tools with side effects in `consequential`, mapping each tool to its
business idempotency key: the name of a string argument, or a callable that
derives the key from the arguments. agentbarrier then provides:

- **Approvals** bound to the exact arguments, with expiry.
- **Exactly-once execution** for each key; a repeated call replays the stored
  result instead of running the tool again.
- **Audit receipts** for every decision and execution.

Results must be JSON-compatible, because agentbarrier stores them for replay.
A rule that requires approval for a tool not declared consequential raises
`PolicyConfigurationError`.

A duplicate model call made while the same action runs (for example two
identical tool calls in one turn) waits and replays the result. A duplicate
direct call (`execute_capability`, `policy.execute`) raises agentbarrier's
`ActionInProgress`, so the application can report "in progress" or retry.

## Arguments

Every call is validated against the tool's schema before the policy sees it,
so rules judge the values the tool receives: the tool's own Pydantic model
applies its conversions, validators, and defaults (`"5000"` meets
`amount > 100` as `5000`), and JSON schemas (MCP tools) are checked strictly
with top-level defaults filled in. An invalid call, or a missing or blank
idempotency key, never runs and records no action: the model gets an error
tool message and can fix the call, and `ToolCallDenied` reports reason
`InvalidArguments`. `execute_capability` raises `CapabilityResolutionError`
instead.

`create_agent` requires every governed tool to declare its arguments with a
Pydantic v2 model or a JSON schema that has a JSON form, and every string
idempotency key to name an argument that can be a string. A tool without an
`args_schema` counts only when its `_run` signature names every argument (no
`*args`, `**kwargs`, or `RunnableConfig` parameter); give a plain single-input
`Tool` an `args_schema`.

Tools that are not consequential are checked statelessly. `allow` runs the
tool. `deny` raises `PolicyViolationError`, or, with
`on_denied="inform_model"`, returns the denial to the model as an error tool
message so it can adjust. Tools denied for every argument are never offered to
the model.

## Approval flow

With a checkpointer, an approval pauses the run:

```python
from agentbarrier import Decision
from agloom import APPROVAL_INTERRUPT, ExecutionPaused

try:
    agent.invoke("Order 120 laptops from Globex", config)
except ExecutionPaused as paused:
    request = paused.interrupts[0].value
    assert request["type"] == APPROVAL_INTERRUPT
    store.decide(request["action_id"], Decision.APPROVE, decided_by="alice")
    answer = agent.resume(config)   # runs the approved call exactly once
```

The payload also carries `tool`, `arguments`, `policy_rule`, `namespace`,
`organization_id`, and `expires_at_ns`. Reviewers can also decide through the
agentbarrier CLI, approval API, dashboard, or Slack integration. A rejected or
expired call fails the run with agentbarrier's `ApprovalRejected` or
`ApprovalExpired`, or, with `on_denied="inform_model"`, is reported to the
model as not executed. Without a checkpointer, agentbarrier's
`ApprovalRequired` propagates instead.

## Failures and reconciliation

If a consequential tool raises, agentbarrier cannot know whether the side
effect happened, so it records the outcome as unknown and refuses to run the
action again until an operator reconciles it. The `ActionFailed` event carries
the `action_id`.

With a checkpointer, the run pauses with a `RECONCILIATION_INTERRUPT` payload
(`action_id`, `tool`, `arguments`, and the `error` type name). After
reconciliation, resuming retries the call:

```python
from agentbarrier.runtime import RuntimeReconciliation
from agloom import RECONCILIATION_INTERRUPT, ExecutionPaused

try:
    agent.invoke("Order 120 laptops from Initech", config)
except ExecutionPaused as paused:
    request = paused.interrupts[0].value
    assert request["type"] == RECONCILIATION_INTERRUPT
    store.reconcile(request["action_id"], RuntimeReconciliation.COMMITTED,
                    resolved_by="alice", reason="confirmed with the vendor",
                    result={"content": "PO-881 created", "status": "success"})
    answer = agent.resume(config)
```

- **Committed:** the operator's result is replayed to the model. Agloom's
  `{"content", "status"}` shape and plain strings are used as-is, and any other
  value is shown as JSON text.
- **Not committed:** the call runs again, exactly once.
- **Not yet reconciled:** the run pauses again.

Without a checkpointer, and for direct `execute_capability` and
`policy.execute` calls, agentbarrier's `ActionOutcomeUnknown` is raised,
chained from the tool's error; its `.action` identifies the action.

## Other entry points

- `agent.execute_capability(name, input)` and `aexecute_capability` run a
  registered capability directly under the same policy.
- `policy.execute(tool, arguments, operation)` and `aexecute` run any
  application operation under the policy, outside an agent.
- MCP tools are named `<server_id>__<tool>`, so rules can target them, for
  example `tool="github__*"`.
- Install `agloom[postgres]` for `PostgresRuntimeStore`, or
  `agloom[postgres-binary]` on hosts without a PostgreSQL client library.
