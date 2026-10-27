# Writing policies

This guide shows how to express real policies for a
[`PolicyManager`][agloom.PolicyManager]: rules in Python or JSON, argument
conditions, consequential tools and their idempotency keys, approvals and
reviewers, stores, denial handling, and tests. Read
[Policies and approvals](../concepts/policies.md) first for the model.

## Rules in Python

```python
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
)

support_policy = RuntimePolicy(
    version="support-2026-10",
    rules=(
        PolicyRule(
            "block refunds to closed accounts",
            PolicyEffect.DENY,
            tool="refund_order",
            conditions=(ArgumentCondition("account.status", ConditionOperator.EQ, "closed"),),
        ),
        PolicyRule(
            "review refunds over 100",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="refund_order",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
            approval_ttl_seconds=3600,
        ),
        PolicyRule("allow small refunds", PolicyEffect.ALLOW, tool="refund_order"),
        PolicyRule("allow read-only tools", PolicyEffect.ALLOW, tool="lookup_*"),
    ),
)
```

- **Order matters.** Rules are evaluated top to bottom and the first match
  wins. Put specific denials first, approvals next, broad allows last.
- **Default deny.** A call that no rule matches is denied. Every tool the
  agent should use needs an allowing (or approval) rule.
- **Tool globs.** `tool` is a shell-style pattern: `refund_order`, `lookup_*`,
  `crm__*` for every tool of the `crm` MCP server, or `*`.
- **Rule names** must be unique; they appear in denials, approvals, and audit
  receipts. Bump `version` whenever the rules change.

## Argument conditions

A rule matches only when **all** of its conditions hold. A condition compares
the value at a dotted path of the tool's arguments:

| Operator | Matches when the argument |
|---|---|
| `exists` | is present (`value=True`) or absent (`value=False`) |
| `eq`, `ne` | equals or differs from the value |
| `lt`, `le`, `gt`, `ge` | compares to a number or string of the same type |
| `in`, `not_in` | is in, or not in, a list of values |
| `contains` | (a string, list, or object) contains the value |
| `starts_with`, `ends_with` | is a string with that prefix or suffix |

A missing path never matches (except for `exists` with `value=False`), and
values of mismatched types never compare, so conditions fail closed.

Conditions see the arguments the tool runs with: validated, converted to the
tool's types, and with the declared default of every argument the model left
out. An optional argument is therefore usually present, with the value `null`,
even when the model did not set it. To match calls that set an optional
argument, compare its value (`ne` with `null`) instead of testing its presence
with `exists`:

```python
PolicyRule(
    "browser tools never write files",
    PolicyEffect.DENY,
    tool="playwright__*",
    conditions=(ArgumentCondition("filename", ConditionOperator.NE, None),),
)
```

## Rules in JSON

Keep policies in version control and review them like code. A strict JSON
document has `version`, an optional `default` (`"deny"` unless stated), and
`rules`. Unknown keys are rejected.

```json
{
  "version": "support-2026-10",
  "default": "deny",
  "rules": [
    {
      "name": "review refunds over 100",
      "effect": "require_approval",
      "tool": "refund_order",
      "conditions": [{"path": "amount", "operator": "gt", "value": 100}],
      "approval_ttl_seconds": 3600
    },
    {"name": "allow small refunds", "effect": "allow", "tool": "refund_order"},
    {"name": "allow read-only tools", "effect": "allow", "tool": "lookup_*"}
  ]
}
```

Pass the file path, or an equivalent mapping, as `policy=`:

```python
from agloom import PolicyManager

policy = PolicyManager(policy="policies/support.json", store=store)
```

??? example "examples/policies/procurement.json (used by examples/42_policy_custom_rules.py)"

    ```json
    --8<-- "examples/policies/procurement.json"
    ```

## Consequential tools and idempotency keys

Declare every tool with real-world side effects in `consequential=`, mapped to
its **business idempotency key**: the identity of the action in your domain,
not of the model's call.

```python
policy = PolicyManager(
    policy=support_policy,
    store=store,
    consequential={
        "refund_order": "request_id",
        "send_email": lambda arguments: f"{arguments['to']}:{arguments['template']}",
    },
)
```

- A **string** names a string argument of the tool. A **callable** receives
  the canonical arguments and returns the key. The key must be a non-blank
  string.
- The same key always refers to the same action. Calling the tool again with
  the same key and the same arguments returns the recorded result without
  running the tool; reusing a key with different arguments is rejected by
  agentbarrier as a binding error.
- Design tools so the key is a natural part of the request (a request id, an
  order and refund reason, a message id), and say so in the tool docstring so
  the model supplies it.
- Consequential tools must return plain results (not LangGraph `Command`s),
  because results are recorded for replay.
- Declare approval rules only for consequential tools; an approval rule for
  any other tool fails closed with
  [`PolicyConfigurationError`][agloom.PolicyConfigurationError].
- Tools contributed by middleware can be declared consequential like your own
  tools.
- If a consequential tool raises, the action's outcome is recorded as unknown,
  and calls with the same key are refused until an operator reconciles the
  action in agentbarrier. A checkpointed run pauses with a
  `RECONCILIATION_INTERRUPT` meanwhile; a run without a checkpointer raises
  `ActionOutcomeUnknown`. See
  [Failures and reconciliation](../concepts/policies.md#failures-and-reconciliation).
  Let transient errors be retried inside the tool, where you know whether the
  external effect happened.

## Approvals and reviewers

A pending approval pauses an agent that has a checkpointer and raises
[`ExecutionPaused`][agloom.ExecutionPaused]. Reviewers decide in agentbarrier,
wherever suits your organization:

=== "Python"

    ```python
    from agentbarrier import Decision

    policy.store.decide(action_id, Decision.APPROVE, decided_by="alice")
    policy.store.decide(other_id, Decision.REJECT, decided_by="bob", reason="Duplicate")
    ```

=== "CLI"

    ```bash
    agentbarrier approvals list --db agentbarrier.db --status pending
    agentbarrier approvals approve <action-id> --db agentbarrier.db --decided-by alice
    agentbarrier approvals reject <action-id> --db agentbarrier.db --decided-by bob --reason "Duplicate"
    ```

=== "Services"

    agentbarrier also provides an authenticated approval HTTP API
    (`agentbarrier api`), a review dashboard (`agentbarrier dashboard`), and a
    Slack integration (`agentbarrier slack`) that decide the same actions.

Then resume the thread with `agent.resume(config)`: an approved action
executes exactly once, and a rejected or expired one fails closed. Set
`approval_ttl_seconds` on approval rules so stale requests expire.

For PostgreSQL stores, the CLI reads the connection string from an environment
variable you name, so it never appears on the command line:
`--postgres-dsn-env AGENTBARRIER_DSN`.

## Stores

```python
import os

from agentbarrier.runtime import PostgresRuntimeStore, SQLiteRuntimeStore

store = SQLiteRuntimeStore("agentbarrier.db")  # development, single node
store = PostgresRuntimeStore(os.environ["AGENTBARRIER_DSN"], schema="agentbarrier")
```

Install `agloom[postgres]` for the PostgreSQL store. Use one store for all
agents of a deployment, and set `namespace` (the agent or service) and
`organization_id` (the tenant) on each `PolicyManager` to bind actions to them.

## Denials

`on_denied="raise"` (the default) fails the run closed with
[`PolicyViolationError`][agloom.PolicyViolationError] or the agentbarrier
denial. Use `on_denied="inform_model"` when the model should see the denial as
an error tool result and continue with another approach:

```python
policy = PolicyManager(policy=support_policy, store=store, on_denied="inform_model")
```

## Middleware and MCP tools

Tools contributed by middleware, such as `write_todos` from LangChain's
`TodoListMiddleware` or Deep Agents' file tools, are governed by their tool
names like your own tools. The policy denies anything no rule allows, so allow
them explicitly:

```python
PolicyRule("allow planning", PolicyEffect.ALLOW, tool="write_todos")
PolicyRule("allow file reads", PolicyEffect.ALLOW, tool="read_file")
```

MCP tools are named `<server_id>__<tool>` in Agloom, so write rules against
those names:

```python
PolicyRule("allow inventory reads", PolicyEffect.ALLOW, tool="inventory__get_*")
PolicyRule("review stock adjustments", PolicyEffect.REQUIRE_APPROVAL, tool="inventory__adjust_stock")
```

Consequential MCP tools can be declared in `consequential=` by their Agloom
names even though they are discovered at run time.

## Testing policies

Policies are plain data, so test them directly, without a model:

```python
assert support_policy.evaluate("refund_order", {"amount": 50}).effect is PolicyEffect.ALLOW
assert (
    support_policy.evaluate("refund_order", {"amount": 500}).effect
    is PolicyEffect.REQUIRE_APPROVAL
)
assert support_policy.evaluate("delete_account", {}).effect is PolicyEffect.DENY
```

`policy.visible(tool_name)` tells you whether a tool will be offered to the
model at all.

## Examples

??? example "examples/40_policy_allow_deny.py"

    ```python
    --8<-- "examples/40_policy_allow_deny.py"
    ```

??? example "examples/42_policy_custom_rules.py"

    ```python
    --8<-- "examples/42_policy_custom_rules.py"
    ```

??? example "examples/43_policy_mcp.py"

    ```python
    --8<-- "examples/43_policy_mcp.py"
    ```

??? example "examples/44_policy_postgres.py"

    ```python
    --8<-- "examples/44_policy_postgres.py"
    ```
