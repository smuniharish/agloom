# Policy

[`PolicyManager`][agloom.PolicyManager] governs every tool call of an agent
with an [agentbarrier](https://pypi.org/project/agentbarrier/) runtime policy.
See [Policies and approvals](../concepts/policies.md) and
[Writing policies](../guides/writing-policies.md).

::: agloom.PolicyManager
    options:
      members:
        - policy
        - store
        - barrier
        - consequential
        - on_denied
        - visible
        - is_consequential
        - execute
        - aexecute

Agents apply the policy to every tool call automatically, and
[`Agent.execute_capability`][agloom.Agent.execute_capability] applies it to
calls made by your code. `execute` and `aexecute` run any application
operation under the policy outside an agent, for example in a background job,
with the same rules, approvals, and exactly-once guarantees as an agent tool of
that name.

::: agloom.APPROVAL_INTERRUPT

The `type` of the LangGraph interrupt payload raised when a consequential tool
call waits for a reviewer. The payload also carries `action_id`, `tool`,
`arguments`, `policy_rule`, `namespace`, `organization_id`, and
`expires_at_ns`.

```python
from agloom import APPROVAL_INTERRUPT, ExecutionPaused

try:
    agent.invoke("Refund order 1042", config)
except ExecutionPaused as paused:
    approvals = [
        interrupt.value
        for interrupt in paused.interrupts
        if interrupt.value.get("type") == APPROVAL_INTERRUPT
    ]
```

::: agloom.RECONCILIATION_INTERRUPT

The `type` of the LangGraph interrupt payload raised when a consequential tool
raised in a checkpointed run, so its outcome is unknown. The payload also
carries `action_id`, `tool`, `arguments`, `error` (the type of the exception
the tool raised), `namespace`, and `organization_id`. After an operator
reconciles the action in agentbarrier, `agent.resume(config)` retries the
call: a committed result is replayed, a not-committed action runs exactly
once, and an unreconciled action pauses the run again. See
[Failures and reconciliation](../concepts/policies.md#failures-and-reconciliation).
