# Configuration guide

Agloom configuration is explicit and local to each agent. Start with the
smallest useful setup, then add only the controls your application needs.

## Minimal agent

```python
from agloom import create_agent

agent = create_agent(model=model, tools=tools)
```

Only `model` is required. `tools` defaults to an empty sequence.

## Add bounded recursion

```python
from agloom import RecursionPolicy, create_agent

agent = create_agent(
    model=model,
    pattern="planner",
    recursion=True,
    recursion_policy=RecursionPolicy(
        max_depth=2,
        max_workers=4,
        max_subtasks=12,
        max_execution_time=90,
    ),
)
```

Recursion remains off unless `recursion=True`. Always set limits appropriate
for your latency and cost budget.

## Configure capabilities

```python
agent = create_agent(
    model=model,
    tools=tools,
    capabilities={"document_store": document_store},
    capability_policy=policy,
    capability_retriever=retriever,
    capability_reranker=reranker,
    max_selected_capabilities=10,
)
```

Use `capability_policy` for authorization and governance. Retrieval and
reranking improve relevance; they do not replace policy checks.

## Add production controls

```python
agent = create_agent(
    model=model,
    pattern="planner",
    checkpointer=checkpointer,
    checkpoint_authorizer=authorize_thread,
    interrupt_after=["stage_0_analyze"],
    observers=observers,
)
```

A checkpointer requires one explicit topology. Applications must authorize
every thread identifier; a thread ID is not an access credential.

## Configuration conflicts

Agloom rejects ambiguous combinations instead of choosing silently:

- a custom `capability_router` cannot be combined with router component
  options;
- Hybrid requires `pattern="hybrid"` and two to four child topologies;
- checkpoint interrupts require a checkpointer;
- checkpointing requires one explicit topology;
- `context_model` requires `context_options`;
- MCP registration settings require `mcp_client`;
- refresh source and operation must be supplied together;
- `structured_output_options` requires `structured_output_schema`;
- `xai_enabled=False` cannot be combined with xAI configuration.

For every accepted option, type, requirement status, and default, see the
[`create_agent()` API reference](../api/create-agent.md).
