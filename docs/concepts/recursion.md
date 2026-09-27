# Recursive execution

Recursion is an orthogonal execution capability, not a topology. It is disabled
by default:

```python
agent = create_agent(model=model, tools=[], pattern="planner", recursion=False)
```

When enabled, topology-created subtasks are resolved independently and are
bounded by `RecursionPolicy`:

```python
from agloom import RecursionPolicy, create_agent

agent = create_agent(
    model=model,
    tools=[],
    pattern="planner",
    recursion=True,
    recursion_policy=RecursionPolicy(max_depth=2, max_subtasks=8),
)
```

Depth, task count, worker count, and execution time limits apply to an
invocation and its descendants. See the
[`RecursionPolicy` reference](../api/configuration.md#recursionpolicy).
