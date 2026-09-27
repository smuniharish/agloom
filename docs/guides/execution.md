# Choosing an execution mode

Agloom keeps simple requests inexpensive and gives complex work the structure
it needs. You can let Agloom choose, require one topology, constrain the
choice, or explicitly compose a Hybrid.

## Start with automatic selection

```python
agent = create_agent(model=model, tools=tools)
```

For each request, Agloom chooses DIRECT or one of the eight topologies from the
task and the capabilities available to that agent. Use this when requests vary
and you want simple prompts to avoid unnecessary orchestration.

## Require one topology

```python
agent = create_agent(model=model, tools=tools, pattern="planner")
```

A single value is a requirement, not a suggestion. Agloom constructs that
topology directly and does not run selection first.

Choose an explicit topology when your product workflow already determines the
execution shape:

| Need | Pattern |
|---|---|
| Reason while calling tools | `"react"` |
| Delegate to named specialists | `"supervisor"` |
| Run ordered transformations | `"pipeline"` |
| Decompose and aggregate subtasks | `"planner"` |
| Draft, evaluate, and improve | `"reflection"` |
| Allow bounded peer handoffs | `"swarm"` |
| Coordinate through shared findings | `"blackboard"` |
| Compose several topology behaviors | `"hybrid"` |

## Constrain automatic selection

```python
agent = create_agent(
    model=model,
    tools=tools,
    pattern=["planner", "supervisor"],
)
```

This is an allow-list. Agloom selects either Planner or Supervisor for each
request. It does **not** execute both and does **not** create a Hybrid.

## Hybrid composition

Hybrid is an explicit topology with two to four non-Hybrid children:

```python
agent = create_agent(
    model=model,
    tools=tools,
    pattern="hybrid",
    composition=["planner", "supervisor"],
)
```

Use Hybrid only when one request genuinely needs several execution behaviors.
Do not use it as a substitute for an allow-list.

## Decision guide

1. Start with automatic selection for mixed workloads.
2. Set one `pattern` when the application already knows the workflow.
3. Use a pattern list to limit automatic choice for governance or cost control.
4. Use Hybrid only for an intentional composition.
5. Add recursion separately; recursion is an execution capability, not a ninth
   topology.

See the complete [`create_agent()` parameter reference](../api/create-agent.md)
for configuration rules and defaults.
