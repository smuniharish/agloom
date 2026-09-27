# Agloom

Agloom is a production-grade Python agent harness built on LangChain and
LangGraph. It keeps straightforward requests on lightweight **DIRECT
execution** and uses one of eight composable execution topologies when a task
benefits from tools, planning, workers, reflection, or explicit composition.

> Agloom provides eight execution topologies plus a direct execution mode.
> DIRECT is not a topology.

## What developers get

- one construction API: `create_agent()`;
- automatic, constrained, or explicit execution selection;
- local LangChain tools, application capabilities, and MCP capabilities in one
  policy-aware catalog;
- bounded recursive execution, streaming, checkpointing, and human approval;
- Prometheus, Grafana, Langfuse, and langgraph-xai observability;
- structured output through xstructured;
- a portable [Agent Skill](agent-skills.md) for coding-agent integrations.

## Start here

- [Understand why Agloom exists](guides/why-agloom.md)
- [Install Agloom](guides/installation.md)
- [Build your first agent](guides/quickstart.md)
- Browse [topology concepts](concepts/topologies.md)
- Review [examples and verified results](examples/overview.md)
- Understand the [execution flow](architecture/ARCHITECTURE_OVERVIEW.md)

## Design

The public API stays small:

```python
from agloom import create_agent

agent = create_agent(model=model, tools=tools)
answer = agent.invoke("Use the configured tools only when the request needs them.")
print(answer.content)
```

An unconstrained agent uses its LLM to analyze requests against the configured
capability inventory, then chooses DIRECT or a topology. Missing required
capabilities fail explicitly. An explicit single pattern bypasses selection;
an allow-list constrains automatic selection. See [pattern
selection](architecture/PATTERN_SELECTION.md).
