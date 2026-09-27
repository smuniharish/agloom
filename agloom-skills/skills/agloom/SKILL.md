---
name: agloom
description: Integrate, configure, debug, test, or extend the Agloom Python agent harness for LangChain and LangGraph applications, including automatic versus explicit execution selection, DIRECT mode, eight execution topologies, capability routing, MCP, bounded recursion, checkpointing, HITL, streaming, structured output, explainability, and observability. Use when building or troubleshooting an application that uses create_agent() and needs reliable task-aware orchestration without inventing another agent runtime.
---

# Agloom

Use this skill for the existing `agloom` Python package. Do not create a
parallel agent factory, topology system, capability router, recursive runtime,
checkpoint layer, model-provider abstraction, or observability pipeline.

The primary API is imported from `agloom`:

```python
from agloom import RecursionPolicy, create_agent
```

Agloom owns task-aware execution selection and an isolated agent runtime built
on LangChain and LangGraph. The application owns models, credentials, tools,
business capabilities, persistence, authentication, and deployment.

Authoritative resources:

- [Agloom examples](https://github.com/smuniharish/agloom/tree/master/examples)
- [Agloom documentation](https://agloom.readthedocs.io/en/latest/)

Read [`references/architecture.md`](references/architecture.md) before changing
execution boundaries. Read
[`references/integration.md`](references/integration.md) before adding Agloom
to an application.

## Activate when

Use Agloom when an application needs to:

- construct an agent with `create_agent()`;
- automatically choose DIRECT or one of eight execution topologies;
- force one topology or constrain selection to an allow-list;
- compose topologies explicitly with Hybrid;
- combine local LangChain tools, application capabilities, and MCP
  capabilities;
- apply capability policy, embeddings, retrieval, or reranking;
- enable bounded recursive child execution;
- add LangGraph checkpointing, interrupts, approval, or resume;
- stream, cancel, or inspect a running agent;
- configure feedback, behavior, context, refresh, explainability, or
  structured output integrations;
- connect structured logging, Prometheus, Grafana, or Langfuse;
- debug topology selection, capability resolution, lifecycle, or integration
  failures;
- test an existing Agloom application or add a topology extension.

Do not select it merely because an application uses LangChain, LangGraph, MCP,
tools, or multiple agents without using or evaluating the Agloom harness.

## Required workflow

### Before changing an application

1. Inspect the installed/current Agloom version and dependency manifest.
2. Inspect the existing `create_agent()` call, model, tools, capabilities,
   `pattern`, recursion, checkpoint, integration, and observer options.
3. Identify the requested execution contract: automatic, one explicit
   topology, a constrained allow-list, or explicit Hybrid composition.
4. Verify every argument against the public API and published
   [documentation](https://agloom.readthedocs.io/en/latest/).
5. Start from the closest runnable
   [example](https://github.com/smuniharish/agloom/tree/master/examples).
6. Reproduce failures before changing code, then run the smallest test that
   proves the requested behavior.

### Choose the correct execution path

1. **Automatic selection:** omit `pattern`. Agloom may select DIRECT or one of
   eight topologies based on the task and available capabilities.
2. **One explicit topology:** pass one lowercase topology name. Analysis is
   bypassed and developer intent wins.
3. **Constrained selection:** pass a nonempty list of topology names. Agloom
   selects one candidate; the list does not compose them.
4. **Hybrid composition:** pass `pattern="hybrid"` and an explicit
   `composition` of two to four non-Hybrid topologies.
5. **Simple work:** preserve DIRECT. Do not force graph construction merely
   because tools are configured.
6. **Tool-required work:** provide real LangChain tools and allow ReAct or an
   explicitly configured tool-capable topology.
7. **Recursive work:** enable `recursion=True` and provide a bounded
   `RecursionPolicy`; recursion is not a ninth topology.
8. **Long-running work:** use a caller-provided LangGraph checkpointer, stable
   `configurable.thread_id`, and a `checkpoint_authorizer`.
9. **Remote capabilities:** register the application-owned MCP client through
   Agloom's MCP options; do not implement competing transport or discovery.
10. **Structured output and telemetry:** configure xstructured and observers
    through `create_agent()` rather than wrapping the runnable downstream.

## Integration rules

- Agloom has exactly eight topologies: ReAct, Supervisor, Pipeline, Planner,
  Reflection, Swarm, Blackboard, and Hybrid.
- DIRECT is an execution mode, not a topology.
- A pattern list is an allow-list, not Hybrid composition.
- Explicit `system_prompt`, workers, stages, and composition override automatic
  recommendations.
- Keep credentials in environment or secret infrastructure, never source.
- Use LangChain contracts for models, tools, embeddings, retrievers, and
  rerankers.
- Use LangGraph contracts for checkpointers, commands, interrupts, and graph
  execution.
- Keep capability routing topology-independent. Apply policy before optional
  retrieval and reranking.
- Treat `invoke`/`stream` as synchronous and `ainvoke`/`astream` as
  asynchronous. Use async APIs inside an active event loop.
- Require authorization for every caller-controlled checkpoint thread ID.
- Preserve explicit failures; do not silently downgrade a failed topology to
  DIRECT or hide observer/capability errors.

## Prohibited shortcuts

Do **not**:

- add `create_harness()` or replace `create_agent()` as the primary API;
- document or implement nine topologies;
- treat multiple allowed patterns as a composition;
- run automatic analysis for one explicit topology;
- invent public constructor parameters, environment variables, topology names,
  or capability APIs;
- import private Agloom modules when a public `agloom` export exists;
- add global mutable agent, worker, blackboard, recursion, or observer state;
- bypass capability policy by exposing every discovered MCP tool to the model;
- use a thread ID as proof of checkpoint authorization;
- suppress model, tool, topology, capability, lifecycle, or telemetry failures;
- duplicate LangChain, LangGraph, feedback-manager, behaviorweave,
  ContextSage, mcp-capability-router, refresh-engine, langgraph-xai, or
  xstructured behavior.

## Verification checklist

For an application change, verify the measurable behavior that motivated it:
selected mode/topology, exact tool arguments, capability names, recursion
limit, lifecycle transition, interrupt/resume result, structured object,
observer event, or explicit error. Exercise the real sync/async boundary and
run the application's formatter, linter, type checker, and focused tests.

For changes to this skill, follow
[`../../validation/README.md`](../../validation/README.md). Use the
authoritative [examples](https://github.com/smuniharish/agloom/tree/master/examples)
and [documentation](https://agloom.readthedocs.io/en/latest/) instead of
turning this file into a second product manual.
