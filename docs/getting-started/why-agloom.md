# Why Agloom

Calling a model is easy. Running agents in production is not: requests vary
from "hello" to multi-step investigations, tools have real side effects, and a
run that finishes without errors has not necessarily done what was asked.
Agloom is a harness for exactly that gap. It sits on top of LangChain and
LangGraph and adds the decisions those libraries deliberately leave to you.

## The problems Agloom solves

| Problem in production | Without a harness | With Agloom |
|---|---|---|
| Every request pays for orchestration | One planner or multi-agent graph handles every request, including greetings | Each request is analyzed; simple work runs DIRECT, the rest in the lightest suitable topology |
| One graph does not fit every task | Workflow code accumulates branches for each new kind of request | Eight bounded topologies with explicit semantics, selected per request or pinned by you |
| The model plans work it cannot perform | Routing ignores the real tool inventory | Analysis sees the routed capability inventory; actions no tool can perform are flagged, and the answer says what cannot be done |
| Tools change the world | A prompt asks the model to be careful | Policies allow, deny, or require approval before a tool runs; consequential calls execute exactly once and are audited |
| A finished run is not a finished job | The app re-prompts ad hoc, or loops until a timeout | Loop Engineering evaluates the objective, adapts within policy, and stops with an explicit reason |
| Multi-agent work grows without bounds | Recursion, handoffs, and fan-out are unbounded | Depth, child count, parallelism, plan length, handoffs, and wall-clock time are bounded |
| Integrations become parallel systems | Each library needs its own adapter and lifecycle | Local tools, services, and MCP share one catalog; ecosystem behavior plugs in as standard LangChain middleware |
| Operations are opaque | Logs do not connect requests, decisions, and tool calls | A stable event catalog feeds structured logs, Prometheus, and Langfuse |

## Design principles

**The lightest execution that can do the job.** A request that needs no tools
runs DIRECT with one model call. Orchestration is applied only when the
analyzed task calls for it. Explicit configuration always wins: pin one
topology with `pattern="planner"`, or constrain selection with an allow-list.

**Bounded by construction.** Every topology, recursive child, fan-out, and loop
iteration runs under explicit, validated limits. Nothing loops forever by
accident, and every stop has a typed reason.

**Side effects are governed, not suggested.** Policy is enforced in code,
between the model and the tool, by [agentbarrier](https://pypi.org/project/agentbarrier/).
Tools the policy denies outright are never shown to the model.

**Objectives over runs.** [Loop Engineering](../concepts/loop-engineering.md)
wraps any execution in a deterministic outer loop that measures objective
progress rather than trusting a single successful run.

![Inner model/tool loop inside the outer objective loop](../assets/diagrams/inner-outer-loop-light.png#only-light)
![Inner model/tool loop inside the outer objective loop](../assets/diagrams/inner-outer-loop-dark.png#only-dark)

**Standards over adapters.** Agloom agents are LangChain runnables; topologies
are LangGraph graphs; persistence is a LangGraph checkpointer; ecosystem
behavior is LangChain `AgentMiddleware`; remote capabilities are MCP. You keep
using the tools you already know.

**Untrusted text stays data.** Task analysis notes, loop feedback, and previous
results reach the model as clearly labelled data in the user turn, never as
system instructions. Input messages may contain only user and assistant turns;
trusted instructions belong in `system_prompt`.

**Your application owns infrastructure.** Agloom runs inside your process. You
choose the model, the checkpointer, the policy store, the job queue, and the
observability stack.

## What Agloom is not

- **Not a model provider or a hosted platform.** Bring any LangChain chat
  model; deploy wherever your application runs.
- **Not a replacement for LangGraph.** Every topology is a LangGraph graph, and
  the agent composes with other graphs, including as a Deep Agents sub-agent.
- **Not a sandbox.** Policies decide which tool calls may run; isolate code
  execution and credentials at the tool and infrastructure level.

## When Agloom fits

Choose Agloom when an application serves varied requests, some of which need
real orchestration; when tool calls have business consequences that must be
allowed, approved, and executed exactly once; when long-running work must stop
for auditable reasons; or when you need one consistent runtime across local
tools, services, and MCP servers.

If you need a single open-ended agent that plans in files and runs code in a
sandbox, read [Agloom and Deep Agents](deep-agents.md): the two compose well.

## Next steps

- [Install Agloom](installation.md) and run the [Quickstart](quickstart.md).
- Understand the [architecture](../concepts/architecture.md).
- Browse the [examples](../examples/overview.md).
