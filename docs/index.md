# Agloom

<p class="agloom-tagline">The production-grade agent harness for LangChain and
LangGraph. Agloom chooses the right execution structure for every request, governs
every tool call with explicit policies, and keeps working until the
<em>objective</em> is met, not just until a run finishes.</p>

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Why Agloom](getting-started/why-agloom.md){ .md-button }
[Agloom and Deep Agents](getting-started/deep-agents.md){ .md-button }

```python
from agloom import create_agent

agent = create_agent(model=model, tools=[lookup_order, refund_order])
answer = agent.invoke("Order A-1042 arrived damaged. Refund it if it qualifies.")
print(answer.content)
```

One call to [`create_agent`][agloom.create_agent] returns a LangChain
`Runnable`. By default each request is analyzed first: simple requests run
DIRECT as one plain model call, and requests that need tools, decomposition,
specialists, or review run in one of eight bounded LangGraph topologies, with
policies, budgets, and checkpoints applied the same way everywhere.

![Agloom architecture](assets/diagrams/architecture-light.png#only-light)
![Agloom architecture](assets/diagrams/architecture-dark.png#only-dark)

## What you get

<div class="grid cards" markdown>

-   :material-source-branch: **Task-aware execution**

    ---

    Each request is analyzed against the real capability inventory and runs
    DIRECT or in one of eight topologies: ReAct, Supervisor, Pipeline,
    Planner, Reflection, Swarm, Blackboard, or Hybrid. Pin or constrain the
    choice with `pattern=`.

    [:octicons-arrow-right-24: Execution selection](concepts/execution-selection.md)

-   :material-shield-check-outline: **Governed side effects**

    ---

    A [`PolicyManager`][agloom.PolicyManager] built on agentbarrier allows,
    denies, or requires approval for every tool call. Consequential tools run
    exactly once per business key and leave audit receipts.

    [:octicons-arrow-right-24: Policies and approvals](concepts/policies.md)

-   :material-sync: **Loop Engineering**

    ---

    A deterministic outer loop evaluates the objective, detects stagnation and
    oscillation, adapts within policy, and stops with an explicit reason.
    One-shot by default, so simple requests pay nothing extra.

    [:octicons-arrow-right-24: Loop Engineering](concepts/loop-engineering.md)

-   :material-transit-connection-variant: **One capability catalog**

    ---

    Local LangChain tools, application services, and MCP tools, resources,
    and prompts are filtered, ranked, and exposed per request through one
    router.

    [:octicons-arrow-right-24: Capabilities and MCP](concepts/capabilities.md)

-   :material-puzzle-outline: **Standard LangChain middleware**

    ---

    ContextSage, BehaviorWeave, Deep Agents, and LangChain middleware plug in
    through `middleware=` and apply to every tool-capable agent in every
    topology.

    [:octicons-arrow-right-24: Integrations](concepts/middleware.md)

-   :material-server-network: **Production runtime**

    ---

    Checkpointed resume with thread authorization, streaming, cooperative
    cancellation, bounded recursion, Prometheus metrics, Langfuse traces, and
    structured logs.

    [:octicons-arrow-right-24: Production deployment](guides/production.md)

</div>

## DIRECT plus eight topologies

| Mode | Use it when |
|---|---|
| **DIRECT** | The request needs no tools or orchestration: one model call. |
| **ReAct** | Reasoning interleaved with tool calls. |
| **Supervisor** | A coordinator should delegate to specialist workers. |
| **Pipeline** | Work must pass through fixed, ordered stages. |
| **Planner** | The task decomposes into dependent steps; independent steps run in parallel. |
| **Reflection** | A draft should be reviewed and revised until approved. |
| **Swarm** | Peers should hand control to each other. |
| **Blackboard** | Several specialists contribute to a shared board before synthesis. |
| **Hybrid** | Two to four topologies must run in sequence. |

DIRECT is an execution mode, not a topology. See [Topologies](concepts/topologies.md)
for the graphs, node names, and configuration of each.

## Start here

- New to Agloom? Read [Why Agloom](getting-started/why-agloom.md), then follow
  the [Quickstart](getting-started/quickstart.md).
- Evaluating harnesses? Compare [Agloom and Deep Agents](getting-started/deep-agents.md),
  and learn how to use them together.
- Building for production? Start with [Policies and approvals](concepts/policies.md),
  [Loop Engineering](concepts/loop-engineering.md), and
  [Production deployment](guides/production.md).
- Learning by example? Browse the [47 runnable examples and seven reference
  applications](examples/overview.md).
- Working with a coding agent? Install the [Agloom Agent Skill](agent-skills.md).
