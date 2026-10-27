# Agloom and Deep Agents

!!! info "Fact-checked in October 2026"

    Compared against **Agloom 1.0.1** and **deepagents 0.7.22** (the Deep
    Agents package), using the installed package source, its documentation,
    and its README. Both projects evolve quickly; check the
    [Deep Agents documentation](https://docs.langchain.com/oss/python/deepagents/overview)
    for later changes.

[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) is
LangChain's opinionated, single-agent harness: one agent built on LangChain's
`create_agent`, equipped with a virtual filesystem, sub-agents, summarization,
memory, and skills, where the model decides what to do next. Agloom is a
task-aware execution harness: it analyzes each request, runs it DIRECT or in
one of eight bounded topologies, governs every tool call with policy, and wraps
execution in a deterministic objective loop.

They solve different problems, share the same LangChain and LangGraph
foundations, and compose in both directions.

## At a glance

| Dimension | Agloom 1.0.1 | Deep Agents 0.7.22 |
|---|---|---|
| **Primary abstraction** | `create_agent(...)` returns an `Agent` (a LangChain `Runnable`) that chooses an execution structure per request | `create_deep_agent(...)` returns one LangChain `create_agent` graph with a preconfigured middleware stack |
| **Control flow** | Developer-bounded: a deterministic strategy selects the structure; each topology is an explicit LangGraph graph with enforced limits | Model-driven: the model decides which tools, files, and sub-agents to use inside one agent loop |
| **Execution selection** | Structured task analysis, then a deterministic strategy picks DIRECT or one of eight topologies; pin with `pattern="planner"` or constrain with an allow-list | One harness for every request; behavior is shaped by the prompt, tools, middleware, and model harness profiles |
| **Orchestration structures** | ReAct, Supervisor, Pipeline, Planner (dependency waves in parallel), Reflection, Swarm, Blackboard, Hybrid; bounded recursion | Sub-agents through a `task` tool, each with an isolated context window; a general-purpose sub-agent is added by default; synchronous and asynchronous sub-agents |
| **Objective control and termination** | Loop Engineering: objective evaluation; iteration, duration, tool-call, retry, and replan budgets; progress, stagnation, and oscillation detection; typed termination reasons; checkpointed iterations | The run ends when the model stops calling tools; the beta `RubricMiddleware` re-prompts the agent until a grader is satisfied or `max_iterations` is reached |
| **Governance of side effects** | `PolicyManager` on agentbarrier: ordered allow, deny, and approval rules with argument conditions; consequential tools execute exactly once with audit receipts; tools denied outright are hidden from the model | "Trust the LLM" model: the agent can do anything its tools allow, so boundaries belong at the tool or sandbox level; declarative permissions for the built-in file tools; `interrupt_on` approvals |
| **Capability routing and MCP** | One catalog for local tools, application services, and MCP tools, resources, and prompts; filtering, name matching, semantic retrieval, and reranking per request; MCP servers registered on an `MCPRuntime` | The tools you pass, merged with the built-in tools; MCP tools are passed as ordinary LangChain tools |
| **Context, filesystem, and planning** | Plug in context management as middleware (for example ContextSage or Deep Agents' `FilesystemMiddleware`); the Planner topology plans structurally as a validated dependency graph | Built in: virtual filesystem with pluggable backends, sandboxed `execute`, summarization and large-result offloading, `AGENTS.md` memory, and skills; `write_todos` planning is opt-in since 0.7 |
| **Human-in-the-loop** | Durable policy approvals decided in agentbarrier (CLI, HTTP API, dashboard, or Slack); tool-call review with LangChain's `HumanInTheLoopMiddleware` in every topology; breakpoints on topology nodes; loop-level human intervention; `agent.resume(config)` | `interrupt_on` per tool through LangChain's human-in-the-loop middleware (approve, edit, reject, or respond) |
| **Persistence** | LangGraph checkpointer with a required `checkpoint_authorizer` per thread; loop iterations are checkpointed tasks; a durable agentbarrier store records actions | LangGraph checkpointer; LangGraph store for long-term files and memory (`StoreBackend`) |
| **Cost for simple requests** | Automatic selection: one structured analysis call, then one DIRECT model call; with a pinned `pattern`, no analysis call; the one-shot loop adds no calls | One model call, with the harness prompt and the built-in tool schemas (file tools and `task`) attached |
| **Maturity and dependencies** | 1.0.1, classified Production/Stable; Python 3.12 or later; no model-provider SDK dependencies | 0.7.22, classified Beta (pre-1.0); depends on `langchain-anthropic`, `langchain-google-genai`, and `langsmith` |

## How they differ

**Who decides the structure.** In Deep Agents the model drives: it decides
when to write files, delegate to a sub-agent, or stop. In Agloom the
structure is decided before execution, from a structured analysis and a
deterministic strategy (or your explicit `pattern`), and every topology runs
under validated limits. Agloom still lets the model reason and call tools
inside each worker; it bounds the shape around it.

**What "done" means.** A Deep Agent finishes when the model stops calling
tools; its beta `RubricMiddleware` adds a grader that re-prompts the same agent
loop. Agloom's [Loop Engineering](../concepts/loop-engineering.md) is an outer
loop around any execution, including a whole multi-agent topology: it measures
objective progress, enforces budgets, detects stagnation and oscillation,
adapts only as the `LoopPolicy` allows, and records why it stopped.

**How side effects are controlled.** Deep Agents follows a "trust the LLM"
model and recommends enforcing boundaries in tools and sandboxes. Agloom
enforces policy between the model and every tool: denied calls never run,
approvals pause durably, and consequential calls execute exactly once per
business key, with audit receipts.

**Where context management lives.** Deep Agents ships a rich context harness:
filesystem, offloading, memory, and skills. Agloom keeps context management in
middleware you choose, and applies it to every tool-capable agent in every
topology.

## When to choose which

**Choose Deep Agents** for open-ended, long-horizon work by a single agent:
research, coding, or analysis that benefits from a scratch filesystem, a
sandbox, memory files, and skills, where you want the model to drive.

**Choose Agloom** for applications that serve varied requests and need the
right structure for each; where tool calls have business consequences that
must be allowed, approved, and executed exactly once; where cost must stay
proportional to the request; and where long-running work must stop for
auditable reasons.

**Use both** when an open-ended agent needs governed operations, or a governed
application needs Deep Agents' context tooling.

## Use them together

![Composing Agloom and Deep Agents](../assets/diagrams/deep-agents-composition-light.png#only-light)
![Composing Agloom and Deep Agents](../assets/diagrams/deep-agents-composition-dark.png#only-dark)

### Deep Agents middleware inside Agloom

Deep Agents' middleware is standard LangChain `AgentMiddleware`, so it plugs
into `middleware=` and applies to every tool-capable agent in every topology.
The tools it contributes, such as `ls`, `read_file`, and `write_file`, are
registered as Agloom capabilities: they are routed per request, governed by
your policy like any other tool, and can be declared consequential.

```python
from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware

from agloom import create_agent

agent = create_agent(
    model=model,
    pattern="react",
    middleware=[
        FilesystemMiddleware(backend=FilesystemBackend(root_dir="workspace", virtual_mode=True))
    ],
)
answer = agent.invoke("Read /runbook.md and tell me the exact restart command.")
```

!!! note

    Each worker in a multi-agent topology is its own LangChain agent with its
    own state. Use a backend that persists outside the graph state, such as
    `FilesystemBackend` or `StoreBackend`, when workers or runs must share
    files. With a `PolicyManager`, add rules that allow the file tools you want
    to expose: the policy denies anything no rule allows.

### An Agloom agent inside a Deep Agent

An Agloom agent is a LangChain runnable that accepts and returns `messages`,
so it can serve as a Deep Agents `CompiledSubAgent`. The Deep Agent delegates
through its `task` tool; the Agloom agent keeps its own execution selection,
policies, approvals, and budgets.

```python
from deepagents import CompiledSubAgent, create_deep_agent

operations = create_agent(model=model, tools=[order_status], policy=policy)

deep_agent = create_deep_agent(
    model=model,
    subagents=[
        CompiledSubAgent(
            name="order-operations",
            description="Answers questions about orders using the order systems.",
            runnable=operations,
        )
    ],
)
result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "Ask order-operations where order A-1001 is."}]}
)
```

!!! tip "Replacing the default sub-agent"

    Deep Agents adds a `general-purpose` sub-agent built with LangChain's
    `create_agent(name=...)`, which sets `name` on every assistant message.
    Some OpenAI-compatible providers reject that field. Registering your Agloom
    agent under the name `general-purpose` replaces the default sub-agent;
    Agloom never sets `name` on assistant messages.

### Complete example

[`examples/46_deep_agents.py`](https://github.com/smuniharish/agloom/blob/master/examples/46_deep_agents.py)
demonstrates both directions against a live model.

??? example "examples/46_deep_agents.py"

    ```python
    --8<-- "examples/46_deep_agents.py"
    ```
