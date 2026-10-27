---
name: agloom
description: Builds, configures, debugs, and tests Python applications that use Agloom's create_agent on LangChain and LangGraph. Covers automatic or explicit execution selection (DIRECT, ReAct, Supervisor, Pipeline, Planner, Reflection, Swarm, Blackboard, Hybrid), bounded recursion, capability routing and MCP, agentbarrier Policy Manager rules and approvals, Loop Engineering, checkpointing and resume, streaming, structured output, and Prometheus or Langfuse observability. Use when code imports agloom, or when a user wants a governed agent that picks its own execution structure on LangChain or LangGraph.
license: Apache-2.0
metadata:
  package: agloom
  package-version: "1.0.1"
---

# Agloom

Agloom turns one `create_agent()` call into an agent that analyzes each task,
answers simple requests directly, and runs complex work through one of eight
topologies. It is a LangChain `Runnable` built on LangChain and LangGraph. The
application owns models, credentials, tools, persistence, and deployment.

## Before changing code

1. Check the installed version: `python -c "import agloom; print(agloom.__version__)"`.
   This skill describes Agloom 1.0.1.
2. Read the application's `create_agent()` call and note `model`, `tools`,
   `pattern`, `limits`, `policy`, `checkpointer`, and `loop_policy`.
3. Start from the closest runnable example in the
   [examples directory](https://github.com/smuniharish/agloom/tree/master/examples).
4. Reproduce a failure before fixing it, then run the smallest test that
   proves the behavior.

Import only from the `agloom` package root. Every public name is exported
there. Check names and signatures in [references/api.md](references/api.md).

## Build an agent

```python
from agloom import create_agent
from langchain_openai import ChatOpenAI

agent = create_agent(model=ChatOpenAI(model="gpt-5"), tools=[search_orders])
answer = agent.invoke("Which orders shipped late this week?")
print(answer.text)
```

- Text or message-list input returns an `AIMessage`. Mapping input returns a
  mapping with `messages`.
- `ainvoke`, `astream`, and `aresume` are the async forms. Use them inside a
  running event loop.
- Agents that use `mcp=`, `feedback=`, or `refresh=` are async only. Their sync
  `invoke`, `invoke_loop`, `stream`, and `resume` raise `ConfigurationError`.

## Choose how it executes

| Requirement | Configuration |
|---|---|
| Let Agloom decide | Omit `pattern`. One structured analysis call picks DIRECT or a topology. |
| Always one topology | `pattern="planner"`. Skips analysis. |
| Choose among a few | `pattern=["planner", "supervisor"]`, an allow-list (not a composition); DIRECT is not used. |
| Combine topologies | `pattern="hybrid", composition=["react", "reflection"]`, two to four children. |
| Named specialists | `workers=[WorkerSpec(name=..., description=...)]` for Supervisor, Swarm, Blackboard. |
| Fixed stages | `stages=[PipelineStage(name=..., instruction=...)]` for Pipeline. |
| Recursive subtasks | `limits=ExecutionLimits(max_depth=1)`. Planner steps then run as child executions. |

DIRECT is an execution mode, not a topology. Every topology runs its workers
and steps as tool-capable LangChain agents with the routed tools and the
middleware stack. See [references/topologies.md](references/topologies.md)
for when to use each one and the node names for breakpoints.

## Govern tool calls

Pass a `PolicyManager` to put every tool call behind an
[agentbarrier](https://pypi.org/project/agentbarrier/) policy:

```python
from agentbarrier.runtime import (
    ArgumentCondition, ConditionOperator, PolicyEffect, PolicyRule,
    RuntimePolicy, SQLiteRuntimeStore,
)
from agloom import PolicyManager

policy = PolicyManager(
    policy=RuntimePolicy(version="support-v1", rules=(
        PolicyRule("review large refunds", PolicyEffect.REQUIRE_APPROVAL, tool="refund",
                   conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),)),
        PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="refund"),
        PolicyRule("allow lookups", PolicyEffect.ALLOW, tool="lookup_*"),
    )),
    store=SQLiteRuntimeStore("agentbarrier.db"),
    consequential={"refund": "request_id"},
)
agent = create_agent(model=model, tools=[refund, lookup_order], policy=policy,
                     checkpointer=saver, checkpoint_authorizer=owns_thread)
```

- The default effect is deny, so every tool needs an allowing rule. Rules match
  tool names with shell-style patterns, and the first match wins.
- Policies can be a `RuntimePolicy`, a JSON file path, or a mapping, so
  applications can load their own rules per tenant or environment.
- Consequential tools get approval binding, exactly-once execution keyed by the
  business idempotency key, stored-result replay, and audit receipts.
- Arguments are validated against the tool's schema before the policy judges
  them, so rules see the values the tool receives; invalid calls go back to the
  model and never run.
- A consequential tool that raises has an unknown outcome. With a checkpointer
  the run pauses until an operator reconciles the action in agentbarrier, then
  `agent.resume(config)` continues; without one, agentbarrier's
  `ActionOutcomeUnknown` is raised with the action.
- `policy.execute(tool, arguments, operation)` governs application operations
  outside tool calls with the same rules.

Read [references/policy-manager.md](references/policy-manager.md) before
writing rules, approval flows, or MCP policies.

## Persist, pause, and resume

- A `checkpointer` requires a `checkpoint_authorizer(thread_id) -> bool`.
  A thread ID is not proof of ownership.
- Pass `config={"configurable": {"thread_id": ...}}`. With a checkpointer,
  each turn needs only the new user message.
- A pause raises `ExecutionPaused` with `.interrupts` and `.thread_id`.
  Interrupt payload types are `APPROVAL_INTERRUPT`, `RECONCILIATION_INTERRUPT`,
  `LOOP_INTERRUPT`, and `BREAKPOINT_INTERRUPT`.
- `agent.resume(config, value)` answers every pending interrupt.
  `agent.resume(config)` also continues a thread that stopped at a breakpoint
  or after a crash. Completed tools and loop iterations are not repeated, and
  the run keeps the plan it paused in.
- Inside another LangGraph graph, give the agent the graph's checkpointer and
  pass the node's `config` to `agent.invoke`: pauses become the graph's
  interrupts, and resuming the graph resumes the agent.
- `interrupt_before` and `interrupt_after` take topology node names and need
  exactly one explicit `pattern`.

## Iterate with Loop Engineering

Loops are off by default: one attempt and no extra model calls. Raise
`LoopPolicy.max_iterations` to let the agent evaluate its result and try again:

```python
from agloom import LoopPolicy

agent = create_agent(model=model, tools=tools,
                     loop_policy=LoopPolicy(max_iterations=4, allow_replan=True, max_replans=1))
result = await agent.ainvoke_loop("Draft the migration plan")
result.state.termination_reason, result.state.iteration
```

With more than one iteration and no custom loop observer or evaluator,
`ModelLoopEvaluator` judges each result against the objective. It only
recommends steps the policy allows. Human feedback from `resume` or a
`FeedbackManager` applies to every later iteration. See
[references/loop-engineering.md](references/loop-engineering.md).

## Stream, structure, and observe

- `agent.stream(task, stream_mode="messages")` yields model tokens.
  `stream_mode="custom"` yields Agloom events such as `TopologySelected`.
- `agent.with_structured_output(Schema)` returns a runnable whose result has a
  validated `.structured` value.
- `observers=[LoggingObserver(), PrometheusObserver(), LangfuseObserver(...)]`
  or plain callables receive `AgentEvent`s. Events never contain prompts, tool
  arguments, or credentials.

MCP servers, middleware (ContextSage, BehaviorWeave, LangChain), feedback,
source refresh, explainability, and checkpointer packages are covered in
[references/integrations.md](references/integrations.md).

## Rules

- Use LangChain models, tools, middleware, embeddings, and retrievers, and
  LangGraph checkpointers, interrupts, and commands. Do not build parallel
  versions of them or of Agloom features.
- Keep credentials in the environment or a secret store, never in code.
- Do not bypass the Policy Manager by calling tools directly. Use
  `agent.execute_capability(name, input)`, which enforces the same policy.
- Register MCP servers with `refresh=RefreshPolicy(on_register=True)`, or no
  MCP capabilities are discovered.
- Let failures surface. Agloom never silently falls back to DIRECT.
  Do not catch `AgloomError` just to hide it.
- Do not import private modules (names starting with `_`) or rely on internal
  configuration keys.

## Verify

Confirm the observable behavior that motivated the change:

- the selected mode or topology (`TopologySelected` event);
- tool calls, policy outcomes, and the approval or resume result;
- loop termination reason and iteration count;
- the structured value or the raised error.

Test with a scripted LangChain chat model rather than live calls where
possible. Run the application's formatter, linter, type checker, and tests.
Troubleshooting steps for common errors are in
[references/troubleshooting.md](references/troubleshooting.md).

Full documentation: <https://agloom.readthedocs.io/en/latest/>
