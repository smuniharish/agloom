# Integrations via LangChain middleware

Ecosystem behavior plugs into Agloom through one standard contract: LangChain's
`AgentMiddleware`, passed with `middleware=`. There are no Agloom-specific
adapters to learn. The same middleware instance applies to every tool-capable
agent in every topology (the ReAct agent, every Supervisor and Swarm worker,
every Pipeline stage, every Planner step, the Reflection drafter and reviser,
every Blackboard contributor, and every Hybrid child) and to DIRECT answers.

![Middleware pipeline](../assets/diagrams/middleware-pipeline-light.png#only-light)
![Middleware pipeline](../assets/diagrams/middleware-pipeline-dark.png#only-dark)

## The middleware stack

Around every model call and tool call, the stack runs in this order:

1. **Tool routing (Agloom, outermost).** Offers the model only the routed
   tools for the request (none for DIRECT), adds MCP tools resolved for the
   request, answers calls to agent tools that were not offered with an error
   (`ToolCallDenied`, reason `ToolNotOffered`), checks cooperative
   cancellation, and enforces the loop's tool-call budget.
2. **Your middleware**, in the order you list it.
3. **Policy enforcement (Agloom, innermost).** Applies the
   [`PolicyManager`][agloom.PolicyManager] directly around the tool.

Because policy is innermost, a middleware that retries tool calls, such as
LangChain's `ToolRetryMiddleware`, re-enters the policy on every attempt, and
completed consequential actions are replayed instead of executed twice.

!!! note "What middleware wraps"

    Every model call and tool call of every tool-capable agent, and every
    answer. DIRECT offers the model no tools, but when middleware is
    configured, the direct answer is a tool-less agent turn through this same
    stack, policy included. Final answers written by synthesis steps
    (Supervisor, Planner, Blackboard, and Swarm) are written the same way, so
    output-side middleware such as PII redaction covers every answer. Agloom's
    internal structured decisions (task analysis, supervisor routing, planning,
    reviews, and loop assessments) call the model directly.

## Tools contributed by middleware

Middleware can contribute tools through `AgentMiddleware.tools`, for example
`write_todos` from LangChain's `TodoListMiddleware` or the filesystem tools of
Deep Agents' `FilesystemMiddleware`. Agloom registers them automatically as
capabilities with provider `middleware`:

- they are routed per request and shown to task analysis like your own tools;
- they are hidden when the policy denies them for every call;
- they can be declared consequential in the
  [`PolicyManager`][agloom.PolicyManager], with an idempotency key.

With a `PolicyManager`, middleware tools are governed like any other tool.
agentbarrier denies whatever no rule matches, so add allow rules for the
middleware tools the model should use:

```python
PolicyRule("allow planning", PolicyEffect.ALLOW, tool="write_todos")
```

## BehaviorWeave

[BehaviorWeave](https://pypi.org/project/behaviorweave/) detects behavioral
patterns, such as repeated identical tool calls, failure streaks, and
oscillation, and returns interventions. Its LangChain integration,
`BehaviorWeaveMiddleware`, is the right boundary for Agloom:

- **The right signal.** It records real tool-call events with deterministic
  argument fingerprints and their outcomes, exactly what BehaviorWeave's
  repeat, streak, and loop patterns need.
- **Enforcement, not advice.** It blocks `stop` and `pause` decisions before
  the tool runs, and returns guidance the model can act on as the tool result.
- **Replay-correct.** Redelivery is idempotent, keyed by tool-call ids, so
  LangGraph replays and resumes neither double-count calls nor lose decisions.
- **One source of truth.** The middleware is maintained with the engine and
  composes in an explicit order with ContextSage and any other middleware.

Agloom adds one small bridge: [`behavior_signals`][agloom.behavior_signals], an
`on_decision` hook that feeds actionable decisions into
[Loop Engineering](loop-engineering.md). `stop` stops the loop, `pause` waits,
`human_review` and `escalate` request human intervention, `retry` retries, and
`redirect` replans, all still bounded by the `LoopPolicy`.

```python
from behaviorweave import BehaviorEngine, InterventionType
from behaviorweave import PolicyRule as BehaviorRule
from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware

from agloom import LoopPolicy, behavior_signals, create_agent

engine = BehaviorEngine(
    policies=[
        BehaviorRule(
            policy_id="no-repeated-lookups",
            pattern_id="repeated_tool_call",
            threshold=3,
            intervention=InterventionType.STOP,
            message="Stop repeating the same lookup and answer with what you have.",
        )
    ]
)
agent = create_agent(
    model=model,
    tools=[order_status],
    middleware=[
        BehaviorWeaveMiddleware(engine, on_decision=behavior_signals, scope="session-42")
    ],
    loop_policy=LoopPolicy(max_iterations=3),
)
```

BehaviorWeave keeps behavioral state per **scope**. By default the scope is
the LangGraph `thread_id` of the run, which is what you want with a
checkpointer. For runs without a thread, pass `scope=`: a fixed string, or a
callable that derives the scope from the tool-call request.

??? example "examples/16_behavior.py"

    ```python
    --8<-- "examples/16_behavior.py"
    ```

## ContextSage

[ContextSage](https://pypi.org/project/contextsage/) provides
`IntelligentSummarizationMiddleware`, an information-aware replacement for
LangChain's summarization middleware: it decides what must be preserved,
compacts structured content losslessly, summarizes the rest, and verifies that
every must-preserve fact survived.

```python
from contextsage import IntelligentSummarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        IntelligentSummarizationMiddleware(
            model=model,  # the model that writes summaries
            trigger=("tokens", 60_000),
            keep=("messages", 20),
        )
    ],
)
```

??? example "examples/14_context.py"

    ```python
    --8<-- "examples/14_context.py"
    ```

## Deep Agents middleware

Deep Agents' middleware, such as `FilesystemMiddleware`, is standard LangChain
middleware too. See [Agloom and Deep Agents](../getting-started/deep-agents.md#use-them-together).

## LangChain middleware

Every LangChain middleware works, for example:

```python
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolRetryMiddleware,
)

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        PIIMiddleware("email", strategy="redact"),
        ToolRetryMiddleware(max_retries=2),
        ModelCallLimitMiddleware(run_limit=25),
    ],
)
```

## Composing middleware

- **Order is explicit.** List middleware in the order it should wrap calls:
  the first entry is the outermost of your middleware. For example, put
  ContextSage before BehaviorWeave so summarization happens before every model
  call and behavioral checks happen closest to the tools, just outside policy.
- **Instances are shared.** One middleware instance serves every agent of
  every topology. Stateful middleware, such as BehaviorWeave, therefore sees
  the whole run within its scope, across workers.
- **Names must be unique.** LangChain requires distinct middleware names
  within one agent; when you use two instances of the same class, give them
  distinct names, for example with `name=` where the class accepts it.
- **Human-in-the-loop.** LangChain's `HumanInTheLoopMiddleware` adds
  [tool-call review](../guides/human-in-the-loop.md#tool-call-review) in every
  topology. For consequential tools, prefer [policy approvals](policies.md#approvals):
  they are durable, audited, and execute exactly once.
