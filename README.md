# Agloom

**The production-grade agent harness for LangChain and LangGraph.** Agloom
chooses the right execution structure for every request, governs every tool
call with explicit policies, and keeps working until the *objective* is met,
not just until a run finishes.

[![PyPI version](https://img.shields.io/pypi/v/agloom)](https://pypi.org/project/agloom/)
[![Python versions](https://img.shields.io/pypi/pyversions/agloom)](https://pypi.org/project/agloom/)
[![License](https://img.shields.io/pypi/l/agloom)](https://github.com/smuniharish/agloom/blob/master/LICENSE)
[![Documentation](https://readthedocs.org/projects/agloom/badge/?version=latest)](https://agloom.readthedocs.io/en/latest/)
[![CI](https://github.com/smuniharish/agloom/actions/workflows/ci.yml/badge.svg)](https://github.com/smuniharish/agloom/actions/workflows/ci.yml)

[Documentation](https://agloom.readthedocs.io/en/latest/) ·
[Quickstart](https://agloom.readthedocs.io/en/latest/getting-started/quickstart/) ·
[Agloom and Deep Agents](https://agloom.readthedocs.io/en/latest/getting-started/deep-agents/) ·
[Examples](https://agloom.readthedocs.io/en/latest/examples/overview/) ·
[API reference](https://agloom.readthedocs.io/en/latest/api/agent/) ·
[Changelog](https://github.com/smuniharish/agloom/blob/master/CHANGELOG.md)

## What and why

Calling a model is easy. Running agents in production is not: requests range
from "hello" to multi-step investigations, tools have real side effects, and a
run that ends without errors has not necessarily done what was asked.

Agloom is one `create_agent` call that returns a LangChain `Runnable`. For
every request it analyzes the task against the tools it can actually use and
runs it **DIRECT** (one model call) or in one of **eight bounded LangGraph
topologies**. Every tool call passes through an **agentbarrier policy**, and
every execution runs inside **Loop Engineering**, a deterministic outer loop
that judges the objective, adapts within limits, and stops for explicit,
auditable reasons.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/architecture-dark.png">
  <img alt="Agloom architecture" src="https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/architecture-light.png" width="650">
</picture>

## Key features

- **Task-aware execution.** DIRECT or ReAct, Supervisor, Pipeline, Planner,
  Reflection, Swarm, Blackboard, or Hybrid, chosen per request from a
  structured analysis, or pinned and constrained with `pattern=`.
- **Policy-governed tools.** Allow, deny, or require approval for every tool
  call, judged on the validated arguments the tool actually receives;
  consequential tools execute exactly once per business key, with audit
  receipts; tools a policy always denies are never shown to the model.
- **Loop Engineering.** Objective evaluation; iteration, duration, tool-call,
  retry, and replan budgets; stagnation and oscillation detection;
  policy-bounded retry, replan, refresh, topology change, and human
  intervention; checkpointed iterations. One-shot by default, at no extra cost.
- **One capability catalog.** Local LangChain tools, application services, and
  MCP tools, resources, and prompts, filtered and ranked per request.
- **Standard LangChain middleware.** ContextSage, BehaviorWeave, Deep Agents,
  and LangChain middleware apply to every tool-capable agent in every topology.
- **Production runtime.** Checkpointed resume with thread authorization,
  human-in-the-loop, streaming, cooperative cancellation, bounded recursion,
  structured output, Prometheus metrics, Langfuse traces, and structured logs.

## Install

```bash
pip install agloom
pip install "agloom[postgres]"          # PostgreSQL checkpointer and policy store
pip install "agloom[postgres-binary]"   # the same, without a system libpq
pip install "agloom[sqlite]"            # SQLite checkpointer
```

Agloom requires Python 3.12 or later. Install the LangChain integration for
your model provider too, for example `pip install langchain-openai`.

## Quickstart

```python
import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agloom import create_agent


@tool
def order_status(order_id: str) -> str:
    """Return the shipping status of an order, for example A-1001."""
    statuses = {
        "A-1001": "delivered on 2 October",
        "A-1002": "in transit, arriving 8 October",
    }
    return statuses.get(order_id.strip().upper(), "unknown order")


# Any LangChain chat model works. ChatOpenAI reads OPENAI_API_KEY, and
# OPENAI_BASE_URL for OpenAI-compatible endpoints.
model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])
agent = create_agent(model=model, tools=[order_status])

print(agent.invoke("Write a one-line welcome for a new customer.").content)
print(agent.invoke("Where is order A-1002?").content)
```

Output from a live run:

```text
Welcome to our community—we’re delighted to have you as a new customer!
Order **A-1002** is **in transit** and is expected to arrive on **8 October**.
```

The greeting runs DIRECT with one answer call; the order question runs the
ReAct topology, which calls `order_status` once. Pin a topology with
`pattern="react"` to skip the analysis call, or constrain the choice with an
allow-list such as `pattern=["react", "reflection"]`.

## Policies in brief

Policy is enforced in code, between the model and every tool, by
[agentbarrier](https://pypi.org/project/agentbarrier/). Rules are ordered, the
first match wins, and anything no rule allows is denied. Here a refund over 100
pauses the run until a reviewer approves it, and then executes exactly once:

```python
from agentbarrier import Decision
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langgraph.checkpoint.memory import InMemorySaver

from agloom import ExecutionPaused, PolicyManager


@tool
def refund_order(order_id: str, amount: float, request_id: str) -> str:
    """Refund an order. request_id uniquely identifies the refund request."""
    return f"refunded {amount} EUR for order {order_id}"


policy = PolicyManager(
    policy=RuntimePolicy(
        version="support-v1",
        rules=(
            PolicyRule(
                "review large refunds",
                PolicyEffect.REQUIRE_APPROVAL,
                tool="refund_order",
                conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
            ),
            PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="refund_order"),
            PolicyRule("allow status lookups", PolicyEffect.ALLOW, tool="order_status"),
        ),
    ),
    store=SQLiteRuntimeStore("agentbarrier.db"),
    consequential={"refund_order": "request_id"},  # business idempotency key
)
support = create_agent(
    model=model,
    tools=[order_status, refund_order],
    policy=policy,
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=lambda thread_id: thread_id.startswith("ticket-"),
)

config = {"configurable": {"thread_id": "ticket-4711"}}
try:
    support.invoke("Refund 240 EUR for order A-1001 with request id R-88.", config)
except ExecutionPaused as paused:
    action_id = paused.interrupts[0].value["action_id"]
    policy.store.decide(action_id, Decision.APPROVE, decided_by="alice")
    print(support.resume(config).content)  # the approved refund runs exactly once
```

```text
Refund completed: **240 EUR** for order **A-1001**. Request ID: **R-88**.
```

The same policy governs tools contributed by middleware and the calls your own
code makes with `execute_capability`. Side effects fail closed: if a
consequential tool raises, its outcome is recorded as unknown and the action is
held for reconciliation in agentbarrier instead of being retried blindly. A
checkpointed run pauses meanwhile and resumes once an operator has reconciled
the action.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/policy-enforcement-dark.png">
  <img alt="Policy enforcement" src="https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/policy-enforcement-light.png" width="871">
</picture>

## Loop Engineering in brief

A successful execution is not the same as an achieved objective. Loop
Engineering wraps any execution in an outer loop that evaluates the result
against the objective, carries feedback into the next iteration as data, and
stops with a typed termination reason:

```python
from agloom import LoopPolicy

looping = create_agent(
    model=model,
    tools=[order_status],
    loop_policy=LoopPolicy(max_iterations=3, allow_replan=True, max_replans=1),
)
result = looping.invoke_loop(
    "Report the status of orders A-1001 and A-1002 as a two-row Markdown table."
)
print(result.state.iteration, result.state.termination_reason)
```

```text
1 objective_satisfied
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/inner-outer-loop-dark.png">
  <img alt="Inner and outer loops" src="https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/inner-outer-loop-light.png" width="620">
</picture>

Budgets bound iterations, active time, tool calls, retries, and replans;
stagnation and oscillation are detected deterministically; and retry, replan,
refresh, topology change, and human intervention happen only when the
`LoopPolicy` allows them. The default policy is one-shot and adds no model
calls. Read [why Loop Engineering matters](https://agloom.readthedocs.io/en/latest/concepts/loop-engineering/).

## DIRECT plus eight topologies

| Mode | Use it when |
|---|---|
| **DIRECT** | The request needs no tools or orchestration: one model call |
| **ReAct** | Reasoning interleaved with tool calls |
| **Supervisor** | A coordinator should delegate to specialist workers |
| **Pipeline** | Work must pass through fixed, ordered stages |
| **Planner** | The task decomposes into dependent steps; independent steps run in parallel |
| **Reflection** | A draft should be reviewed and revised until approved |
| **Swarm** | Peers should hand control to each other |
| **Blackboard** | Specialists contribute to a shared board before synthesis |
| **Hybrid** | Two to four topologies must run in sequence |

Every worker in every topology is a tool-capable LangChain agent with the
routed tools, your middleware, and the policy. See
[Topologies](https://agloom.readthedocs.io/en/latest/concepts/topologies/).

## Agloom and Deep Agents

[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) is
LangChain's opinionated single-agent harness, with a filesystem, sub-agents,
memory, and skills, where the model drives. Agloom is a task-aware execution
harness with bounded topologies, policy-governed side effects, and Loop
Engineering. They compose in both directions: Deep Agents middleware plugs into
Agloom through `middleware=`, and an Agloom agent can serve as a Deep Agents
`CompiledSubAgent`. Read the
[dated, fact-checked comparison](https://agloom.readthedocs.io/en/latest/getting-started/deep-agents/).

## Learn more

- **Documentation:** [agloom.readthedocs.io](https://agloom.readthedocs.io/en/latest/),
  including [Why Agloom](https://agloom.readthedocs.io/en/latest/getting-started/why-agloom/),
  [Policies and approvals](https://agloom.readthedocs.io/en/latest/concepts/policies/),
  and [Production deployment](https://agloom.readthedocs.io/en/latest/guides/production/).
- **Examples:** [47 runnable examples](https://github.com/smuniharish/agloom/tree/master/examples),
  each verified against a live model.
- **Reference applications:** seven complete FastAPI and React applications,
  from a [customer-support chatbot](https://github.com/smuniharish/agloom/tree/master/examples/qa_customer_chatbot)
  to an [autonomous product studio](https://github.com/smuniharish/agloom/tree/master/examples/autonomous_product_studio)
  with approvals and background jobs. See the
  [examples overview](https://agloom.readthedocs.io/en/latest/examples/overview/).
- **Agent Skill:** teach your coding agent to build with Agloom; see
  [Agent Skills](https://agloom.readthedocs.io/en/latest/agent-skills/).

## Contributing

Contributions are welcome. Read
[CONTRIBUTING.md](https://github.com/smuniharish/agloom/blob/master/CONTRIBUTING.md)
for the development setup and checks, and report vulnerabilities privately as
described in [SECURITY.md](https://github.com/smuniharish/agloom/blob/master/SECURITY.md).

## License

Agloom is licensed under the
[Apache License 2.0](https://github.com/smuniharish/agloom/blob/master/LICENSE).
Created by S Muni Harish.
