# Quickstart

In a few minutes you will build a tool-using agent, see how Agloom chooses an
execution mode for each request, govern its tools with a policy, and turn on
Loop Engineering. Install Agloom and a model integration first; see
[Installation](installation.md).

## 1. Choose a model

Agloom works with any LangChain chat model. Tools need tool calling; automatic
execution selection needs structured output.

=== "OpenAI-compatible"

    ```python
    import os

    from langchain_openai import ChatOpenAI

    # Reads OPENAI_API_KEY, and OPENAI_BASE_URL for any OpenAI-compatible endpoint.
    model = ChatOpenAI(model=os.environ["OPENAI_MODEL"])
    ```

=== "OpenAI"

    ```python
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-5.5")  # reads OPENAI_API_KEY
    ```

=== "Anthropic"

    ```python
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-4-6")  # reads ANTHROPIC_API_KEY
    ```

## 2. Create an agent

```python
from langchain_core.tools import tool

from agloom import create_agent


@tool
def order_status(order_id: str) -> str:
    """Return the shipping status of an order, for example A-1001."""
    statuses = {
        "A-1001": "delivered on 2 October",
        "A-1002": "in transit, arriving 8 October",
    }
    return statuses.get(order_id.strip().upper(), "unknown order")


agent = create_agent(model=model, tools=[order_status])

print(agent.invoke("Write a one-line welcome for a new customer.").content)
print(agent.invoke("Where is order A-1002?").content)
```

Output from a live run (wording varies by model):

```text
Welcome to our community—we’re delighted to have you as a new customer!
Order **A-1002** is **in transit** and is expected to arrive on **8 October**.
```

For each request, Agloom first analyzes the task against the routed tools. The
greeting needs no tools, so it runs **DIRECT**: one plain model call. The order
question needs `order_status`, so it runs the **ReAct** topology, which calls
the tool once and answers.

`invoke` accepts text, LangChain messages, or a mapping with `messages`, and
returns an `AIMessage` (or a mapping with `messages` for mapping input).

## 3. See what Agloom decided

Every decision is an event. Pass any callable, or a built-in observer such as
[`LoggingObserver`][agloom.LoggingObserver], through `observers=`:

```python
from agloom import AgentEvent, EventName


def show_selection(event: AgentEvent) -> None:
    if event.name == EventName.TOPOLOGY_SELECTED:
        print(f"{event.topology or 'direct'}: {event.payload['reason']}")


traced = create_agent(model=model, tools=[order_status], observers=[show_selection])
traced.invoke("Where is order A-1001?")
```

```text
react: selected from the task requirements
```

See the full [event catalog](../concepts/observability.md#event-catalog).

## 4. Pin or constrain execution

Automatic selection costs one structured analysis call per request. When you
already know the right structure, pin it and skip analysis, or constrain the
choice with an allow-list:

```python
pinned = create_agent(model=model, tools=[order_status], pattern="react")
constrained = create_agent(
    model=model, tools=[order_status], pattern=["react", "reflection"]
)
```

An allow-list always selects one of its topologies, never DIRECT. See
[Execution selection](../concepts/execution-selection.md).

## 5. Govern tool calls with a policy

A [`PolicyManager`][agloom.PolicyManager] enforces an agentbarrier policy on
every tool call. Rules are ordered, the first match wins, and anything no rule
allows is denied. Tools that are denied for every possible call are never shown
to the model.

```python
from agentbarrier.runtime import (
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)

from agloom import PolicyManager


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"order {order_id} cancelled"


policy = PolicyManager(
    policy=RuntimePolicy(
        version="quickstart-v1",
        rules=(PolicyRule("read order status", PolicyEffect.ALLOW, tool="order_status"),),
    ),
    store=SQLiteRuntimeStore("agentbarrier.db"),
)
governed = create_agent(
    model=model, tools=[order_status, cancel_order], policy=policy
)
print(governed.invoke("Cancel order A-1002.").content)
```

```text
I can’t cancel order A-1002 because cancellation isn’t available through the current system.
```

No rule allows `cancel_order`, so it was never offered to the model or to task
analysis. Analysis flagged cancelling as an action no available tool can
perform, and the answer says so instead of pretending. To let a reviewer
approve cancellations instead, see
[Policies and approvals](../concepts/policies.md).

## 6. Keep going until the objective is met

By default every request runs once. A [`LoopPolicy`][agloom.LoopPolicy] with
more iterations adds an outer loop: a model evaluator judges each result
against the objective, feeds concrete feedback into the next iteration, and
stops when the objective is met or a budget is exhausted.

```python
from agloom import LoopPolicy

looping = create_agent(
    model=model,
    tools=[order_status],
    pattern="react",
    loop_policy=LoopPolicy(max_iterations=3),
)
result = looping.invoke_loop(
    "Report the status of orders A-1001 and A-1002 as a two-row Markdown table."
)
print(result.result.content)
print(result.state.iteration, result.state.termination_reason)
```

```text
| Order | Status |
|---|---|
| A-1001 | Delivered on 2 October |
| A-1002 | In transit, arriving 8 October |
1 objective_satisfied
```

The evaluator judged the first result complete, so the loop stopped after one
iteration with an explicit termination reason.

`invoke_loop` returns a [`LoopResult`][agloom.LoopResult] with the answer and
the typed loop state, including every iteration's observation, evaluation, and
decision. Learn more in [Loop Engineering](../concepts/loop-engineering.md).

## 7. Async and streaming

Every API has an async twin, and `stream` exposes LangGraph's stream modes:

```python
import asyncio

from langchain_core.messages import AIMessageChunk


async def main() -> None:
    answer = await agent.ainvoke("Where is order A-1001?")
    print(answer.content)


asyncio.run(main())

for token, metadata in agent.stream("Where is order A-1001?", stream_mode="messages"):
    if isinstance(token, AIMessageChunk):
        print(token.text, end="", flush=True)
```

`stream_mode="messages"` yields `(token, metadata)` pairs from every model call
of the selected execution, including nested topology graphs. Agloom's internal
decisions, such as task analysis, are never streamed. See
[Streaming](../guides/streaming.md) for progress updates and Agloom events.

## Next steps

- Understand the [architecture](../concepts/architecture.md) and the
  [eight topologies](../concepts/topologies.md).
- Add approvals and exactly-once side effects with
  [Writing policies](../guides/writing-policies.md).
- Persist and resume runs: [Human-in-the-loop and resume](../guides/human-in-the-loop.md).
- Explore the [examples](../examples/overview.md).
