# Testing agents

Agloom agents are deterministic around the model: selection, routing,
policies, limits, and loop decisions are all code. Test them with a scripted
chat model, assert on events and tool calls, and use property-based tests for
policies and invariants. Keep a few live-model tests for end-to-end behavior.

## A scripted chat model

Agloom needs tool calling, and its internal decisions use structured output,
which LangChain implements as a tool call named after the schema. A small
scripted model that supports `bind_tools` therefore covers every path:

```python
import itertools
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

_ids = itertools.count(1)


class ScriptedModel(BaseChatModel):
    """Returns scripted responses in order and records every prompt."""

    responses: list[AIMessage]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)


def call(name: str, **args: Any) -> AIMessage:
    """A response that calls one tool or structured-output schema."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])
```

Script structured decisions by their schema names: `TaskAnalysis` (execution
selection), `SupervisorRoute`, `SwarmTurn`, `Plan`, `Review`, and
`LoopAssessment`. For topologies that run branches in parallel (Planner and
Blackboard), answer from the prompt content instead of a fixed order.

## Unit tests

```python
import pytest
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langchain_core.tools import tool

from agloom import EventName, PolicyManager, PolicyViolationError, create_agent

refunds: list[str] = []


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund an order."""
    refunds.append(order_id)
    return f"refunded {amount} for {order_id}"


def test_tool_runs_exactly_once():
    refunds.clear()
    model = ScriptedModel(
        responses=[
            call("refund_order", order_id="A-1", amount=20),
            AIMessage(content="Refunded order A-1."),
        ]
    )
    agent = create_agent(model=model, tools=[refund_order], pattern="react")

    assert agent.invoke("Refund 20 for order A-1.").content == "Refunded order A-1."
    assert refunds == ["A-1"]


def test_simple_requests_run_direct():
    events = []
    model = ScriptedModel(
        responses=[
            call("TaskAnalysis", complexity=0, estimated_steps=1),
            AIMessage(content="Hello!"),
        ]
    )
    agent = create_agent(model=model, tools=[refund_order], observers=[events.append])

    assert agent.invoke("Hi").content == "Hello!"
    selected = next(event for event in events if event.name == EventName.TOPOLOGY_SELECTED)
    assert selected.payload["mode"] == "direct"


def test_policy_blocks_large_refunds(tmp_path):
    refunds.clear()
    policy = PolicyManager(
        policy=RuntimePolicy(
            version="test",
            rules=(
                PolicyRule(
                    "deny large refunds",
                    PolicyEffect.DENY,
                    tool="refund_order",
                    conditions=(ArgumentCondition("amount", ConditionOperator.GT, 1000),),
                ),
                PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="refund_order"),
            ),
        ),
        store=SQLiteRuntimeStore(str(tmp_path / "agentbarrier.db")),
    )
    model = ScriptedModel(responses=[call("refund_order", order_id="A-1", amount=5000)])
    agent = create_agent(model=model, tools=[refund_order], pattern="react", policy=policy)

    with pytest.raises(PolicyViolationError):
        agent.invoke("Refund 5000 for order A-1.")
    assert refunds == []
```

Useful assertions:

- **Tool calls**: record calls inside your tools, or inspect the messages of
  `agent.invoke({"messages": [...]})`.
- **Decisions**: collect events with `observers=[events.append]` and assert on
  `TopologySelected`, `ToolCallDenied`, or `LoopIterationCompleted` payloads.
- **Prompts**: `model.prompts` shows exactly what the model received, for
  example that loop feedback arrives as data in the user turn.
- **Loops**: `agent.invoke_loop(...)` returns the typed state; assert on
  `termination_reason`, `iteration`, and `history`.

## Property-based tests

Policies and execution selection are pure functions, which makes them ideal for
[Hypothesis](https://hypothesis.readthedocs.io/):

```python
from hypothesis import given
from hypothesis import strategies as st

from agloom import StrategyEngine, TaskAnalysis, TopologyName

support_policy = RuntimePolicy(
    version="support",
    rules=(
        PolicyRule(
            "review refunds over 100",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="refund_order",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
        ),
        PolicyRule("allow refunds", PolicyEffect.ALLOW, tool="refund_order"),
    ),
)


@given(st.floats(min_value=100.01, max_value=1e9, allow_nan=False))
def test_large_refunds_always_need_approval(amount):
    decision = support_policy.evaluate("refund_order", {"amount": amount})
    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL


@given(st.text(min_size=1).filter(lambda name: name.strip() and name != "refund_order"))
def test_every_other_tool_is_denied(tool_name):
    assert support_policy.evaluate(tool_name, {}).effect is PolicyEffect.DENY


@given(st.integers(0, 10), st.integers(1, 128))
def test_tool_only_tasks_use_react(complexity, steps):
    analysis = TaskAnalysis(complexity=complexity, estimated_steps=steps, requires_tools=True)
    assert StrategyEngine().decide(analysis).topology is TopologyName.REACT
```

Run more examples in CI than locally with Hypothesis settings profiles, for
example a `dev` and a `ci` profile selected by an environment variable.

## Integration tests

- Use `InMemorySaver` with a permissive `checkpoint_authorizer` to test pauses
  and resumes, and an agentbarrier `SQLiteRuntimeStore` in a temporary
  directory to test approvals: `store.decide(...)`, then `agent.resume(config)`.
- Set `LANGGRAPH_STRICT_MSGPACK=true` so that checkpoint serialization issues
  fail in tests, not in production.
- Register in-process `FastMCP` servers on an `MCPRuntime`, with
  `refresh=RefreshPolicy(on_register=True)`, to test MCP tools without a
  network.

## Live-model tests

Keep a small suite of end-to-end tests against a real model, gated by a pytest
marker and credentials from the environment, and assert on facts rather than
exact wording. The repository's [examples](../examples/overview.md) are
written this way.
