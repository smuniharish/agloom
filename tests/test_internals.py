"""Internal helpers: run context, inputs, recursion budgets, and routing middleware."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agloom import (
    CancellationError,
    ConfigurationError,
    ExecutionLimits,
    LoopBudgetExceededError,
    RecursionLimitError,
    create_agent,
)
from agloom._runtime import (
    RunContext,
    ToolBudget,
    configured_run,
    current_run,
    scoped,
    tools_withheld,
    with_run,
    write_stream,
)
from agloom.agent._inputs import answer_message, shape_result
from agloom.agent._recursion import RecursionSession
from agloom.agent.agent import _arguments, _call, _instructed, _last_ai
from agloom.capabilities._middleware import _prepare_tool_call, _routed_tools
from agloom.loop.runner import _human_feedback
from helpers import ScriptedChatModel, tool_call


def run_context(**changes: Any) -> RunContext:
    return RunContext(
        agent_id="a",
        run_id="r",
        cancel_event=threading.Event(),
        emit=lambda name, payload: None,
        **changes,
    )


@tool
def ping() -> str:
    """Ping."""
    return "pong"


class TestRuntime:
    def test_tool_budget(self) -> None:
        budget = ToolBudget(1)
        budget.consume()
        assert budget.used == 1
        with pytest.raises(LoopBudgetExceededError, match="budget of 1"):
            budget.consume()

    def test_run_context(self) -> None:
        run = run_context()
        run.check_cancelled()
        run.cancel_event.set()
        with pytest.raises(CancellationError):
            run.check_cancelled()
        assert run.derive(depth=2).depth == 2
        assert run.tool_names == frozenset()

    def test_outside_langgraph(self) -> None:
        assert current_run() is None
        assert current_run({"configurable": {}}) is None
        outer = run_context(tool_names=frozenset({"t"}))
        config = with_run({"tags": ["x"]}, outer)
        assert current_run(config) is outer
        assert configured_run(config) is outer
        with tools_withheld():
            assert current_run() is None
        with scoped(outer):
            assert current_run() is outer
            assert configured_run({}) is None
            with tools_withheld():
                withheld = current_run()
                assert withheld is not None
                assert withheld.tool_names == frozenset()
            assert current_run() is outer
        assert current_run() is None
        write_stream({"ignored": True})


class TestInputs:
    def test_answer_and_shape(self) -> None:
        message = AIMessage(content="a")
        assert answer_message(message) is message
        assert answer_message(HumanMessage(content="h")).text == "h"
        assert answer_message(None).text == ""
        assert answer_message(3).text == "3"
        shaped = shape_result(AIMessage(content="done"), {"task": "t", "k": 1}, True)
        assert shaped["k"] == 1
        assert [m.text for m in shaped["messages"]] == ["t", "done"]

    def test_agent_helpers(self) -> None:
        assert _last_ai({"messages": [HumanMessage(content="x")]}).text == ""
        assert _last_ai("plain").text == ""
        assert _arguments(None) == {}
        assert _arguments({"a": 1}) == {"a": 1}
        assert _arguments(5) == {"input": 5}
        assert _instructed({"other": 1}, "rules") == {"other": 1}
        assert _instructed({"input": "q"}, "rules") == {"input": "q\n\nrules"}
        unconvertible = [object()]
        assert _instructed(unconvertible, "rules") is unconvertible
        assert _instructed("q", None) == "q"
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            with pytest.raises(Exception, match="asynchronous"):
                _call(lambda value: future, None)
        finally:
            loop.close()


class TestRecursionSession:
    def test_depth_children_and_deadline(self) -> None:
        session = RecursionSession(ExecutionLimits(max_depth=1, max_children=1))
        with pytest.raises(RecursionLimitError, match="depth limit"):
            session.admit(2)
        session.admit(1)
        with pytest.raises(RecursionLimitError, match="child limit"):
            session.admit(1)
        quick = RecursionSession(ExecutionLimits(max_depth=1, max_execution_time=0.01))
        time.sleep(0.1)
        quick.check()
        quick.admit(1)
        time.sleep(0.1)
        with pytest.raises(RecursionLimitError, match="time limit"):
            quick.check()


class TestRoutingMiddleware:
    def test_agents_outside_agloom_runs_are_unrestricted(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        from agloom import TopologyBuildContext
        from agloom.capabilities._middleware import ToolRoutingMiddleware

        scripted = model(
            lambda call: "done" if call.last.type == "tool" else tool_call("ping", {})
        )
        context = TopologyBuildContext(
            model=scripted,
            tools=(ping,),
            middleware=(ToolRoutingMiddleware(),),
            workers=(),
            stages=(),
            system_prompt=None,
            limits=ExecutionLimits(),
            max_reflections=1,
        )
        agent = context.agent("Use tools.", name="raw")
        result = agent.invoke({"messages": [HumanMessage(content="go")]})
        assert result["messages"][-1].text == "done"
        assert scripted.calls[0].tools == ("ping",)
        result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="go")]}))
        assert result["messages"][-1].text == "done"

    def test_routing_helpers(self) -> None:
        builtin = {"type": "web_search"}
        request = SimpleNamespace(tools=[builtin, ping])
        assert _routed_tools(request, run_context()) == [builtin]  # type: ignore[arg-type]
        offered = run_context(tool_names=frozenset({"ping"}))
        assert _routed_tools(request, offered) == [builtin, ping]  # type: ignore[arg-type]
        call = SimpleNamespace(
            runtime=SimpleNamespace(config=with_run({}, run_context())),
            tool_call={"name": "external", "id": "1"},
            tool=None,
        )
        assert _prepare_tool_call(call) is call  # type: ignore[arg-type]


async def test_withheld_tools_are_rejected_asynchronously(
    model: Callable[..., ScriptedChatModel],
) -> None:
    from langchain.agents.middleware import TodoListMiddleware

    from helpers import analysis

    def respond(call: Any) -> Any:
        if call.wants("TaskAnalysis"):
            return analysis()
        if call.last.type == "tool":
            return call.last.text
        return tool_call("write_todos", {"todos": []})

    agent = create_agent(model=model(respond), middleware=[TodoListMiddleware()])
    answer = await agent.ainvoke("Hi")
    assert "not available" in answer.text


def test_human_feedback_shapes() -> None:
    assert _human_feedback("Use units.") == ["Use units."]
    assert _human_feedback("  ") == []
    assert _human_feedback({"feedback": "one"}) == ["one"]
    assert _human_feedback({"feedback": " "}) == []
    assert _human_feedback({"feedback": ["a", " ", 2]}) == ["a", "2"]
    assert _human_feedback({"other": 1}) == []
    assert _human_feedback(True) == []


def test_loop_graphs_require_an_agent_run(
    model: Callable[..., ScriptedChatModel],
) -> None:
    agent = create_agent(model=model(None, "x"), pattern="pipeline")
    with pytest.raises(ConfigurationError, match="through an Agloom agent"):
        agent._runner.graph.invoke(
            {"task": "t", "messages": [HumanMessage(content="t")]}
        )
