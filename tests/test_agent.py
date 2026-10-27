"""The Agent runnable and `create_agent`."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from agentbarrier import Decision
from agentbarrier.runtime import (
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from agloom import (
    AgentEvent,
    CancellationError,
    CapabilityResolutionError,
    CompilationError,
    ConfigurationError,
    DefaultLoopController,
    DefaultLoopEvaluator,
    EventName,
    ExecutionError,
    ExecutionLimits,
    ExecutionPaused,
    LoopAction,
    LoopDecision,
    LoopEvaluation,
    LoopEvaluationError,
    LoopPolicy,
    LoopStateError,
    Observer,
    PipelineStage,
    PolicyManager,
    RecursionLimitError,
    TopologyName,
    TopologySelectionError,
    WorkerSpec,
    __version__,
    create_agent,
)
from helpers import (
    Call,
    EventLog,
    ScriptedChatModel,
    allow_threads,
    analysis,
    tool_call,
)


def direct(*answers: Any) -> Callable[[Call], Any]:
    queue = list(answers)

    def respond(call: Call) -> Any:
        if call.wants("TaskAnalysis"):
            return analysis()
        return queue.pop(0)

    return respond


def echo(call: Call) -> Any:
    if call.wants("TaskAnalysis"):
        return analysis()
    return f"echo: {call.last.text}"


@tool
def lookup(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id} shipped"


class TestCreateAgent:
    def test_version_is_exposed(self) -> None:
        assert __version__

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"model": "gpt"}, "BaseChatModel"),
            ({"max_reflections": 0}, "between 1 and 5"),
            ({"pattern": "hybrid"}, "explicit composition"),
            (
                {"composition": ["react", "planner"], "pattern": "react"},
                "Hybrid can be",
            ),
            (
                {"workers": [WorkerSpec(name="a", description="x")] * 2},
                "worker names must be unique",
            ),
            (
                {"stages": [PipelineStage(name="a", instruction="x")] * 2},
                "stage names must be unique",
            ),
            ({"tools": [lookup, lookup]}, "tool names must be unique"),
            (
                {"workers": [WorkerSpec(name="STOP", description="x")]},
                "reserved worker names",
            ),
            ({"interrupt_before": ["tools"]}, "require a checkpointer"),
            ({"checkpointer": InMemorySaver()}, "checkpoint_authorizer"),
            ({"loop_policy": LoopPolicy(allow_refresh=True)}, "refresh engine"),
            (
                {
                    "pattern": "react",
                    "loop_policy": LoopPolicy(allow_topology_change=True),
                },
                "allow_topology_change",
            ),
            (
                {
                    "interrupt_before": ["tools"],
                    "checkpointer": InMemorySaver(),
                    "checkpoint_authorizer": allow_threads,
                },
                "one explicit pattern",
            ),
            ({"pattern": ["react", "react"]}, "unique"),
            ({"observers": [42]}, "Observer instances or callables"),
        ],
    )
    def test_rejects_invalid_configuration(
        self,
        model: Callable[..., ScriptedChatModel],
        options: dict[str, Any],
        message: str,
    ) -> None:
        options.setdefault("model", model())
        with pytest.raises((ConfigurationError, TypeError), match=message):
            create_agent(**options)

    def test_invalid_explicit_specs_fail_fast(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        with pytest.raises(ConfigurationError, match="requires workers"):
            create_agent(model=model(), pattern="supervisor")
        with pytest.raises(CompilationError) as error:
            create_agent(
                model=model(),
                pattern="react",
                checkpointer=InMemorySaver(),
                checkpoint_authorizer=allow_threads,
                interrupt_before=["plan"],
            )
        assert "plan" in str(error.value.__cause__)

    def test_automatic_selection_requires_structured_output(self) -> None:
        with pytest.raises(ConfigurationError, match="structured output"):
            create_agent(model=FakeListChatModel(responses=["x"]))
        agent = create_agent(
            model=FakeListChatModel(responses=["x"]), pattern="pipeline"
        )
        assert agent.invoke("go").text == "x"

    def test_properties(self, model: Callable[..., ScriptedChatModel]) -> None:
        limits = ExecutionLimits(max_depth=1)
        agent = create_agent(model=model(), tools=[lookup], limits=limits)
        assert agent.id
        assert agent.capabilities.names == ("lookup",)
        assert agent.policy is None
        assert agent.loop_policy == LoopPolicy()
        assert agent.limits is limits
        assert agent.xai is None
        assert agent.feedback is None


class TestInputs:
    def test_accepts_text_messages_and_mappings(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(model=model(echo))
        assert agent.invoke("  hi  ").text == "echo: hi"
        assert agent.invoke(HumanMessage(content="msg")).text == "echo: msg"
        assert agent.invoke(
            [("user", "a"), ("assistant", "b"), ("user", "c")]
        ).text == ("echo: c")
        assert agent.invoke([{"role": "user", "content": "d"}]).text == "echo: d"
        shaped = agent.invoke({"messages": [HumanMessage(content="e")], "extra": 1})
        assert shaped["extra"] == 1
        assert shaped["messages"][-1].text == "echo: e"
        for key in ("task", "input", "prompt"):
            result = agent.invoke({key: f"via {key}"})
            assert result["messages"][-1].text == f"echo: via {key}"

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("   ", "must not be empty"),
            (42, "unsupported input type"),
            ({"other": 1}, "messages, task, input, or prompt"),
            (
                [SystemMessage(content="x"), HumanMessage(content="y")],
                "user and assistant",
            ),
            ([AIMessage(content="only ai")], "at least one user message"),
            ([object()], "role/content mappings"),
        ],
    )
    def test_rejects_invalid_inputs(
        self, model: Callable[..., ScriptedChatModel], value: Any, message: str
    ) -> None:
        with pytest.raises(ConfigurationError, match=message):
            create_agent(model=model(echo)).invoke(value)

    def test_rejects_unknown_options(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(model=model(echo))
        with pytest.raises(TypeError, match="unexpected invocation options"):
            agent.invoke("hi", unknown=True)


class TestExecution:
    def test_direct_with_system_prompt_and_events(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        scripted = model(direct("Hello."))
        agent = create_agent(
            model=scripted, system_prompt="  Be brief.  ", observers=[events]
        )
        assert agent.invoke("Hi").text == "Hello."
        assert scripted.calls[-1].system == "Be brief."
        assert scripted.calls[-1].tools == ()
        assert events.names[0] == EventName.AGENT_CREATED
        for name in (
            EventName.EXECUTION_STARTED,
            EventName.CAPABILITIES_ROUTED,
            EventName.TOPOLOGY_SELECTION_STARTED,
            EventName.TOPOLOGY_SELECTED,
            EventName.DIRECT_EXECUTION_STARTED,
            EventName.EXECUTION_COMPLETED,
        ):
            assert name in events.names

    def test_explicit_pattern_reports_its_topology(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        agent = create_agent(
            model=model(None, "ok"), pattern="react", observers=[events]
        )
        agent.invoke("Hi")
        selected = [e for e in events if e.name == EventName.TOPOLOGY_SELECTED]
        assert selected[0].topology == "react"
        assert selected[0].payload["mode"] == "topology"

    def test_long_tool_loops_use_langgraphs_recursion_limit(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        def respond(call: Call) -> Any:
            calls = sum(message.type == "tool" for message in call.messages)
            if calls < 30:
                return tool_call("lookup", {"order_id": str(calls)})
            return f"looked up {calls} orders"

        agent = create_agent(model=model(respond), tools=[lookup], pattern="react")
        assert agent.invoke("Check 30 orders").text == "looked up 30 orders"
        with pytest.raises(ExecutionError, match="GraphRecursionError"):
            agent.invoke("Check 30 orders", {"recursion_limit": 8})

    def test_answers_never_carry_message_names(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            lambda call: (
                "done"
                if call.last.type == "tool"
                else tool_call("lookup", {"order_id": "1"})
            )
        )
        agent = create_agent(
            model=scripted,
            tools=[lookup],
            pattern="react",
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "names"}}
        assert agent.invoke("Where is 1?", config).name is None
        agent.invoke("And again?", config)
        for call in scripted.calls:
            assert all(m.name is None for m in call.messages if m.type == "ai")

    def test_conversation_memory_per_thread(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(direct("first answer", "second answer"))
        agent = create_agent(
            model=scripted,
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "memory"}}
        agent.invoke("first question", config)
        agent.invoke("second question", config)
        history = [m.text for m in scripted.calls[-1].messages]
        assert history[-3:] == ["first question", "first answer", "second question"]

    def test_checkpoint_authorization(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=model(echo),
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=lambda thread: thread == "allowed",
        )
        with pytest.raises(ConfigurationError, match="thread_id is required"):
            agent.invoke("hi")
        with pytest.raises(ConfigurationError, match="not authorized"):
            agent.invoke("hi", {"configurable": {"thread_id": "other"}})
        with pytest.raises(ConfigurationError, match="nothing to resume"):
            agent.resume({"configurable": {"thread_id": "allowed"}})
        no_saver = create_agent(model=model(echo))
        with pytest.raises(
            ConfigurationError, match="requires a configured checkpointer"
        ):
            no_saver.get_state({"configurable": {"thread_id": "x"}})
        with pytest.raises(
            ConfigurationError, match="requires a configured checkpointer"
        ):
            no_saver.invoke(None)

    def test_model_failures_are_wrapped(self) -> None:
        class Broken(ScriptedChatModel):
            def _generate(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("provider down")

        agent = create_agent(model=Broken(), pattern="pipeline")
        with pytest.raises(ExecutionError, match="RuntimeError") as error:
            agent.invoke("hi")
        assert isinstance(error.value.__cause__, RuntimeError)

    def test_observer_failures_are_isolated(
        self, model: Callable[..., ScriptedChatModel], caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(event: AgentEvent) -> None:
            raise RuntimeError("observer down")

        agent = create_agent(model=model(direct("ok")), observers=[broken])
        assert agent.invoke("hi").text == "ok"
        assert "failed to record" in caplog.text

    async def test_async_and_stop(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        @tool
        def slow() -> str:
            """Wait for release."""
            started.set()
            release.wait(5)
            return "late"

        scripted = model(
            lambda call: "done" if call.last.type == "tool" else tool_call("slow", {})
        )
        events = EventLog()
        agent = create_agent(
            model=scripted, tools=[slow], pattern="react", observers=[events]
        )
        task = asyncio.create_task(agent.ainvoke("go"))
        await asyncio.to_thread(started.wait, 5)
        agent.stop()
        release.set()
        with pytest.raises(CancellationError):
            await task
        assert EventName.EXECUTION_CANCELLED in events.names
        assert (
            await create_agent(model=model(direct("async ok"))).ainvoke("hi")
        ).text == ("async ok")


class TestStreaming:
    def scripted(self, model: Callable[..., ScriptedChatModel]) -> ScriptedChatModel:
        return model(
            lambda call: (
                "Order 1 shipped."
                if call.last.type == "tool"
                else tool_call("lookup", {"order_id": "1"})
            )
        )

    def test_message_tokens_without_subgraphs(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=self.scripted(model), tools=[lookup], pattern="react"
        )
        parts = list(agent.stream("Where is 1?", stream_mode="messages"))
        assert parts
        texts = [chunk.text for chunk, _ in parts]
        assert "Order 1 shipped." in "".join(texts)

    def test_custom_events_updates_and_namespaces(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(
            model=self.scripted(model), tools=[lookup], pattern="react"
        )
        custom = list(agent.stream("Where is 1?", stream_mode="custom"))
        assert any(part.get("agloom_event") == "LoopCompleted" for part in custom)
        updates = list(agent.stream("Where is 1?"))
        assert all(isinstance(part, dict) for part in updates)
        multi = list(agent.stream("Where is 1?", stream_mode=["updates", "messages"]))
        assert {mode for mode, _ in multi} == {"updates", "messages"}
        nested = list(
            agent.stream("Where is 1?", stream_mode="updates", subgraphs=True)
        )
        assert any(namespace for namespace, _ in nested)

    async def test_async_streaming_and_paused_streams(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        agent = create_agent(
            model=self.scripted(model),
            tools=[lookup],
            pattern="react",
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=allow_threads,
            interrupt_before=["tools"],
            observers=[events],
        )
        config = {"configurable": {"thread_id": "stream"}}
        parts = [
            part
            async for part in agent.astream("Where is 1?", config, stream_mode="custom")
        ]
        assert parts
        assert events.names[-1] == EventName.EXECUTION_PAUSED
        sync_parts = list(agent.stream(None, config, stream_mode="custom"))
        assert sync_parts
        assert events.names[-1] == EventName.EXECUTION_COMPLETED

    def test_composition_in_chains(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(model=model(echo))
        chain = RunnableLambda(lambda text: f"{text}!") | agent
        assert chain.invoke("hi").text == "echo: hi!"
        streamed = list(chain.stream("hey"))
        assert [message.text for message in streamed] == ["echo: hey!"]

    async def test_async_composition(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(model=model(echo))
        chain = RunnableLambda(lambda text: f"{text}?") | agent
        assert (await chain.ainvoke("ok")).text == "echo: ok?"
        streamed = [message async for message in chain.astream("go")]
        assert [message.text for message in streamed] == ["echo: go?"]


class Ticket(BaseModel):
    title: str


class TestStructuredOutput:
    def responder(self, call: Call) -> Any:
        if call.wants("TaskAnalysis"):
            return analysis()
        assert "<xstructured>" in call.last.text
        return '<xstructured>{"title": "Login fails"}</xstructured>'

    def test_all_input_shapes(self, model: Callable[..., ScriptedChatModel]) -> None:
        structured = create_agent(model=model(self.responder)).with_structured_output(
            Ticket
        )
        for value in (
            "Users cannot log in",
            [HumanMessage(content="Users cannot log in")],
            {"messages": [("user", "Users cannot log in")]},
            {"task": "Users cannot log in"},
        ):
            assert structured.invoke(value).structured == Ticket(title="Login fails")

    async def test_async_and_manual_instructions(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        agent = create_agent(model=model(self.responder))
        structured = agent.with_structured_output(Ticket)
        assert (await structured.ainvoke("x")).structured.title == "Login fails"
        manual = agent.with_structured_output(Ticket, inject_instructions=False)
        prompt = f"Users cannot log in\n\n{manual.instructions}"
        assert manual.invoke(prompt).structured.title == "Login fails"


class TestCapabilities:
    def test_route_and_execute(self, model: Callable[..., ScriptedChatModel]) -> None:
        class Invoker:
            description = "Invokable service."

            def invoke(self, value: Any) -> str:
                return f"invoked {value}"

        def plain(value: Any) -> str:
            return f"called {value}"

        async def coroutine(value: Any) -> str:
            return "never"

        agent = create_agent(
            model=model(),
            tools=[lookup],
            capabilities={
                "invoker": Invoker(),
                "plain": plain,
                "async_only": coroutine,
                "inert": object(),
            },
        )
        assert agent.route_capabilities("x").names == (
            "lookup",
            "invoker",
            "plain",
            "async_only",
            "inert",
        )
        assert (
            agent.execute_capability("lookup", {"order_id": "9"}) == "order 9 shipped"
        )
        assert agent.execute_capability("invoker", 1) == "invoked 1"
        assert agent.execute_capability("plain", 2) == "called 2"
        with pytest.raises(CapabilityResolutionError, match="asynchronous"):
            agent.execute_capability("async_only")
        with pytest.raises(CapabilityResolutionError, match="not executable"):
            agent.execute_capability("inert")
        with pytest.raises(CapabilityResolutionError, match="not registered"):
            agent.execute_capability("missing")

    async def test_async_execution(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class AsyncService:
            async def ainvoke(self, value: Any) -> str:
                return f"async {value}"

        class SyncService:
            def invoke(self, value: Any) -> str:
                return f"sync {value}"

        async def coroutine(value: Any) -> str:
            return f"coroutine {value}"

        agent = create_agent(
            model=model(),
            tools=[lookup],
            capabilities={
                "async_service": AsyncService(),
                "sync_service": SyncService(),
                "coroutine": coroutine,
                "inert": object(),
            },
        )
        assert (await agent.aroute_capabilities("x")).names[0] == "lookup"
        assert await agent.aexecute_capability("lookup", {"order_id": "3"}) == (
            "order 3 shipped"
        )
        assert await agent.aexecute_capability("async_service", 1) == "async 1"
        assert await agent.aexecute_capability("sync_service", 2) == "sync 2"
        assert await agent.aexecute_capability("coroutine", 3) == "coroutine 3"
        with pytest.raises(CapabilityResolutionError, match="not executable"):
            await agent.aexecute_capability("inert")
        with pytest.raises(CapabilityResolutionError, match="not registered"):
            await agent.aexecute_capability("missing")


class TestMiddlewareIntegration:
    def test_middleware_tools_are_routed_and_direct_offers_none(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        offered: list[tuple[str, ...]] = []

        def respond(call: Call) -> Any:
            if call.wants("TaskAnalysis"):
                return analysis()
            offered.append(call.tools)
            if call.last.type == "tool":
                return f"after: {call.last.text}"
            return tool_call("write_todos", {"todos": []})

        react = create_agent(
            model=model(respond), pattern="react", middleware=[TodoListMiddleware()]
        )
        assert react.capabilities.names == ("write_todos",)
        react.invoke("Plan")
        assert offered[0] == ("write_todos",)
        offered.clear()
        direct_agent = create_agent(
            model=model(respond), middleware=[TodoListMiddleware()], observers=[events]
        )
        assert "not available for this request" in direct_agent.invoke("Hi").text
        assert offered[0] == ()
        denied = [e for e in events if e.name == EventName.TOOL_CALL_DENIED]
        assert denied[0].payload == {"tool": "write_todos", "reason": "ToolNotOffered"}

    def test_final_answers_pass_through_application_middleware(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Redact(AgentMiddleware):
            def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
                last = state["messages"][-1]
                if "secret" in last.text:
                    return {
                        "messages": [
                            AIMessage(
                                content=last.text.replace("secret", "[redacted]"),
                                id=last.id,
                            )
                        ]
                    }
                return None

        plan = {"steps": [{"id": "a", "task": "Find"}]}

        def respond(call: Call) -> Any:
            if call.wants("Plan"):
                return tool_call("Plan", plan)
            if "Combine the step results" in call.system:
                return "the secret answer"
            return "step"

        plain = create_agent(model=model(respond), pattern="planner")
        assert plain.invoke("Go").text == "the secret answer"
        screened = create_agent(
            model=model(respond), pattern="planner", middleware=[Redact()]
        )
        assert screened.invoke("Go").text == "the [redacted] answer"

    def test_supervisor_answer_shortcut_respects_middleware(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        def respond(call: Call) -> Any:
            if call.wants("SupervisorRoute"):
                return tool_call(
                    "SupervisorRoute", {"next": "FINISH", "final_answer": "shortcut"}
                )
            return "synthesized"

        workers = [WorkerSpec(name="w", description="d")]

        class Passthrough(AgentMiddleware):
            pass

        quick = create_agent(
            model=model(respond), pattern="supervisor", workers=workers
        )
        assert quick.invoke("Go").text == "shortcut"
        screened = create_agent(
            model=model(respond),
            pattern="supervisor",
            workers=workers,
            middleware=[Passthrough()],
        )
        assert screened.invoke("Go").text == "synthesized"


class TestRecursion:
    def responder(self, plan: dict[str, Any]) -> Callable[[Call], Any]:
        def respond(call: Call) -> Any:
            if call.wants("TaskAnalysis"):
                return analysis()
            if call.wants("Plan"):
                return tool_call("Plan", plan)
            if "Combine the step results" in call.system:
                return "combined"
            return f"child: {call.last.text.splitlines()[0]}"

        return respond

    def test_planner_steps_run_as_children(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        plan = {"steps": [{"id": "a", "task": "A"}, {"id": "b", "task": "B"}]}
        scripted = model(self.responder(plan))
        agent = create_agent(
            model=scripted,
            pattern="planner",
            limits=ExecutionLimits(
                max_depth=1, allowed_nested_topologies=(TopologyName.REACT,)
            ),
        )
        assert agent.invoke("Do A and B").text == "combined"
        assert sum(call.wants("TaskAnalysis") for call in scripted.calls) == 2

    def test_child_limits(self, model: Callable[..., ScriptedChatModel]) -> None:
        plan = {"steps": [{"id": "a", "task": "A"}, {"id": "b", "task": "B"}]}
        agent = create_agent(
            model=model(self.responder(plan)),
            pattern="planner",
            limits=ExecutionLimits(max_depth=1, max_children=1, max_workers=1),
        )
        with pytest.raises(RecursionLimitError, match="child limit"):
            agent.invoke("Do A and B")

    def test_child_budget_spans_loop_iterations(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        events = EventLog()
        agent = create_agent(
            model=model(self.responder({"steps": [{"id": "a", "task": "A"}]})),
            pattern="planner",
            limits=ExecutionLimits(max_depth=1, max_children=1),
            loop_policy=LoopPolicy(max_iterations=2),
            loop_evaluator=DefaultLoopEvaluator(),
            observers=[events],
        )
        with pytest.raises(RecursionLimitError, match="child limit"):
            agent.invoke("Do A")
        children = [
            event
            for event in events
            if event.name == EventName.TOPOLOGY_SELECTION_STARTED
            and event.payload["depth"] == 1
        ]
        assert len(children) == 1

    async def test_async_children(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        plan = {"steps": [{"id": "a", "task": "A"}]}
        agent = create_agent(
            model=model(self.responder(plan)),
            pattern="planner",
            limits=ExecutionLimits(max_depth=2),
        )
        assert (await agent.ainvoke("Do A")).text == "combined"

    def test_concurrency_settings_never_block_children(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        plan = {"steps": [{"id": "a", "task": "A"}, {"id": "b", "task": "B"}]}
        agent = create_agent(
            model=model(self.responder(plan)),
            pattern="planner",
            limits=ExecutionLimits(
                max_depth=1,
                max_workers=1,
                allowed_nested_topologies=(TopologyName.REACT,),
            ),
        )
        answers = agent.batch(["Do A and B", "Do B and A"], {"max_concurrency": 1})
        assert [answer.text for answer in answers] == ["combined", "combined"]


class TestPinnedPlans:
    """A resumed run continues the plan it paused in, even if analysis would differ."""

    @staticmethod
    def governed_export(
        store: SQLiteRuntimeStore, exported: list[str]
    ) -> tuple[Any, PolicyManager]:
        @tool
        def export(target: str) -> str:
            """Export data to a target."""
            exported.append(target)
            return f"exported to {target}"

        policy = PolicyManager(
            policy=RuntimePolicy(
                version="exports-v1",
                rules=(
                    PolicyRule(
                        "review exports", PolicyEffect.REQUIRE_APPROVAL, tool="export"
                    ),
                ),
            ),
            store=store,
            consequential={"export": "target"},
        )
        return export, policy

    @staticmethod
    def responder() -> Callable[[Call], Any]:
        """Analysis picks tools once; planning again would answer without them."""
        analyses = iter([analysis(requires_tools=True), analysis()])
        plan = {"steps": [{"id": "a", "task": "Export the data to s3://bucket"}]}

        def respond(call: Call) -> Any:
            if call.wants("Plan"):
                return tool_call("Plan", plan)
            if call.wants("TaskAnalysis"):
                return next(analyses)
            if "Combine the step results" in call.system:
                return "combined"
            if call.last.type == "tool":
                return f"final: {call.last.text}"
            if call.wants("export"):
                return tool_call("export", {"target": "s3://bucket"})
            return "answered without tools"

        return respond

    @staticmethod
    def approve(store: SQLiteRuntimeStore, approval: dict[str, Any]) -> None:
        store.decide(approval["action_id"], Decision.APPROVE, decided_by="alice")

    def test_resumed_runs_keep_their_topology(
        self,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        barrier_store: SQLiteRuntimeStore,
    ) -> None:
        exported: list[str] = []
        export, policy = self.governed_export(barrier_store, exported)
        agent = create_agent(
            model=model(self.responder()),
            tools=[export],
            policy=policy,
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "pinned"}}
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Export the data", config)
        self.approve(barrier_store, paused.value.interrupts[0].value)
        assert agent.resume(config).text == "final: exported to s3://bucket"
        assert exported == ["s3://bucket"]

    def test_resumed_children_keep_their_topology(
        self,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        barrier_store: SQLiteRuntimeStore,
    ) -> None:
        exported: list[str] = []
        export, policy = self.governed_export(barrier_store, exported)
        agent = create_agent(
            model=model(self.responder()),
            tools=[export],
            pattern="planner",
            limits=ExecutionLimits(max_depth=1),
            policy=policy,
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "pinned"}}
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Export the data", config)
        self.approve(barrier_store, paused.value.interrupts[0].value)
        assert agent.resume(config).text == "combined"
        assert exported == ["s3://bucket"]

    async def test_async_resumed_children_keep_their_topology(
        self,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        barrier_store: SQLiteRuntimeStore,
    ) -> None:
        exported: list[str] = []
        export, policy = self.governed_export(barrier_store, exported)
        agent = create_agent(
            model=model(self.responder()),
            tools=[export],
            pattern="planner",
            limits=ExecutionLimits(max_depth=1),
            policy=policy,
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "pinned"}}
        with pytest.raises(ExecutionPaused) as paused:
            await agent.ainvoke("Export the data", config)
        self.approve(barrier_store, paused.value.interrupts[0].value)
        assert (await agent.aresume(config)).text == "combined"
        assert exported == ["s3://bucket"]

    def test_pauses_inside_an_enclosing_graph(
        self,
        model: Callable[..., ScriptedChatModel],
        saver: Any,
        barrier_store: SQLiteRuntimeStore,
        events: EventLog,
    ) -> None:
        exported: list[str] = []
        export, policy = self.governed_export(barrier_store, exported)
        agent = create_agent(
            model=model(self.responder()),
            tools=[export],
            policy=policy,
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
            observers=[events],
        )

        class Outer(TypedDict, total=False):
            task: str
            answer: str

        def delegate(state: Outer, config: RunnableConfig) -> dict[str, str]:
            return {"answer": agent.invoke(state["task"], config).text}

        outer = (
            StateGraph(Outer)
            .add_node("delegate", delegate)
            .add_edge(START, "delegate")
            .add_edge("delegate", END)
            .compile(checkpointer=saver)
        )
        config = {"configurable": {"thread_id": "outer"}}
        paused = outer.invoke({"task": "Export the data"}, config)
        assert events.names[-1] == EventName.EXECUTION_PAUSED
        self.approve(barrier_store, paused["__interrupt__"][0].value)
        resumed = outer.invoke(Command(resume=True), config)
        assert resumed["answer"] == "final: exported to s3://bucket"
        assert exported == ["s3://bucket"]
        assert events.names[-1] == EventName.EXECUTION_COMPLETED


class RecordingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.starts += 1


class CallbackObserver(Observer):
    def __init__(self) -> None:
        self.handler = RecordingHandler()
        self.flushed = 0
        self.closed = 0

    def on_event(self, event: AgentEvent) -> None:
        return

    @property
    def callback_handlers(self) -> tuple[BaseCallbackHandler, ...]:
        return (self.handler,)

    def flush(self) -> None:
        self.flushed += 1

    def shutdown(self) -> None:
        self.closed += 1


class TestCallbacksAndNesting:
    def test_observer_callbacks_merge_with_caller_callbacks(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        observer = CallbackObserver()
        caller = RecordingHandler()
        agent = create_agent(model=model(echo), observers=[observer])
        agent.invoke("a", {"callbacks": [caller]})
        agent.invoke("b", {"callbacks": CallbackManager(handlers=[caller])})
        agent.invoke("c")
        assert observer.handler.starts >= 3
        assert caller.starts >= 2
        agent.flush()
        agent.close()
        assert observer.flushed == 2
        assert observer.closed == 1

    def test_agent_inside_a_parent_graph_keeps_its_own_thread(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        observer = CallbackObserver()
        agent = create_agent(
            model=model(echo),
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=lambda thread: thread == "inner",
            observers=[observer],
        )

        class State(TypedDict):
            question: str
            answer: str

        def ask(state: State) -> dict[str, str]:
            reply = agent.invoke(
                state["question"], {"configurable": {"thread_id": "inner"}}
            )
            return {"answer": reply.text}

        parent = (
            StateGraph(State)
            .add_node("ask", ask)
            .add_edge(START, "ask")
            .add_edge("ask", END)
            .compile(checkpointer=InMemorySaver())
        )
        result = parent.invoke(
            {"question": "hello"}, {"configurable": {"thread_id": "outer"}}
        )
        assert result["answer"] == "echo: hello"
        assert agent.get_state({"configurable": {"thread_id": "inner"}}).values


class TestEdgeCases:
    async def test_async_cancellation_is_reported_as_cancelled(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        def slow(call: Call) -> Any:
            time.sleep(0.3)
            return "late"

        agent = create_agent(model=model(slow), observers=[events])
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(agent.ainvoke("hi"), 0.05)
        assert events.names[-1] == EventName.EXECUTION_CANCELLED

    def test_drained_runs_are_reported_as_paused(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        def drain(call: Call) -> Any:
            raise GraphDrained("shutdown")

        agent = create_agent(model=model(drain), observers=[events])
        with pytest.raises(GraphDrained):
            agent.invoke("hi")
        assert events.names[-1] == EventName.EXECUTION_PAUSED
        assert events[-1].payload == {"interrupts": 0}

    def test_stream_failures_are_reported(
        self, model: Callable[..., ScriptedChatModel], events: EventLog
    ) -> None:
        class Failing:
            def evaluate(self, *args: Any) -> Any:
                raise RuntimeError("judge down")

        agent = create_agent(
            model=model(direct("x", "y")),
            loop_evaluator=Failing(),
            observers=[events],
        )
        with pytest.raises(LoopEvaluationError):
            list(agent.stream("hi"))
        assert events.names[-1] == EventName.EXECUTION_FAILED
        assert list(agent.transform(iter([]))) == []

    async def test_async_stream_failures_and_empty_transform(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Failing:
            def evaluate(self, *args: Any) -> Any:
                raise RuntimeError("judge down")

        agent = create_agent(model=model(direct("x")), loop_evaluator=Failing())
        with pytest.raises(LoopEvaluationError):
            async for _ in agent.astream("hi"):
                pass

        async def nothing() -> Any:
            return
            yield

        assert [item async for item in agent.atransform(nothing())] == []

    def test_custom_controllers_cannot_escape_the_configuration(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Runaway:
            def decide(self, *args: Any) -> LoopDecision:
                return LoopDecision(action=LoopAction.CONTINUE, reason="again")

        runaway = create_agent(
            model=model(direct("a", "b")),
            loop_controller=Runaway(),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        with pytest.raises(LoopStateError, match="beyond the budget"):
            runaway.invoke("go")

        class Switch(DefaultLoopController):
            def decide(self, state: Any, observation: Any, *args: Any) -> LoopDecision:
                if observation.success and state.iteration == 1:
                    return LoopDecision(
                        action=LoopAction.CHANGE_TOPOLOGY,
                        reason="custom",
                        next_topology=TopologyName.SWARM,
                    )
                return super().decide(state, observation, *args)

        fixed = create_agent(
            model=model(None, "a", "b"),
            pattern="pipeline",
            stages=[PipelineStage(name="only", instruction="Do it.")],
            loop_policy=LoopPolicy(max_iterations=2),
            loop_controller=Switch(),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        with pytest.raises(ConfigurationError, match="cannot override"):
            fixed.invoke("go")
        allow_list = create_agent(
            model=model(direct("a", "b")),
            pattern=["react", "planner"],
            loop_policy=LoopPolicy(max_iterations=2),
            loop_controller=Switch(),
            loop_evaluator=DefaultLoopEvaluator(),
        )
        with pytest.raises(
            ConfigurationError, match="outside the developer allow-list"
        ):
            allow_list.invoke("go")

    def test_analysis_outcomes(self, model: Callable[..., ScriptedChatModel]) -> None:
        def respond(call: Call) -> Any:
            if call.wants("TaskAnalysis"):
                return analysis(missing_capabilities=["send_email"])
            return "I cannot send email, but here is a draft."

        scripted = model(respond)
        events = EventLog()
        missing = create_agent(model=scripted, observers=[events])
        assert missing.invoke("Email the team").text.startswith("I cannot send")
        assert "do not claim to perform them" in scripted.calls[-1].prompt
        assert "- send_email" in scripted.calls[-1].prompt
        selected = next(e for e in events if e.name == EventName.TOPOLOGY_SELECTED)
        assert selected.payload["missing_capabilities"] == 1
        assert "send_email" not in str(selected.payload)
        no_workers = create_agent(
            model=model(lambda call: analysis(requires_delegation=True)),
            pattern=["supervisor", "react"],
        )
        with pytest.raises(
            TopologySelectionError, match="supervisor topology cannot run"
        ):
            no_workers.invoke("Coordinate")

    def test_failed_loops_report_the_decision(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Failing:
            def evaluate(self, state: Any, observation: Any, policy: Any) -> Any:
                return LoopEvaluation(
                    reason="the answer is unusable", recommended_action=LoopAction.FAIL
                )

        agent = create_agent(model=model(direct("bad")), loop_evaluator=Failing())
        with pytest.raises(ExecutionError, match="the answer is unusable"):
            agent.invoke("go")

    def test_graph_cache_is_bounded(
        self, model: Callable[..., ScriptedChatModel], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agloom.agent.agent as agent_module

        monkeypatch.setattr(agent_module, "_GRAPH_CACHE_SIZE", 1)
        flags = iter(
            [
                {"requires_tools": True},
                {"requires_ordering": True, "estimated_steps": 3},
            ]
        )

        def respond(call: Call) -> Any:
            if call.wants("TaskAnalysis"):
                return analysis(**next(flags))
            return "done"

        agent = create_agent(model=model(respond), tools=[lookup])
        agent.invoke("first")
        agent.invoke("second")
        assert len(agent._graphs) == 1

    def test_structured_output_validation(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        structured = create_agent(model=model(echo)).with_structured_output(Ticket)
        with pytest.raises(ConfigurationError):
            structured.invoke({"other": 1})
        with pytest.raises(ConfigurationError):
            structured.invoke([object()])

    def test_sync_capabilities_under_policy(
        self, model: Callable[..., ScriptedChatModel], barrier_store: Any
    ) -> None:
        from agentbarrier.runtime import PolicyEffect, PolicyRule, RuntimePolicy

        from agloom import PolicyManager

        calls: list[Any] = []

        def notify(value: Any) -> str:
            calls.append(value)
            return "sent"

        policy = PolicyManager(
            policy=RuntimePolicy(
                version="v",
                rules=(PolicyRule("allow", PolicyEffect.ALLOW, tool="notify"),),
            ),
            store=barrier_store,
            consequential={"notify": "id"},
        )
        agent = create_agent(
            model=model(), capabilities={"notify": notify}, policy=policy
        )
        assert agent.execute_capability("notify", {"id": "n-1"}) == "sent"
        assert agent.execute_capability("notify", {"id": "n-1"}) == "sent"
        assert calls == [{"id": "n-1"}]
        assert agent.policy is policy

    def test_parallel_approvals_resume_together(
        self, model: Callable[..., ScriptedChatModel], barrier_store: Any
    ) -> None:
        from agentbarrier import Decision
        from agentbarrier.runtime import PolicyEffect, PolicyRule, RuntimePolicy

        from agloom import ExecutionPaused, PolicyManager

        refunded: list[str] = []

        @tool
        def refund(request_id: str) -> str:
            """Refund a request."""
            refunded.append(request_id)
            return f"refunded {request_id}"

        plan = {
            "steps": [
                {"id": "a", "task": "Refund r-a"},
                {"id": "b", "task": "Refund r-b"},
            ]
        }

        def respond(call: Call) -> Any:
            if call.wants("Plan"):
                return tool_call("Plan", plan)
            if "Combine the step results" in call.system:
                return "both refunded"
            if call.last.type == "tool":
                return call.last.text
            target = "r-a" if "Refund r-a" in call.prompt else "r-b"
            return tool_call("refund", {"request_id": target})

        policy = PolicyManager(
            policy=RuntimePolicy(
                version="v",
                rules=(
                    PolicyRule("review", PolicyEffect.REQUIRE_APPROVAL, tool="refund"),
                ),
            ),
            store=barrier_store,
            consequential={"refund": "request_id"},
        )
        agent = create_agent(
            model=model(respond),
            tools=[refund],
            pattern="planner",
            policy=policy,
            checkpointer=InMemorySaver(),
            checkpoint_authorizer=allow_threads,
        )
        config = {"configurable": {"thread_id": "parallel"}}
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Refund both", config)
        assert len(paused.value.interrupts) == 2
        for pending in paused.value.interrupts:
            barrier_store.decide(
                pending.value["action_id"], Decision.APPROVE, decided_by="ops"
            )
        assert agent.resume(config).text == "both refunded"
        assert sorted(refunded) == ["r-a", "r-b"]


async def test_async_updates_stream_reports_the_top_level(
    model: Callable[..., ScriptedChatModel],
) -> None:
    scripted = model(
        lambda call: (
            "Order 1 shipped."
            if call.last.type == "tool"
            else tool_call("lookup", {"order_id": "1"})
        )
    )
    agent = create_agent(model=scripted, tools=[lookup], pattern="react")
    updates = [part async for part in agent.astream("Where is 1?")]
    assert updates
    assert all(isinstance(part, dict) for part in updates)


def test_nested_agents_keep_their_own_scope(
    model: Callable[..., ScriptedChatModel],
) -> None:
    @tool
    def inner_tool(question: str) -> str:
        """Answer from the inner agent's data."""
        return f"inner:{question}"

    inner_model = model(
        lambda call: (
            f"inner says {call.last.text}"
            if call.last.type == "tool"
            else tool_call("inner_tool", {"question": "x"})
        )
    )
    inner = create_agent(model=inner_model, tools=[inner_tool], pattern="react")

    @tool
    def ask_inner(question: str) -> str:
        """Delegate a question to the inner agent."""
        return inner.invoke(question).text

    outer_model = model(
        lambda call: (
            f"outer: {call.last.text}"
            if call.last.type == "tool"
            else tool_call("ask_inner", {"question": "hi"})
        )
    )
    outer = create_agent(model=outer_model, tools=[ask_inner], pattern="react")
    assert outer.invoke("Go").text == "outer: inner says inner:x"
    assert outer_model.calls[0].tools == ("ask_inner",)
    assert inner_model.calls[0].tools == ("inner_tool",)
