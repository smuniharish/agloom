"""Ecosystem integrations with their real packages (offline, scripted models)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from behaviorweave import BehaviorEngine, InterventionType, PolicyRule
from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware
from contextsage import IntelligentSummarizationMiddleware
from deepagents import CompiledSubAgent, FilesystemMiddleware, create_deep_agent
from feedback_manager import (
    FeedbackCategory,
    FeedbackManager,
    FeedbackQuery,
    FeedbackSource,
    FeedbackTarget,
    FeedbackTargetType,
)
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_xai import XAIRuntime
from refresh_engine import DiscoveryResult, RefreshEngine, Resource, ResourceSnapshot

from agloom import (
    EventName,
    ExecutionPaused,
    LoopPolicy,
    LoopTerminationReason,
    behavior_signals,
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

checked: list[str] = []


def test_langchain_human_review_of_tool_calls(
    model: Callable[..., ScriptedChatModel],
) -> None:
    restocked: list[tuple[str, int]] = []

    @tool
    def restock(sku: str, units: int) -> str:
        """Restock a product."""
        restocked.append((sku, units))
        return f"restocked {units} of {sku}"

    def respond(call: Call) -> Any:
        if call.last.type == "tool":
            return f"done: {call.last.text.splitlines()[-1]}"
        return tool_call("restock", {"sku": "SKU-778", "units": 40})

    agent = create_agent(
        model=model(respond),
        tools=[restock],
        pattern="react",
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"restock": True})],
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=allow_threads,
    )
    config = {"configurable": {"thread_id": "review"}}
    with pytest.raises(ExecutionPaused) as paused:
        agent.invoke("Restock 40 units of SKU-778.", config)
    request = paused.value.interrupts[0].value["action_requests"][0]
    assert (request["name"], request["args"]) == (
        "restock",
        {"sku": "SKU-778", "units": 40},
    )
    assert restocked == []
    edited = {"name": "restock", "args": {"sku": "SKU-778", "units": 20}}
    answer = agent.resume(
        config, {"decisions": [{"type": "edit", "edited_action": edited}]}
    )
    assert answer.text == "done: restocked 20 of SKU-778"
    assert restocked == [("SKU-778", 20)]


@tool
def check_inventory(sku: str) -> str:
    """Check inventory for a SKU."""
    checked.append(sku)
    return f"{sku}: 4 units"


def test_behaviorweave_stops_repeated_calls_and_the_loop(
    model: Callable[..., ScriptedChatModel],
) -> None:
    checked.clear()
    engine = BehaviorEngine(
        policies=[
            PolicyRule(
                policy_id="stop-repeats",
                pattern_id="repeated_tool_call",
                threshold=2,
                intervention=InterventionType.STOP,
                message="Stop repeating the same call.",
            )
        ]
    )

    def respond(call: Call) -> Any:
        if sum(message.type == "tool" for message in call.messages) >= 2:
            return "Giving up."
        return tool_call("check_inventory", {"sku": "A-1"})

    agent = create_agent(
        model=model(respond),
        tools=[check_inventory],
        pattern="react",
        middleware=[
            BehaviorWeaveMiddleware(engine, scope="tests", on_decision=behavior_signals)
        ],
    )
    result = agent.invoke_loop("Check A-1")
    assert checked == ["A-1"]
    assert (
        result.state.termination_reason is LoopTerminationReason.BEHAVIOR_INTERVENTION
    )
    assert "stop" in result.state.history[0].observation.behavior_signals


def test_contextsage_summarizes_long_conversations(
    model: Callable[..., ScriptedChatModel],
) -> None:
    def respond(call: Call) -> Any:
        if "Context Extraction Assistant" in call.last.text:
            return "Summary: the user asked about orders A-100 and B-200."
        return f"Answered with {len(call.messages)} messages."

    scripted = model(respond)
    conversation = []
    for index in range(6):
        conversation.append(
            HumanMessage(content=f"Question {index} about order A-100.")
        )
        conversation.append(AIMessage(content=f"Answer {index} about order B-200."))
    conversation.append(HumanMessage(content="Summarize the status."))
    agent = create_agent(
        model=scripted,
        pattern="react",
        middleware=[
            IntelligentSummarizationMiddleware(
                scripted, trigger=("messages", 6), keep=("messages", 2)
            )
        ],
    )
    answer = agent.invoke(conversation)
    final_call = scripted.calls[-1]
    assert answer.text == f"Answered with {len(final_call.messages)} messages."
    assert len(final_call.messages) < len(conversation)


def test_explainability_instruments_runs(
    model: Callable[..., ScriptedChatModel],
) -> None:
    runtime = XAIRuntime(application_id="agloom-tests")
    try:

        def respond(call: Call) -> Any:
            return analysis() if call.wants("TaskAnalysis") else "explained"

        agent = create_agent(model=model(respond), explainability=runtime)
        assert agent.xai is runtime
        assert agent.invoke("Explain").text == "explained"
        agent.flush()
        agent.close()
    finally:
        runtime.run_sync(runtime.close())


async def test_feedback_and_refresh_drive_the_loop(
    model: Callable[..., ScriptedChatModel],
) -> None:
    manager = FeedbackManager()
    await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.CORRECTION,
        target=FeedbackTarget(type=FeedbackTargetType.THREAD, id="thread-7"),
        payload={"text": "Cite the source document."},
    )
    refreshed: list[str] = []

    class Source:
        async def discover(self) -> DiscoveryResult:
            return DiscoveryResult([Resource("doc-1", version="2")])

        async def snapshot(self, resource: Resource, /) -> ResourceSnapshot:
            return ResourceSnapshot(resource.resource_id, content="fresh", version="2")

    async def operation(
        resource: Any, snapshot: Any, action: Any, request: Any, /
    ) -> None:
        refreshed.append(resource.resource_id)

    engine = RefreshEngine(Source(), operation)

    def respond(call: Call) -> Any:
        if call.wants("TaskAnalysis"):
            return analysis()
        if call.wants("LoopAssessment"):
            complete = "updated report" in call.prompt
            return tool_call(
                "LoopAssessment",
                {
                    "objective_complete": complete,
                    "progress": 1.0 if complete else 0.3,
                    "quality": 0.9,
                    "feedback": [] if complete else ["Sources are stale."],
                    "next_step": "complete" if complete else "refresh",
                },
            )
        return "updated report" if refreshed else "stale report"

    scripted = model(respond)
    events = EventLog()
    agent = create_agent(
        model=scripted,
        loop_policy=LoopPolicy(max_iterations=3, allow_refresh=True),
        feedback=manager,
        refresh=engine,
        observers=[events],
    )
    try:
        result = await agent.ainvoke_loop(
            "Write the report", {"configurable": {"thread_id": "thread-7"}}
        )
    finally:
        await engine.close()
    assert result.result.text == "updated report"
    assert refreshed == ["doc-1"]
    assert EventName.LOOP_REFRESHED in events.names
    answers = [
        c
        for c in scripted.calls
        if not (c.wants("TaskAnalysis") or c.wants("LoopAssessment"))
    ]
    assert len(answers) == 2
    assert all("Cite the source document." in a.prompt for a in answers)
    recorded = await manager.query(FeedbackQuery(source=FeedbackSource.EVALUATOR))
    assert recorded
    assert recorded[0].payload["feedback"] == ["Sources are stale."]


def test_deep_agents_middleware_inside_agloom(
    model: Callable[..., ScriptedChatModel],
) -> None:
    def respond(call: Call) -> Any:
        tool_results = [m for m in call.messages if m.type == "tool"]
        if not tool_results:
            return tool_call(
                "write_file", {"file_path": "/notes.md", "content": "Revenue grew 12%."}
            )
        if len(tool_results) == 1:
            return tool_call("read_file", {"file_path": "/notes.md"})
        return f"Notes: {tool_results[-1].text}"

    scripted = model(respond)
    agent = create_agent(
        model=scripted, pattern="react", middleware=[FilesystemMiddleware()]
    )
    assert {"write_file", "read_file"} <= set(agent.capabilities.names)
    answer = agent.invoke("Record and read back the revenue note.")
    assert "Revenue grew 12%." in answer.text


def test_agloom_agent_as_a_deep_agents_subagent(
    model: Callable[..., ScriptedChatModel],
) -> None:
    analyst = create_agent(model=model(None, "Q3 revenue grew 12%."), pattern="react")

    def lead(call: Call) -> Any:
        if call.last.type == "tool":
            return f"Lead summary: {call.last.text}"
        return tool_call(
            "task",
            {"description": "Analyze Q3 revenue.", "subagent_type": "analyst"},
        )

    deep = create_deep_agent(
        model=model(lead),
        subagents=[
            CompiledSubAgent(
                name="analyst", description="Analyzes revenue data.", runnable=analyst
            )
        ],
    )
    result = deep.invoke({"messages": [HumanMessage(content="How did Q3 go?")]})
    assert result["messages"][-1].text == "Lead summary: Q3 revenue grew 12%."
