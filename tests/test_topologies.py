"""The eight topologies, the compiler, and shared topology helpers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from agloom import (
    BREAKPOINT_INTERRUPT,
    ArchitectureCompiler,
    ArchitectureSpec,
    CompilationError,
    ConfigurationError,
    ExecutionError,
    ExecutionLimits,
    ExecutionPaused,
    PipelineStage,
    TopologyArtifact,
    TopologyBuildContext,
    TopologyName,
    WorkerSpec,
    create_agent,
)
from agloom.topology._shared import (
    answer_update,
    choices,
    conversation,
    final_message,
    node_name,
    request,
    structured_response,
    topology_input,
    with_context,
)
from agloom.topology.planner import PlanStep, ready_steps, validate_plan
from agloom.topology.supervisor import SupervisorTopology
from agloom.topology.swarm import SwarmTopology
from helpers import Call, ScriptedChatModel, allow_threads, tool_call

Rule = tuple[Callable[[Call], bool], Any]

RESEARCHER = WorkerSpec(name="researcher", description="Finds facts.")
WRITER = WorkerSpec(name="writer", description="Writes prose.")


def script(*rules: Rule) -> Callable[[Call], Any]:
    """Answer each call with the response of the first matching rule."""

    def respond(call: Call) -> Any:
        for matches, response in rules:
            if matches(call):
                return response(call) if callable(response) else response
        raise AssertionError(f"unexpected model call: {call.system[:120]!r}")

    return respond


def wants(schema: str) -> Callable[[Call], bool]:
    return lambda call: call.wants(schema)


def system_has(text: str) -> Callable[[Call], bool]:
    return lambda call: text in call.system


def anything(call: Call) -> bool:
    return True


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class Overlap:
    """Wraps a response and records the most calls running at the same time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def __call__(self, response: Any) -> Callable[[Call], Any]:
        def respond(call: Call) -> Any:
            with self._lock:
                self._active += 1
                self.peak = max(self.peak, self._active)
            try:
                time.sleep(0.2)
                return response(call) if callable(response) else response
            finally:
                with self._lock:
                    self._active -= 1

        return respond


class TestReact:
    def test_tool_loop(self, model: Callable[..., ScriptedChatModel]) -> None:
        scripted = model(
            script(
                (lambda call: call.last.type == "tool", "The sum is 5."),
                (anything, tool_call("add", {"a": 2, "b": 3})),
            )
        )
        agent = create_agent(model=scripted, tools=[add], pattern="react")
        assert agent.invoke("Add 2 and 3").text == "The sum is 5."
        assert scripted.calls[0].tools == ("add",)
        replayed = scripted.calls[1].messages
        assert all(message.name is None for message in replayed if message.type == "ai")

    def test_interrupt_before_tools(
        self, model: Callable[..., ScriptedChatModel], saver: Any
    ) -> None:
        executed: list[int] = []

        @tool
        def record(value: int) -> str:
            """Record a value."""
            executed.append(value)
            return "recorded"

        agent = create_agent(
            model=model(
                script(
                    (lambda call: call.last.type == "tool", "Recorded."),
                    (anything, tool_call("record", {"value": 7})),
                )
            ),
            tools=[record],
            pattern="react",
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
            interrupt_before=["tools"],
        )
        config = {"configurable": {"thread_id": "react-breakpoint"}}
        with pytest.raises(ExecutionPaused) as paused:
            agent.invoke("Record 7", config)
        assert paused.value.interrupts[0].value == {
            "type": BREAKPOINT_INTERRUPT,
            "interrupt_before": ["tools"],
            "interrupt_after": [],
        }
        assert executed == []
        assert agent.resume(config).text == "Recorded."
        assert executed == [7]
        with pytest.raises(ConfigurationError, match="nothing to resume"):
            agent.resume(config)

    def test_interrupt_after_a_topology_node(
        self, model: Callable[..., ScriptedChatModel], saver: Any
    ) -> None:
        plan = {"steps": [{"id": "a", "task": "Find A"}]}
        scripted = model(
            script(
                (wants("Plan"), tool_call("Plan", plan)),
                (system_has("assigned step"), "result-a"),
                (system_has("Combine the step results"), "Planned answer."),
            )
        )
        agent = create_agent(
            model=scripted,
            pattern="planner",
            checkpointer=saver,
            checkpoint_authorizer=allow_threads,
            interrupt_after=["plan"],
        )
        config = {"configurable": {"thread_id": "planner-breakpoint"}}
        paused = agent.invoke_loop("Plan it", config)
        assert paused.paused
        assert paused.state.phase.value == "waiting"
        assert sum(call.wants("Plan") for call in scripted.calls) == 1
        assert agent.resume(config).text == "Planned answer."
        assert sum(call.wants("Plan") for call in scripted.calls) == 1


class TestSupervisor:
    def agent(self, scripted: ScriptedChatModel, **options: Any) -> Any:
        return create_agent(
            model=scripted,
            pattern="supervisor",
            workers=[RESEARCHER, WRITER],
            **options,
        )

    def test_delegates_then_finishes_with_answer(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (
                    lambda call: (
                        call.wants("SupervisorRoute") and "[researcher]" in call.prompt
                    ),
                    tool_call(
                        "SupervisorRoute", {"next": "FINISH", "final_answer": "42."}
                    ),
                ),
                (
                    wants("SupervisorRoute"),
                    tool_call(
                        "SupervisorRoute",
                        {"next": "researcher", "assignment": "Find the answer."},
                    ),
                ),
                (system_has("You are researcher"), "The answer is 42."),
            )
        )
        assert self.agent(scripted).invoke("What is the answer?").text == "42."
        worker_call = next(
            c for c in scripted.calls if "You are researcher" in c.system
        )
        assert "Find the answer." in worker_call.prompt

    def test_finish_without_answer_synthesizes(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (
                    wants("SupervisorRoute"),
                    tool_call("SupervisorRoute", {"next": "FINISH"}),
                ),
                (system_has("from the workers' results"), "Synthesized."),
            )
        )
        assert self.agent(scripted).invoke("Summarize").text == "Synthesized."

    def test_handoff_limit_forces_synthesis(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (
                    wants("SupervisorRoute"),
                    tool_call(
                        "SupervisorRoute", {"next": "writer", "assignment": "Write."}
                    ),
                ),
                (system_has("You are writer"), "Draft."),
                (system_has("from the workers' results"), "Final from limit."),
            )
        )
        agent = self.agent(scripted, limits=ExecutionLimits(max_handoffs=1))
        assert agent.invoke("Write").text == "Final from limit."
        assert sum(c.wants("SupervisorRoute") for c in scripted.calls) == 1

    def test_reserved_names(self, model: Callable[..., ScriptedChatModel]) -> None:
        with pytest.raises(ConfigurationError, match="reserved worker names"):
            create_agent(
                model=model(),
                pattern="supervisor",
                workers=[WorkerSpec(name="FINISH", description="x")],
            )


class TestSwarm:
    def test_handoffs_then_stop(self, model: Callable[..., ScriptedChatModel]) -> None:
        scripted = model(
            script(
                (
                    system_has("You are researcher"),
                    tool_call(
                        "SwarmTurn",
                        {"contribution": "Facts gathered.", "next": "writer"},
                    ),
                ),
                (
                    system_has("You are writer"),
                    tool_call(
                        "SwarmTurn",
                        {
                            "contribution": "Prose written.",
                            "next": "STOP",
                            "final_answer": "Swarm answer.",
                        },
                    ),
                ),
            )
        )
        agent = create_agent(
            model=scripted, pattern="swarm", workers=[RESEARCHER, WRITER]
        )
        assert agent.invoke("Write a report").text == "Swarm answer."

    def test_stop_without_answer_and_handoff_limit(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (
                    system_has("from the peers' work"),
                    lambda call: f"Combined: {'Facts' in call.prompt}",
                ),
                (
                    anything,
                    tool_call(
                        "SwarmTurn", {"contribution": "Facts.", "next": "researcher"}
                    ),
                ),
            )
        )
        agent = create_agent(
            model=scripted,
            pattern="swarm",
            workers=[RESEARCHER],
            limits=ExecutionLimits(max_handoffs=2),
        )
        assert agent.invoke("Loop").text == "Combined: True"


class TestPlanner:
    def test_runs_dependency_waves_then_synthesizes(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        plan = {
            "steps": [
                {"id": "a", "task": "Find A"},
                {"id": "b", "task": "Find B"},
                {"id": "c", "task": "Combine A and B", "depends_on": ["a", "b"]},
            ]
        }
        scripted = model(
            script(
                (wants("Plan"), tool_call("Plan", plan)),
                (
                    system_has("assigned step"),
                    lambda call: (
                        "result-c"
                        if "Combine A and B" in call.prompt
                        else f"result-{'a' if 'Find A' in call.prompt else 'b'}"
                    ),
                ),
                (system_has("Combine the step results"), lambda call: call.prompt),
            )
        )
        answer = create_agent(model=scripted, pattern="planner").invoke("Plan it").text
        assert "[a] Find A\nresult-a" in answer
        assert "[c] Combine A and B\nresult-c" in answer
        combine = next(c for c in scripted.calls if "Combine A and B" in c.prompt)
        assert "[a] result-a" in combine.prompt
        assert "[b] result-b" in combine.prompt

    @pytest.mark.parametrize("max_workers", [1, 2, 3])
    def test_steps_run_in_parallel_up_to_max_workers(
        self, model: Callable[..., ScriptedChatModel], max_workers: int
    ) -> None:
        plan = {"steps": [{"id": name, "task": f"Find {name}"} for name in "abc"]}
        overlap = Overlap()
        scripted = model(
            script(
                (wants("Plan"), tool_call("Plan", plan)),
                (system_has("assigned step"), overlap("found")),
                (system_has("Combine the step results"), lambda call: call.prompt),
            )
        )
        agent = create_agent(
            model=scripted,
            pattern="planner",
            limits=ExecutionLimits(max_workers=max_workers),
        )
        answer = agent.invoke("Find a, b, and c", {"max_concurrency": 1}).text
        assert all(f"[{name}] Find {name}\nfound" in answer for name in "abc")
        assert overlap.peak == max_workers

    def test_invalid_plans_fail(self, model: Callable[..., ScriptedChatModel]) -> None:
        cyclic = {
            "steps": [
                {"id": "a", "task": "A", "depends_on": ["b"]},
                {"id": "b", "task": "B", "depends_on": ["a"]},
            ]
        }
        agent = create_agent(
            model=model(script((wants("Plan"), tool_call("Plan", cyclic)))),
            pattern="planner",
        )
        with pytest.raises(ExecutionError):
            agent.invoke("Plan")

    @pytest.mark.parametrize(
        ("steps", "message"),
        [
            ([PlanStep(id="a", task="x"), PlanStep(id="a", task="y")], "unique"),
            ([PlanStep(id="a", task="x", depends_on=["z"])], "unknown steps"),
            ([PlanStep(id=f"s{i}", task="x") for i in range(3)], "the limit is 2"),
        ],
    )
    def test_validate_plan_errors(self, steps: list[PlanStep], message: str) -> None:
        with pytest.raises(ExecutionError, match=message):
            validate_plan(steps, 2)


@st.composite
def acyclic_plans(draw: st.DrawFn) -> list[PlanStep]:
    size = draw(st.integers(1, 8))
    steps = []
    for index in range(size):
        dependencies = draw(
            st.lists(st.sampled_from(range(index)), unique=True)
            if index
            else st.just([])
        )
        steps.append(
            PlanStep(
                id=f"s{index}", task="t", depends_on=[f"s{d}" for d in dependencies]
            )
        )
    return draw(st.permutations(steps))


@given(acyclic_plans())
def test_waves_respect_dependencies_and_cover_every_step(steps: list[PlanStep]) -> None:
    plan = validate_plan(steps, 8)
    results: dict[str, str] = {}
    waves = 0
    while ready := ready_steps(plan, results):
        for step in ready:
            assert all(dependency in results for dependency in step["depends_on"])
        results.update({step["id"]: "done" for step in ready})
        waves += 1
    assert set(results) == {step.id for step in steps}
    assert waves <= len(steps)


class TestReflection:
    def responder(self, verdicts: list[bool]) -> Callable[[Call], Any]:
        queue = list(verdicts)
        return script(
            (system_has("Write a complete, correct answer"), "Draft."),
            (
                wants("Review"),
                lambda call: tool_call(
                    "Review",
                    {"approved": queue.pop(0), "issues": ["Be precise."]},
                ),
            ),
            (system_has("Revise the answer"), "Revised."),
        )

    def test_early_exit_when_approved(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(self.responder([True]))
        agent = create_agent(model=scripted, pattern="reflection", max_reflections=3)
        assert agent.invoke("Answer").text == "Draft."
        assert len(scripted.calls) == 2

    def test_revises_until_approved(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(self.responder([False, True]))
        agent = create_agent(model=scripted, pattern="reflection", max_reflections=3)
        assert agent.invoke("Answer").text == "Revised."
        revise = next(c for c in scripted.calls if "Revise the answer" in c.system)
        assert "- Be precise." in revise.prompt

    def test_stops_after_max_rounds(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(self.responder([False]))
        agent = create_agent(model=scripted, pattern="reflection", max_reflections=1)
        assert agent.invoke("Answer").text == "Revised."
        assert sum(c.wants("Review") for c in scripted.calls) == 1


class TestBlackboardPipelineHybrid:
    def test_blackboard_contributions_are_ordered(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (system_has("You are researcher"), "Facts."),
                (system_has("You are writer"), "Prose."),
                (system_has("Combine the board"), lambda call: call.prompt),
            )
        )
        agent = create_agent(
            model=scripted, pattern="blackboard", workers=[RESEARCHER, WRITER]
        )
        board = agent.invoke("Report").text
        assert board.index("[researcher] Facts.") < board.index("[writer] Prose.")

    @pytest.mark.parametrize(("max_workers", "peak"), [(1, 1), (2, 2), (4, 3)])
    def test_blackboard_contributors_run_up_to_max_workers_at_once(
        self, model: Callable[..., ScriptedChatModel], max_workers: int, peak: int
    ) -> None:
        editor = WorkerSpec(name="editor", description="Edits prose.")
        overlap = Overlap()
        scripted = model(
            script(
                (system_has("Combine the board"), lambda call: call.prompt),
                (anything, overlap("noted")),
            )
        )
        agent = create_agent(
            model=scripted,
            pattern="blackboard",
            workers=[RESEARCHER, WRITER, editor],
            limits=ExecutionLimits(max_workers=max_workers),
        )
        board = agent.invoke("Report").text
        assert all(f"[{name}] noted" in board for name in ("researcher", "writer"))
        assert "[editor] noted" in board
        assert overlap.peak == peak

    def test_pipeline_stages_and_transforms(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (system_has("Outline"), "outline"),
                (
                    system_has("Expand"),
                    lambda call: f"expanded {call.prompt.count('outline')}",
                ),
            )
        )
        stages = [
            PipelineStage(name="outline", instruction="Outline."),
            PipelineStage(name="expand", instruction="Expand.", transform=str.upper),
        ]
        agent = create_agent(model=scripted, pattern="pipeline", stages=stages)
        assert agent.invoke("Write").text == "EXPANDED 1"

    def test_pipeline_default_stages(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(script((anything, "stage output")))
        agent = create_agent(model=scripted, pattern="pipeline")
        assert agent.invoke("Do").text == "stage output"
        assert len(scripted.calls) == 3

    def test_hybrid_passes_results_forward(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (system_has("Use the available tools"), "react result"),
                (
                    system_has("Analyze the request"),
                    lambda call: f"saw: {'react result' in call.prompt}",
                ),
                (anything, "pipeline step"),
            )
        )
        agent = create_agent(
            model=scripted, pattern="hybrid", composition=["react", "pipeline"]
        )
        assert agent.invoke("Do").text == "pipeline step"
        analyze = next(c for c in scripted.calls if "Analyze the request" in c.system)
        assert "Result of the previous stage:\nreact result" in analyze.prompt


class TestCompiler:
    def context(self, model: ScriptedChatModel, **changes: Any) -> TopologyBuildContext:
        options: dict[str, Any] = {
            "model": model,
            "tools": (),
            "middleware": (),
            "workers": (),
            "stages": (),
            "system_prompt": None,
            "limits": ExecutionLimits(),
            "max_reflections": 1,
        }
        return TopologyBuildContext(**{**options, **changes})

    def test_rejects_mismatched_builder_registration(self) -> None:
        with pytest.raises(ConfigurationError, match="reports"):
            ArchitectureCompiler({TopologyName.REACT: SwarmTopology()})

    def test_custom_builders_and_failures(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Broken:
            name = TopologyName.REACT

            def build(self, context: TopologyBuildContext) -> TopologyArtifact:
                raise RuntimeError("boom")

        class Liar:
            name = TopologyName.REACT

            def build(self, context: TopologyBuildContext) -> TopologyArtifact:
                return TopologyArtifact(TopologyName.PLANNER, model())

        spec = ArchitectureSpec(topology=TopologyName.REACT)
        context = self.context(model())
        with pytest.raises(CompilationError, match="failed to compile"):
            ArchitectureCompiler({TopologyName.REACT: Broken()}).compile(spec, context)
        with pytest.raises(CompilationError, match="returned planner"):
            ArchitectureCompiler({TopologyName.REACT: Liar()}).compile(spec, context)
        compiler = ArchitectureCompiler()
        assert set(compiler.topologies) == set(TopologyName)

    def test_hybrid_requires_child_compiler(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        from agloom.topology.hybrid import HybridTopology

        with pytest.raises(CompilationError, match="child compiler"):
            HybridTopology().build(self.context(model()))

    @pytest.mark.parametrize(
        ("builder", "reserved"),
        [(SupervisorTopology(), "FINISH"), (SwarmTopology(), "STOP")],
    )
    def test_worker_topologies_validate_workers(
        self,
        model: Callable[..., ScriptedChatModel],
        builder: Any,
        reserved: str,
    ) -> None:
        with pytest.raises(ConfigurationError, match="at least one worker"):
            builder.build(self.context(model()))
        worker = WorkerSpec(name=reserved, description="x")
        with pytest.raises(ConfigurationError, match="reserved"):
            builder.build(self.context(model(), workers=(worker,)))

    def test_blackboard_requires_workers(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        from agloom.topology.blackboard import BlackboardTopology

        with pytest.raises(ConfigurationError, match="at least one worker"):
            BlackboardTopology().build(self.context(model()))


class TestSharedHelpers:
    def test_request_includes_bounded_conversation_and_context(self) -> None:
        messages = [HumanMessage(content=f"q{i}") for i in range(12)]
        messages.insert(3, AIMessage(content="a"))
        messages.insert(4, SystemMessage(content="ignored"))
        prompt = request(
            {"task": "now", "messages": messages, "context": "CTX"},
            ("Section", "body"),
            ("Empty", None),
        )
        assert prompt.startswith("Conversation so far:")
        assert "Request:\nnow" in prompt
        assert "Section:\nbody" in prompt
        assert "Empty" not in prompt
        assert prompt.endswith("CTX")
        assert "ignored" not in conversation(messages)
        assert conversation([HumanMessage(content="only")]) == ""

    def test_with_context(self) -> None:
        human = HumanMessage(content="hi", id="1")
        assert with_context([human], None) == [human]
        updated = with_context([human, AIMessage(content="x")], "CTX")
        assert updated[0].text == "hi\n\nCTX"
        assert updated[0].id == "1"
        assert with_context([AIMessage(content="x")], "CTX")[-1].text == "CTX"

    def test_topology_input_and_text_helpers(self) -> None:
        assert topology_input(True, "task", [], None) == {
            "messages": [HumanMessage(content="task")]
        }
        assert topology_input(False, "t", [], "c") == {
            "task": "t",
            "messages": [],
            "context": "c",
        }
        assert node_name("worker", 2, "Data Analyst!") == "worker_2_data_analyst"
        assert node_name("worker", 0, "!!!") == "worker_0_worker"
        assert get_args(choices(["a", "b"])) == ("a", "b")
        with pytest.raises(ExecutionError, match="no structured response"):
            structured_response({})
        assert final_message("plain text") is None
        assert answer_update(None) == {"output": "", "loop": {}}
        signalled = AIMessage(content="x", additional_kwargs={"loop": {"quality": 1}})
        assert answer_update({"messages": [signalled]}, str.upper) == {
            "output": "X",
            "loop": {"quality": 1},
        }


class TestAsyncAndRepairPaths:
    async def test_planner_repairs_an_invalid_plan(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        cyclic = {
            "steps": [
                {"id": "a", "task": "A", "depends_on": ["b"]},
                {"id": "b", "task": "B", "depends_on": ["a"]},
            ]
        }
        valid = {"steps": [{"id": "a", "task": "Find A"}]}

        def plan(call: Call) -> Any:
            return tool_call("Plan", valid if "rejected" in call.prompt else cyclic)

        scripted = model(
            script(
                (wants("Plan"), plan),
                (system_has("assigned step"), "result-a"),
                (system_has("Combine the step results"), "repaired"),
            )
        )
        agent = create_agent(model=scripted, pattern="planner")
        assert (await agent.ainvoke("Plan it")).text == "repaired"
        corrections = [
            c for c in scripted.calls if c.wants("Plan") and "rejected" in c.prompt
        ]
        assert "dependency cycle" in corrections[0].prompt

    def test_planner_repairs_a_missing_plan(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        valid = {"steps": [{"id": "a", "task": "Find A"}]}
        scripted = model(
            script(
                (
                    wants("Plan"),
                    lambda call: (
                        tool_call("Plan", valid)
                        if "rejected" in call.prompt
                        else "no plan at all"
                    ),
                ),
                (system_has("assigned step"), "result-a"),
                (system_has("Combine the step results"), "planned"),
            )
        )
        assert (
            create_agent(model=scripted, pattern="planner").invoke("Go").text
            == "planned"
        )

    async def test_async_supervisor_shortcut_and_worker_instructions(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(
            script(
                (
                    lambda call: (
                        call.wants("SupervisorRoute") and "[analyst]" in call.prompt
                    ),
                    tool_call(
                        "SupervisorRoute", {"next": "FINISH", "final_answer": "Done."}
                    ),
                ),
                (
                    wants("SupervisorRoute"),
                    tool_call(
                        "SupervisorRoute", {"next": "analyst", "assignment": "Go."}
                    ),
                ),
                (system_has("Cite sources."), "Analysis."),
            )
        )
        analyst = WorkerSpec(
            name="analyst", description="Analyzes.", instructions="Cite sources."
        )
        agent = create_agent(model=scripted, pattern="supervisor", workers=[analyst])
        assert (await agent.ainvoke("Analyze")).text == "Done."

    async def test_writers_and_graphs_outside_agent_runs(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        scripted = model(lambda call: "")
        context = TopologyBuildContext(
            model=scripted,
            tools=(),
            middleware=(),
            workers=(),
            stages=(),
            system_prompt=None,
            limits=ExecutionLimits(),
            max_reflections=1,
            screen_answers=True,
        )
        writer = context.writer("Write.", name="writer")
        assert (await writer.ainvoke("prompt")).text == ""
        assert writer.invoke("prompt").text == ""
        spec = ArchitectureSpec(topology=TopologyName.PIPELINE)
        graph = ArchitectureCompiler().compile(spec, context).graph
        result = graph.invoke({"task": "t", "messages": [HumanMessage(content="t")]})
        assert result["output"] == ""

    def test_builder_configuration_errors_pass_through(
        self, model: Callable[..., ScriptedChatModel]
    ) -> None:
        class Strict:
            name = TopologyName.REACT

            def build(self, context: TopologyBuildContext) -> TopologyArtifact:
                raise ConfigurationError("strict builder")

        context = TopologyBuildContext(
            model=model(),
            tools=(),
            middleware=(),
            workers=(),
            stages=(),
            system_prompt=None,
            limits=ExecutionLimits(),
            max_reflections=1,
        )
        compiler = ArchitectureCompiler({TopologyName.REACT: Strict()})
        with pytest.raises(ConfigurationError, match="strict builder"):
            compiler.compile(ArchitectureSpec(topology=TopologyName.REACT), context)
