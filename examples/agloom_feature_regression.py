"""Reusable full-surface Agloom regression checks for the real-world apps."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Sequence
from typing import Any

from behaviorweave import BehaviorEvent, InterventionType, PolicyRule
from feedback_manager import (
    FeedbackCategory,
    FeedbackQuery,
    FeedbackSource,
    FeedbackTarget,
    FeedbackTargetType,
)
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from agloom import (
    AgentEvent,
    ArchitectureCompiler,
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityPolicy,
    CapabilityRegistry,
    CapabilityRouter,
    CapabilitySelection,
    InMemoryCapabilityRegistry,
    Observer,
    PipelineStage,
    RecursionPolicy,
    WorkerSpec,
    create_agent,
)
from agloom.capabilities.integrations import mcp_refresh_policy, xai_runtime
from agloom.strategy.engine import StrategyEngine


class Summary(BaseModel):
    title: str
    complete: bool


class ToolCapableModel(FakeListChatModel):
    def bind_tools(self, tools: object, **kwargs: object) -> ToolCapableModel:
        del tools, kwargs
        return self


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float("status" in normalized),
            float("document" in normalized),
        ]


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        del query, run_manager
        return self.documents


class StableReranker(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Any = None,
    ) -> Sequence[Document]:
        del query, callbacks
        return tuple(documents)


class DenyBlocked(CapabilityPolicy):
    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        del task
        return not capability.metadata.get("blocked", False)


class PassthroughRouter(CapabilityRouter):
    def route(self, task: str, catalog: CapabilityRegistry) -> CapabilitySelection:
        return CapabilitySelection(task, catalog.descriptors())

    async def aroute(
        self, task: str, catalog: CapabilityRegistry
    ) -> CapabilitySelection:
        return self.route(task, catalog)


class RecordingObserver(Observer):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


class Client:
    async def get_tools(self, *, server_name: str | None = None) -> list[object]:
        del server_name
        return []

    async def get_resources(
        self, server_name: str | None = None, *, uris: object = None
    ) -> list[object]:
        del server_name, uris
        return []

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        *,
        arguments: object = None,
    ) -> None:
        del server_name, prompt_name, arguments


class RefreshSource:
    async def discover(self) -> object:
        from refresh_engine import DiscoveryResult

        async def resources():
            if False:
                yield

        return DiscoveryResult(resources=resources(), complete=True)


def _model(*responses: str) -> ToolCapableModel:
    return ToolCapableModel(responses=list(responses) or ["unused"])


def assert_complete_parameter_surface(application_id: str) -> None:
    """Construct compatible agents whose options cover every public parameter."""

    @tool
    def status() -> str:
        """Return the application status."""
        return "ready"

    registry = InMemoryCapabilityRegistry(
        [
            CapabilityDescriptor(
                name="blocked",
                description="Blocked test capability.",
                kind=CapabilityKind.APPLICATION,
                value=object(),
                metadata={"blocked": True},
            )
        ]
    )
    observer = RecordingObserver()
    source = RefreshSource()

    async def refresh_operation(*args: object) -> None:
        del args

    primary: dict[str, Any] = {
        "model": _model("ready"),
        "tools": [status],
        "pattern": "react",
        "recursion": True,
        "recursion_policy": RecursionPolicy(max_depth=2),
        "workers": [WorkerSpec(name="worker", description="Regression worker.")],
        "stages": [PipelineStage(name="verify", instruction="Verify output.")],
        "event_sink": lambda event: None,
        "observers": [observer],
        "capabilities": {"application_status": lambda: "ready"},
        "capability_registry": registry,
        "capability_policy": DenyBlocked(),
        "capability_embeddings": KeywordEmbeddings(),
        "capability_retriever": StaticRetriever(
            documents=[
                Document(
                    page_content="Application status",
                    metadata={"capability_name": "application_status"},
                )
            ]
        ),
        "capability_reranker": StableReranker(),
        "max_selected_capabilities": 8,
        "middleware": [ModelCallLimitMiddleware(run_limit=8)],
        "feedback_options": {},
        "behavior_options": {},
        "context_model": _model("context"),
        "context_options": {
            "trigger": ("tokens", 100_000),
            "keep": ("messages", 8),
        },
        "mcp_options": {"max_concurrency": 2, "operation_timeout": 5.0},
        "mcp_client": Client(),
        "mcp_server_name": "regression",
        "mcp_server_id": f"{application_id}-mcp",
        "mcp_client_server_name": "regression",
        "mcp_discover_resources": True,
        "mcp_metadata": {"application": application_id},
        "mcp_refresh": mcp_refresh_policy(on_register=True),
        "refresh_source": source,
        "refresh_operation": refresh_operation,
        "refresh_options": {},
        "xai_options": {"application_id": application_id},
        "xai_enabled": True,
        "structured_output_schema": Summary,
        "structured_output_options": {"inject_instructions": False},
        "system_prompt": "Use verified application evidence.",
        "max_reflections": 2,
        "strategy": StrategyEngine(),
        "compiler": ArchitectureCompiler(),
    }
    agent = create_agent(**primary)
    assert {
        "feedback_manager",
        "behavior_engine",
        "context_middleware",
        "mcp_runtime",
        "refresh_engine",
        "xai_runtime",
        "structured_output",
    } <= set(agent.capabilities.names)

    checkpoint: dict[str, Any] = {
        "model": _model("draft", "final"),
        "tools": (),
        "pattern": "pipeline",
        "checkpointer": InMemorySaver(),
        "checkpoint_authorizer": lambda thread_id: thread_id == application_id,
        "interrupt_before": ["stage_0_prepare"],
        "interrupt_after": (),
        "stages": [PipelineStage(name="prepare", instruction="Prepare output.")],
        "xai_enabled": False,
    }
    create_agent(**checkpoint)

    hybrid: dict[str, Any] = {
        "model": _model("answer"),
        "tools": (),
        "pattern": "hybrid",
        "composition": ["react", "reflection"],
        "xai_enabled": False,
    }
    create_agent(**hybrid)

    custom_router: dict[str, Any] = {
        "model": _model(),
        "tools": (),
        "pattern": "react",
        "capability_router": PassthroughRouter(),
        "xai_enabled": False,
    }
    create_agent(**custom_router)

    explicit_xai: dict[str, Any] = {
        "model": _model(),
        "tools": (),
        "pattern": "react",
        "explainability": xai_runtime(application_id=f"{application_id}-explicit"),
    }
    create_agent(**explicit_xai)

    covered = (
        set(primary)
        | set(checkpoint)
        | set(hybrid)
        | set(custom_router)
        | set(explicit_xai)
    )
    assert covered == set(inspect.signature(create_agent).parameters)


def assert_all_integration_combinations() -> None:
    """Construct all enabled/disabled combinations of seven integrations."""

    names = ("feedback", "behavior", "context", "mcp", "refresh", "xai", "structured")
    for values in itertools.product((False, True), repeat=len(names)):
        enabled = dict(zip(names, values, strict=True))
        options: dict[str, Any] = {
            "model": _model(),
            "tools": (),
            "pattern": "react",
            "feedback_options": {} if enabled["feedback"] else None,
            "behavior_options": {} if enabled["behavior"] else None,
            "context_options": (
                {"trigger": ("tokens", 100_000), "keep": ("messages", 5)}
                if enabled["context"]
                else None
            ),
            "mcp_options": {} if enabled["mcp"] else None,
            "mcp_client": Client() if enabled["mcp"] else None,
            "mcp_server_name": "matrix" if enabled["mcp"] else None,
            "refresh_source": object() if enabled["refresh"] else None,
            "refresh_operation": ((lambda *args: None) if enabled["refresh"] else None),
            "xai_enabled": enabled["xai"],
            "structured_output_schema": Summary if enabled["structured"] else None,
            "structured_output_options": (
                {"inject_instructions": False} if enabled["structured"] else None
            ),
        }
        agent = create_agent(**options)
        expected = {
            "feedback_manager": enabled["feedback"],
            "behavior_engine": enabled["behavior"],
            "context_middleware": enabled["context"],
            "mcp_runtime": enabled["mcp"],
            "refresh_engine": enabled["refresh"],
            "xai_runtime": enabled["xai"],
            "structured_output": enabled["structured"],
        }
        for capability, present in expected.items():
            assert (capability in agent.capabilities.names) is present


def assert_all_execution_modes() -> None:
    """Compile all eight topologies and execute the automatic DIRECT path."""

    topologies = (
        "react",
        "supervisor",
        "pipeline",
        "planner",
        "reflection",
        "swarm",
        "blackboard",
        "hybrid",
    )
    for topology in topologies:
        options: dict[str, Any] = {
            "model": _model("ready"),
            "pattern": topology,
            "xai_enabled": False,
        }
        if topology in {"supervisor", "swarm", "blackboard"}:
            options["workers"] = [
                WorkerSpec(name="worker", description="Regression worker.")
            ]
        if topology == "pipeline":
            options["stages"] = [
                PipelineStage(name="verify", instruction="Verify output.")
            ]
        if topology == "hybrid":
            options["composition"] = ["react", "reflection"]
        agent = create_agent(**options)
        assert agent.config.pattern_candidates[0].value == topology

    direct = create_agent(
        model=_model('{"complexity":0,"estimated_steps":1}', "direct ready"),
        xai_enabled=False,
    )
    assert direct.invoke("Answer directly.").content == "direct ready"


async def assert_semantic_runtime_features(application_id: str) -> None:
    """Exercise routing, events, MCP setup, structured output, and policies."""

    observer = RecordingObserver()
    events: list[AgentEvent] = []
    agent = create_agent(
        model=_model("runtime ready"),
        tools=(),
        pattern="react",
        capabilities={"double": lambda value: value * 2},
        observers=[observer],
        event_sink=events.append,
        feedback_options={},
        behavior_options={
            "policies": (
                PolicyRule(
                    policy_id="pause-repeat",
                    pattern_id="repeated_tool_call",
                    threshold=2,
                    intervention=InterventionType.PAUSE,
                ),
            )
        },
        context_options={
            "trigger": ("tokens", 100_000),
            "keep": ("messages", 5),
        },
        mcp_options={},
        mcp_client=Client(),
        mcp_server_name="runtime",
        mcp_server_id=f"{application_id}-runtime",
        xai_options={"application_id": application_id},
    )
    result = await agent.ainvoke("Report runtime readiness.")
    assert result.content == "runtime ready"
    assert agent.execute_capability("double", 4) == 8
    assert "double" in agent.route_capabilities("Use double").names
    assert any(event.name == "ExecutionCompleted" for event in events)
    assert any(event.name == "ExecutionCompleted" for event in observer.events)
    assert await agent.capabilities.resolve("mcp_runtime").health(
        f"{application_id}-runtime"
    )
    assert agent.config.middleware
    assert agent.xai is not None

    behavior = agent.capabilities.resolve("behavior_engine")
    first = behavior.process(
        BehaviorEvent.tool_call("lookup", {"id": "1"}, scope=application_id)
    )
    second = behavior.process(
        BehaviorEvent.tool_call("lookup", {"id": "1"}, scope=application_id)
    )
    assert first.intervention.kind is InterventionType.NOOP
    assert second.intervention.kind is InterventionType.PAUSE

    manager = agent.capabilities.resolve("feedback_manager")
    feedback = await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.RATING,
        target=FeedbackTarget(
            type=FeedbackTargetType.GENERATION,
            id=f"{application_id}-answer",
        ),
        payload={"rating": "up"},
    )
    await manager.acknowledge(feedback.feedback_id)
    await manager.mark_handled(feedback.feedback_id)
    resolved = await manager.resolve(
        feedback.feedback_id,
        resolution={"verified": True},
    )
    stored = await manager.query(FeedbackQuery(target_id=f"{application_id}-answer"))
    assert resolved.status.value == "resolved"
    assert len(stored) == 1

    structured = create_agent(
        model=_model('<xstructured>{"title":"Verified","complete":true}</xstructured>'),
        pattern="react",
        structured_output_schema=Summary,
        structured_output_options={"inject_instructions": False},
        xai_enabled=False,
    )
    parsed = structured.structured_output.invoke("Return the report.").structured
    assert parsed == Summary(title="Verified", complete=True)

    refresh_agent = create_agent(
        model=_model(),
        pattern="react",
        refresh_source=RefreshSource(),
        refresh_operation=lambda *args: None,
        xai_enabled=False,
    )
    refresh = refresh_agent.capabilities.resolve("refresh_engine")
    refresh_result = await refresh.refresh()
    assert refresh_result.added_count == 0
    await refresh.close()

    checkpoint_agent = create_agent(
        model=_model("final"),
        pattern="pipeline",
        checkpointer=InMemorySaver(),
        checkpoint_authorizer=lambda thread_id: thread_id == application_id,
        interrupt_before=["stage_0_prepare"],
        stages=[PipelineStage(name="prepare", instruction="Prepare output.")],
        xai_enabled=False,
    )
    config = {"configurable": {"thread_id": application_id}}
    paused = checkpoint_agent.invoke("Prepare the report.", config=config)
    resumed = checkpoint_agent.invoke(Command(resume=True), config=config)
    assert paused["__interrupt__"]
    assert resumed.content == "final"

    streaming = create_agent(
        model=_model('{"complexity":0,"estimated_steps":1}', "stream ready"),
        recursion=True,
        recursion_policy=RecursionPolicy(max_depth=2),
        xai_enabled=False,
    )
    chunks = list(streaming.stream("Stream readiness."))
    assert "".join(chunk.content for chunk in chunks) == "stream ready"
    assert streaming.config.recursion is True
