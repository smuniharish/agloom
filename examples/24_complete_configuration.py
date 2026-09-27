from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from _support import real_model, verified_text
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
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph_xai import Execution, ProvenanceStore, StoreFilter
from pydantic import BaseModel
from refresh_engine import (
    DiscoveryResult,
    PlanAction,
    RefreshConfig,
    Resource,
    ResourceSnapshot,
)

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


class Answer(BaseModel):
    result: int
    verified: bool


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float("multiply" in normalized),
            float("status" in normalized),
            float("guide" in normalized),
        ]


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager):
        del query, run_manager
        return self.documents


class StableReranker(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        del query, callbacks
        return tuple(documents)


class DenyBlockedNetwork(CapabilityPolicy):
    def allows(
        self,
        task: str,
        capability: CapabilityDescriptor,
    ) -> bool:
        del task
        return not capability.metadata.get("blocked", False)


class RecordingObserver(Observer):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


class FeedbackEventSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)


class RecordingCompiler(ArchitectureCompiler):
    def __init__(self) -> None:
        super().__init__()
        self.compilations = 0

    def compile(self, spec: Any, config: Any) -> Any:
        self.compilations += 1
        return super().compile(spec, config)


class PassthroughRouter(CapabilityRouter):
    def route(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> CapabilitySelection:
        return CapabilitySelection(task, catalog.descriptors())

    async def aroute(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> CapabilitySelection:
        return self.route(task, catalog)


class DocumentSource:
    def __init__(self) -> None:
        self.documents = {"shipping": "Shipping takes five days."}

    async def discover(self) -> DiscoveryResult:
        async def resources():
            for document_id in self.documents:
                yield Resource(resource_id=document_id)

        return DiscoveryResult(resources=resources(), complete=True)

    async def snapshot(self, resource: Resource) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=resource.resource_id,
            content=self.documents[resource.resource_id],
        )


async def xai_executions(agent) -> list[Execution]:
    store = agent.xai.registry.require(ProvenanceStore)
    return [
        item
        async for item in store.query(
            StoreFilter(
                application_id="complete-example",
                tenant_id="example-tenant",
                item_type=Execution,
            )
        )
    ]


async def main() -> None:
    model = real_model()
    tool_calls: list[tuple[int, int]] = []
    indexed_documents: dict[str, str] = {}
    runtime_events: list[AgentEvent] = []
    feedback_sink = FeedbackEventSink()
    observer = RecordingObserver()
    compiler = RecordingCompiler()
    registry = InMemoryCapabilityRegistry(
        [
            CapabilityDescriptor(
                name="blocked_network",
                description="A deliberately blocked network capability.",
                kind=CapabilityKind.APPLICATION,
                value=object(),
                metadata={"blocked": True},
            )
        ]
    )
    source = DocumentSource()

    @tool
    def multiply(left: int, right: int) -> int:
        """Multiply two integers exactly."""

        tool_calls.append((left, right))
        return left * right

    async def update_index(
        resource: Resource | None,
        snapshot: ResourceSnapshot | None,
        action: PlanAction,
        request,
    ) -> None:
        del request
        if action is PlanAction.DELETE:
            return
        assert resource is not None
        assert snapshot is not None
        indexed_documents[resource.resource_id] = str(snapshot.content)

    server = Path(__file__).with_name("_mcp_server.py")
    mcp_client = MultiServerMCPClient(
        {
            "demo": {
                "command": sys.executable,
                "args": [str(server)],
                "transport": "stdio",
            }
        }
    )
    primary_options: dict[str, Any] = {
        "model": model,
        "tools": [multiply],
        "pattern": "react",
        "recursion": True,
        "recursion_policy": RecursionPolicy(
            max_depth=2,
            max_workers=3,
            max_subtasks=6,
            max_execution_time=90,
        ),
        "workers": [
            WorkerSpec(
                name="calculator",
                description="Perform exact arithmetic.",
            )
        ],
        "stages": [
            PipelineStage(
                name="verify",
                instruction="Verify the numeric result.",
            )
        ],
        "event_sink": runtime_events.append,
        "observers": [observer],
        "capabilities": {
            "application_status": lambda: "ready",
        },
        "capability_registry": registry,
        "capability_policy": DenyBlockedNetwork(),
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
        "feedback_options": {
            "observability_sink": feedback_sink,
        },
        "behavior_options": {
            "policies": (
                PolicyRule(
                    policy_id="pause-repeat",
                    pattern_id="repeated_tool_call",
                    threshold=2,
                    intervention=InterventionType.PAUSE,
                ),
            )
        },
        "context_model": model,
        "context_options": {
            "policy": "maximum_compression",
            "trigger": ("tokens", 100_000),
            "keep": ("messages", 8),
            "maximum_context_tokens": 16_000,
            "safety_margin": 0.2,
        },
        "mcp_options": {
            "max_concurrency": 4,
            "operation_timeout": 15.0,
        },
        "mcp_client": mcp_client,
        "mcp_server_name": "complete-demo",
        "mcp_server_id": "complete-demo",
        "mcp_client_server_name": "demo",
        "mcp_discover_resources": True,
        "mcp_metadata": {"owner": "complete-example"},
        "mcp_refresh": mcp_refresh_policy(on_register=True),
        "refresh_source": source,
        "refresh_operation": update_index,
        "refresh_options": {
            "config": RefreshConfig(max_concurrency=2),
        },
        "xai_options": {
            "application_id": "complete-example",
            "tenant_id": "example-tenant",
        },
        "xai_enabled": True,
        "structured_output_schema": Answer,
        "structured_output_options": {
            "inject_instructions": True,
        },
        "system_prompt": (
            "Use configured tools when requested and preserve " "exact numeric results."
        ),
        "max_reflections": 2,
        "strategy": StrategyEngine(),
        "compiler": compiler,
    }
    agent = create_agent(**primary_options)

    response = await agent.ainvoke(
        "Use multiply to calculate 12 times 8 and report the result."
    )
    assert tool_calls == [(12, 8)]
    print("ReAct:", verified_text(response, "96"))
    assert compiler.compilations == 1
    assert any(event.name == "ExecutionCompleted" for event in runtime_events)
    assert any(event.name == "ExecutionCompleted" for event in observer.events)

    selection = await agent.aroute_capabilities(
        "Check application status and use the multiplication capability."
    )
    assert "application_status" in selection.names
    assert "blocked_network" not in selection.names
    assert len(selection.names) <= 8
    print("Capability selection:", selection.names)

    mcp_sum = await agent.aexecute_capability(
        "mcp:complete-demo:tool:add",
        {"left": 19, "right": 23},
    )
    mcp_guide = await agent.aexecute_capability(
        "mcp:complete-demo:resource:agloom://guide"
    )
    mcp_prompt = await mcp_client.get_prompt(
        "demo",
        "summarize",
        arguments={"topic": "capability routing"},
    )
    assert "42" in str(mcp_sum)
    assert "local, application, and MCP" in str(mcp_guide)
    assert "capability routing" in str(mcp_prompt)
    print("MCP tool/resource/prompt: verified")

    manager = agent.capabilities.resolve("feedback_manager")
    feedback = await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.CORRECTION,
        target=FeedbackTarget(
            type=FeedbackTargetType.GENERATION,
            id="calculation-96",
        ),
        payload={"corrected_text": "The verified result is 96."},
    )
    feedback = await manager.acknowledge(feedback.feedback_id)
    feedback = await manager.mark_handled(feedback.feedback_id)
    feedback = await manager.resolve(
        feedback.feedback_id,
        resolution={"applied": True},
    )
    stored = await manager.query(FeedbackQuery(target_id="calculation-96"))
    assert feedback.status.value == "resolved"
    assert len(stored) == 1
    assert feedback_sink.events
    print("Feedback lifecycle: resolved")

    behavior = agent.capabilities.resolve("behavior_engine")
    first = behavior.process(
        BehaviorEvent.tool_call(
            "multiply",
            {"left": 12, "right": 8},
            scope="math",
        )
    )
    second = behavior.process(
        BehaviorEvent.tool_call(
            "multiply",
            {"left": 12, "right": 8},
            scope="math",
        )
    )
    assert first.intervention.kind is InterventionType.NOOP
    assert second.intervention.kind is InterventionType.PAUSE
    print("Behavior policy:", second.intervention.kind.value)

    refresh = agent.capabilities.resolve("refresh_engine")
    initial_refresh = await refresh.refresh()
    source.documents["shipping"] = "Shipping takes three days."
    changed_refresh = await refresh.refresh()
    await refresh.close()
    assert initial_refresh.added_count == 1
    assert changed_refresh.modified_count == 1
    assert indexed_documents["shipping"].endswith("three days.")
    print("Refresh: one modified document")

    structured = await agent.structured_output.ainvoke(
        "Use multiply to calculate 7 times 6. " "Return result=42 and verified=true."
    )
    assert structured.structured == Answer(result=42, verified=True)
    assert tool_calls[-1] == (7, 6)
    print("Structured output:", structured.structured)

    executions = await xai_executions(agent)
    assert executions
    assert all(execution.status.value == "completed" for execution in executions)
    print(f"xAI executions: {len(executions)} completed")

    checkpoint_options: dict[str, Any] = {
        "model": model,
        "tools": (),
        "pattern": "pipeline",
        "checkpointer": InMemorySaver(),
        "checkpoint_authorizer": (lambda thread_id: thread_id == "complete-example"),
        "interrupt_before": ["stage_0_prepare"],
        "interrupt_after": (),
        "stages": [
            PipelineStage(
                name="prepare",
                instruction="Prepare the answer 24.",
            )
        ],
        "xai_enabled": False,
    }
    checkpoint_agent = create_agent(**checkpoint_options)
    checkpoint_config: RunnableConfig = {
        "configurable": {"thread_id": "complete-example"}
    }
    paused = checkpoint_agent.invoke(
        "Return the number 24.",
        config=checkpoint_config,
    )
    assert paused.get("__interrupt__")
    resumed = checkpoint_agent.invoke(
        Command(resume=True),
        config=checkpoint_config,
    )
    print("Checkpoint resume:", verified_text(resumed, "24"))

    hybrid_options: dict[str, Any] = {
        "model": model,
        "tools": [multiply],
        "pattern": "hybrid",
        "composition": ["react", "reflection"],
        "max_reflections": 2,
        "system_prompt": "Use multiply when asked, then verify the result.",
    }
    hybrid = create_agent(**hybrid_options)
    hybrid_result = hybrid.invoke("Use multiply to calculate 5 times 5.")
    print("Hybrid:", verified_text(hybrid_result, "25"))

    custom_router_options: dict[str, Any] = {
        "model": model,
        "tools": (),
        "pattern": "react",
        "capability_router": PassthroughRouter(),
    }
    custom_router_agent = create_agent(**custom_router_options)
    assert isinstance(custom_router_agent.capability_router, PassthroughRouter)

    explicit_xai_options: dict[str, Any] = {
        "model": model,
        "tools": (),
        "pattern": "react",
        "explainability": xai_runtime(application_id="explicit-xai"),
    }
    explicit_xai_agent = create_agent(**explicit_xai_options)
    assert explicit_xai_agent.xai.application_id == "explicit-xai"

    covered = (
        set(primary_options)
        | set(checkpoint_options)
        | set(hybrid_options)
        | set(custom_router_options)
        | set(explicit_xai_options)
    )
    parameters = set(inspect.signature(create_agent).parameters)
    assert covered == parameters, (
        f"parameter coverage mismatch; missing={parameters - covered}, "
        f"unknown={covered - parameters}"
    )
    print("create_agent parameter coverage: " f"{len(parameters)}/{len(parameters)}")


asyncio.run(main())
