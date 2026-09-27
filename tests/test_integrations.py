from __future__ import annotations

import itertools
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from agloom import (
    AgentEvent,
    ConfigurationError,
    DefaultCapabilityRouter,
    create_agent,
)
from agloom.capabilities.integrations import (
    BehaviorWeaveAdapter,
    behavior_engine,
    context_middleware,
    feedback_manager,
    mcp_runtime,
    refresh_engine,
    with_structured_output,
    xai_runtime,
)

DIRECT_ANALYSIS = '{"complexity":0,"estimated_steps":1}'


def test_required_package_constructors_use_installed_public_apis() -> None:
    from behaviorweave import BehaviorEngine
    from contextsage import IntelligentSummarizationMiddleware
    from feedback_manager import FeedbackManager
    from langgraph_xai import XAIRuntime
    from mcp_capability_router import MCPRuntime
    from refresh_engine import RefreshEngine

    assert isinstance(feedback_manager(), FeedbackManager)
    assert isinstance(behavior_engine(), BehaviorEngine)
    assert isinstance(
        context_middleware(FakeListChatModel(responses=["unused"])),
        IntelligentSummarizationMiddleware,
    )
    assert isinstance(mcp_runtime(), MCPRuntime)
    assert isinstance(
        refresh_engine(object(), lambda *args: None),
        RefreshEngine,
    )
    assert isinstance(xai_runtime(application_id="agloom-tests"), XAIRuntime)


def test_behaviorweave_adapter_translates_normalized_completion_event() -> None:
    from behaviorweave import BehaviorEngine

    engine = BehaviorEngine()
    adapter = BehaviorWeaveAdapter(engine, scope="agloom-test-agent")

    decision = adapter(
        AgentEvent(
            name="ExecutionCompleted",
            agent_id="agent-test",
            run_id="run-test",
        )
    )

    assert decision.intervention.kind.value == "noop"


def test_contextsage_middleware_is_wired_into_langchain_react() -> None:
    model = FakeListChatModel(responses=["A focused answer."])
    middleware = context_middleware(
        model,
        trigger=("tokens", 100_000),
        keep=("messages", 5),
    )
    agent = create_agent(
        model=model,
        tools=[],
        pattern="react",
        middleware=[middleware],
    )

    result = agent.invoke("Answer briefly.")

    assert isinstance(result, AIMessage)
    assert result.content == "A focused answer."


def test_xai_runtime_instruments_topology_compilation() -> None:
    agent = create_agent(
        model=FakeListChatModel(responses=["First", "Second", "Final"]),
        tools=[],
        pattern="pipeline",
        explainability=xai_runtime(application_id="agloom-tests"),
    )

    result = agent.invoke("Run the stages.")

    assert result.content == "Final"


def test_xai_runtime_is_automatic_and_can_be_disabled() -> None:
    from langgraph_xai import XAIRuntime

    automatic = create_agent(
        model=FakeListChatModel(responses=["answer"]),
        tools=[],
        pattern="react",
    )
    disabled = create_agent(
        model=FakeListChatModel(responses=["answer"]),
        tools=[],
        pattern="react",
        xai_enabled=False,
    )

    assert isinstance(automatic.capabilities.resolve("xai_runtime"), XAIRuntime)
    assert automatic.xai is automatic.capabilities.resolve("xai_runtime")
    assert disabled.config.explainability is None
    assert disabled.xai is None
    assert "xai_runtime" not in disabled.capabilities.names


class Summary(BaseModel):
    title: str
    complete: bool


def test_xstructured_wraps_an_agloom_runnable() -> None:
    agent = create_agent(
        model=FakeListChatModel(
            responses=[
                DIRECT_ANALYSIS,
                '<xstructured>{"title":"Report","complete":true}</xstructured>',
            ]
        ),
        tools=[],
    )
    structured = with_structured_output(agent, Summary)

    result = structured.invoke("Return a short report.")

    assert result.structured == Summary(title="Report", complete=True)


@pytest.mark.asyncio
async def test_feedback_manager_submission_uses_real_domain_contract() -> None:
    from feedback_manager import (
        FeedbackCategory,
        FeedbackSource,
        FeedbackTarget,
        FeedbackTargetType,
    )

    manager = feedback_manager()
    result = await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.CORRECTION,
        target=FeedbackTarget(
            type=FeedbackTargetType.GENERATION,
            id="generation-1",
        ),
        payload={"corrected_text": "Use the verified result."},
    )

    assert result.target.id == "generation-1"


def test_create_agent_accepts_upstream_constructor_overrides() -> None:
    from behaviorweave import BehaviorEngine
    from contextsage import IntelligentSummarizationMiddleware
    from feedback_manager import FeedbackManager
    from langgraph_xai import XAIRuntime
    from mcp_capability_router import MCPRuntime

    class Sink:
        def emit(self, event) -> None:
            pass

    model = FakeListChatModel(responses=["A focused answer."])
    agent = create_agent(
        model=model,
        tools=[],
        pattern="react",
        feedback_options={"observability_sink": Sink()},
        behavior_options={"policies": (), "patterns": ()},
        context_options={
            "policy": "maximum_compression",
            "maximum_context_tokens": 12_000,
            "safety_margin": 0.2,
        },
        mcp_options={"max_concurrency": 3, "operation_timeout": 1.5},
        xai_options={
            "application_id": "override-test",
            "tenant_id": "tenant-test",
        },
    )

    assert isinstance(agent.capabilities.resolve("feedback_manager"), FeedbackManager)
    assert isinstance(agent.capabilities.resolve("behavior_engine"), BehaviorEngine)
    assert isinstance(
        agent.capabilities.resolve("context_middleware"),
        IntelligentSummarizationMiddleware,
    )
    assert isinstance(agent.capabilities.resolve("mcp_runtime"), MCPRuntime)
    assert isinstance(agent.capabilities.resolve("xai_runtime"), XAIRuntime)


INTEGRATION_COMBINATIONS = tuple(itertools.product((False, True), repeat=7))


@pytest.mark.parametrize(
    (
        "feedback_enabled",
        "behavior_enabled",
        "context_enabled",
        "mcp_enabled",
        "refresh_enabled",
        "xai_enabled",
        "structured_enabled",
    ),
    INTEGRATION_COMBINATIONS,
)
def test_every_supported_integration_combination_constructs(
    feedback_enabled,
    behavior_enabled,
    context_enabled,
    mcp_enabled,
    refresh_enabled,
    xai_enabled,
    structured_enabled,
) -> None:
    class Client:
        async def get_tools(self, *, server_name=None):
            del server_name
            return []

        async def get_resources(self, server_name=None, *, uris=None):
            del server_name, uris
            return []

        async def get_prompt(self, server_name, prompt_name, *, arguments=None):
            del server_name, prompt_name, arguments
            return None

    options: dict[str, Any] = {
        "model": FakeListChatModel(responses=["unused"]),
        "tools": [],
        "pattern": "react",
        "feedback_options": {} if feedback_enabled else None,
        "behavior_options": {} if behavior_enabled else None,
        "context_options": (
            {"trigger": ("tokens", 100_000), "keep": ("messages", 5)}
            if context_enabled
            else None
        ),
        "mcp_options": {} if mcp_enabled else None,
        "mcp_client": Client() if mcp_enabled else None,
        "mcp_server_name": "matrix" if mcp_enabled else None,
        "refresh_source": (object() if refresh_enabled else None),
        "refresh_operation": ((lambda *args: None) if refresh_enabled else None),
        "xai_enabled": xai_enabled,
        "structured_output_schema": Summary if structured_enabled else None,
        "structured_output_options": (
            {"inject_instructions": False} if structured_enabled else None
        ),
    }

    agent = create_agent(**options)

    expected = {
        "feedback_manager": feedback_enabled,
        "behavior_engine": behavior_enabled,
        "context_middleware": context_enabled,
        "mcp_runtime": mcp_enabled,
        "refresh_engine": refresh_enabled,
        "xai_runtime": xai_enabled,
        "structured_output": structured_enabled,
    }
    for name, enabled in expected.items():
        assert (name in agent.capabilities.names) is enabled
    assert (agent.structured_output is not None) is structured_enabled
    assert (agent.xai is not None) is xai_enabled


@pytest.mark.parametrize(
    "conflicting_options",
    [
        {
            "capability_router": DefaultCapabilityRouter(),
            "capability_policy": object(),
        },
        {
            "capability_router": DefaultCapabilityRouter(),
            "capability_embeddings": object(),
        },
        {
            "capability_router": DefaultCapabilityRouter(),
            "capability_retriever": object(),
        },
        {
            "capability_router": DefaultCapabilityRouter(),
            "capability_reranker": object(),
        },
        {
            "capability_router": DefaultCapabilityRouter(),
            "max_selected_capabilities": 2,
        },
    ],
)
def test_custom_router_rejects_every_component_override(
    conflicting_options,
) -> None:
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        create_agent(
            model=FakeListChatModel(responses=["unused"]),
            **conflicting_options,
        )


@pytest.mark.parametrize(
    "conflicting_options",
    [
        {
            "xai_enabled": False,
            "explainability": object(),
        },
        {
            "xai_enabled": False,
            "xai_options": {"application_id": "invalid"},
        },
        {
            "explainability": object(),
            "xai_options": {"application_id": "invalid"},
        },
    ],
)
def test_xai_override_modes_are_mutually_exclusive(
    conflicting_options,
) -> None:
    with pytest.raises(ConfigurationError):
        create_agent(
            model=FakeListChatModel(responses=["unused"]),
            **conflicting_options,
        )


def test_structured_options_require_a_schema() -> None:
    with pytest.raises(ConfigurationError, match="schema is required"):
        create_agent(
            model=FakeListChatModel(responses=["unused"]),
            structured_output_options={"inject_instructions": False},
        )


@pytest.mark.asyncio
async def test_create_agent_configures_refresh_mcp_client_and_structured_output() -> (
    None
):
    from mcp_capability_router import MCPRuntime
    from refresh_engine import RefreshEngine

    class Client:
        async def get_tools(self, *, server_name: str | None = None) -> list[object]:
            return []

        async def get_resources(self, server_name=None, *, uris=None) -> list[object]:
            return []

        async def get_prompt(self, server_name, prompt_name, *, arguments=None):
            return None

    response = '<xstructured>{"title":"Report","complete":true}</xstructured>'
    agent = create_agent(
        model=FakeListChatModel(
            responses=[
                DIRECT_ANALYSIS,
                response,
                DIRECT_ANALYSIS,
                response,
            ]
        ),
        tools=[],
        mcp_client=Client(),
        mcp_server_name="documents",
        mcp_server_id="document-router",
        mcp_client_server_name="remote-documents",
        mcp_discover_resources=True,
        mcp_metadata={"owner": "tests"},
        refresh_source=object(),
        refresh_operation=lambda *args: None,
        structured_output_schema=Summary,
        structured_output_options={"inject_instructions": False},
    )

    assert isinstance(agent.capabilities.resolve("mcp_runtime"), MCPRuntime)
    assert isinstance(agent.capabilities.resolve("refresh_engine"), RefreshEngine)
    assert agent.structured_output is agent.capabilities.resolve("structured_output")
    await agent.ainvoke("Initialize MCP and respond.")
    assert await agent.capabilities.resolve("mcp_runtime").health("document-router")
    assert agent.structured_output.invoke("Return a report.").structured == Summary(
        title="Report",
        complete=True,
    )


@pytest.mark.asyncio
async def test_automatic_xai_captures_direct_and_topology_executions() -> None:
    from langgraph_xai import Execution, ProvenanceStore, StoreFilter

    direct = create_agent(
        model=FakeListChatModel(responses=[DIRECT_ANALYSIS, "direct answer"]),
        tools=[],
    )
    topology = create_agent(
        model=FakeListChatModel(responses=["topology answer"]),
        tools=[],
        pattern="react",
    )

    assert direct.invoke("Answer directly.").content == "direct answer"
    assert topology.invoke("Use the topology.").content == "topology answer"

    async def executions(agent):
        runtime = agent.capabilities.resolve("xai_runtime")
        store = runtime.registry.require(ProvenanceStore)
        return [
            item
            async for item in store.query(
                StoreFilter(
                    application_id="agloom",
                    tenant_id="default",
                    item_type=Execution,
                )
            )
        ]

    direct_executions = await executions(direct)
    topology_executions = await executions(topology)

    assert len(direct_executions) == 2
    assert all(item.status.value == "completed" for item in direct_executions)
    assert len(topology_executions) == 1
    assert topology_executions[0].status.value == "completed"


def test_create_agent_rejects_incomplete_package_configuration() -> None:
    model = FakeListChatModel(responses=["unused"])
    with pytest.raises(ConfigurationError, match="must be provided together"):
        create_agent(
            model=model,
            refresh_source=object(),
        )
    with pytest.raises(ConfigurationError, match="require mcp_client"):
        create_agent(
            model=model,
            mcp_options={"max_concurrency": 4},
            mcp_server_name="unconfigured",
        )
