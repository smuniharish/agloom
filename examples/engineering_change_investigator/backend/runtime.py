"""Composition of the investigator: stores, MCP servers, policy, and the agent.

`open_runtime` owns everything it starts and stops it in reverse order. The
production runtime persists in PostgreSQL; tests compose the same runtime with
in-memory stores and in-process MCP servers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from agentbarrier.runtime import PostgresRuntimeStore, RuntimePolicy, RuntimeStore
from behaviorweave import BehaviorEngine, BehaviorStateStore, InterventionType
from behaviorweave import PolicyRule as BehaviorRule
from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware
from contextsage import IntelligentSummarizationMiddleware
from feedback_manager import FeedbackManager
from feedback_manager.contracts.store import FeedbackStore
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph_xai import ProvenanceStore, Registry, XAIRuntime
from mcp_capability_router import CapabilityRegistry, MCPRuntime, RefreshPolicy
from refresh_engine import RefreshEngine, StateStoreABC

from agloom import (
    Agent,
    CapabilityDescriptor,
    CapabilityFilter,
    CapabilityKind,
    DefaultCapabilityRouter,
    ExecutionLimits,
    behavior_signals,
    create_agent,
)

from .evidence import (
    EvidenceIndex,
    EvidenceReranker,
    EvidenceRetriever,
    EvidenceStore,
    IndexOperation,
    WorkspaceSource,
    evidence_search_tool,
)
from .governance import RELEASE_DECISION, policy_manager, runtime_policy
from .investigator import SYSTEM_PROMPT, InvestigationRecords, Investigator
from .mcp_servers import FILESYSTEM, server_targets
from .observability import Telemetry, build_telemetry
from .settings import APPLICATION_ID, TENANT_ID, Settings, model_api_key
from .stores import (
    InvestigationLog,
    PostgresBehaviorStateStore,
    PostgresCapabilityRegistry,
    PostgresFeedbackStore,
    PostgresProvenanceStore,
    PostgresRefreshStore,
    ReleaseRegister,
    open_pool,
    table_counts,
)

BEHAVIOR_RULES = (
    BehaviorRule(
        policy_id="stop-repeated-investigation",
        pattern_id="repeated_tool_call",
        threshold=3,
        intervention=InterventionType.STOP,
        message="The same evidence lookup repeated three times; stop and report.",
    ),
)


@dataclass
class Backends:
    """The persistence the investigator uses."""

    evidence: EvidenceStore
    feedback: FeedbackStore
    behavior: BehaviorStateStore
    mcp_registry: CapabilityRegistry
    refresh: StateStoreABC
    provenance: ProvenanceStore
    policy: RuntimeStore
    investigations: InvestigationRecords
    release_register: Any
    counts: Callable[[], Mapping[str, int]]
    close: Callable[[], None]


@dataclass
class Runtime:
    """A running investigator and the components it owns."""

    settings: Settings
    telemetry: Telemetry
    backends: Backends
    mcp: MCPRuntime
    refresh: RefreshEngine
    policy: RuntimePolicy
    agent: Agent | None
    investigator: Investigator | None


class ModelEvidenceFilter(CapabilityFilter):
    """Offers the model local tools and the read-only workspace server.

    Browser and reference MCP surfaces stay with the application, which
    collects that evidence itself under the same policy.
    """

    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        """Admit local tools and filesystem MCP tools."""
        del task
        if capability.kind is CapabilityKind.TOOL:
            return True
        return (
            capability.kind is CapabilityKind.MCP
            and capability.model_callable
            and capability.metadata.get("mcp_server_id") == FILESYSTEM
        )


def chat_model(settings: Settings) -> ChatOpenAI | None:
    """Return the OpenAI-compatible model, or `None` without an API key."""
    api_key = model_api_key()
    if api_key is None:
        return None
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.model_base_url,
        api_key=api_key,
        timeout=120,
        max_retries=2,
        use_responses_api=False,
    )


def build_agent(
    *,
    model: BaseChatModel,
    embeddings: Embeddings,
    backends: Backends,
    mcp: MCPRuntime,
    policy: RuntimePolicy,
    explainability: XAIRuntime,
    telemetry: Telemetry,
) -> Agent:
    """Create the investigator's ReAct agent."""
    return create_agent(
        model=model,
        tools=[evidence_search_tool(backends.evidence)],
        system_prompt=SYSTEM_PROMPT,
        pattern="react",
        limits=ExecutionLimits(max_depth=0),
        capabilities={RELEASE_DECISION: backends.release_register},
        capability_router=DefaultCapabilityRouter(
            capability_filter=ModelEvidenceFilter(),
            embeddings=embeddings,
            retriever=EvidenceRetriever(index=backends.evidence),
            reranker=EvidenceReranker(),
            max_results=12,
        ),
        mcp=mcp,
        middleware=[
            IntelligentSummarizationMiddleware(
                model, trigger=("tokens", 16_000), keep=("messages", 12)
            ),
            BehaviorWeaveMiddleware(
                BehaviorEngine(policies=BEHAVIOR_RULES, state_store=backends.behavior),
                on_decision=behavior_signals,
            ),
            ModelCallLimitMiddleware(run_limit=8),
        ],
        policy=policy_manager(backends.policy, policy),
        explainability=explainability,
        observers=telemetry.observers,
    )


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    backends: Backends,
    *,
    model: BaseChatModel | None,
    embeddings: Embeddings,
    mcp_targets: Mapping[str, Any],
    telemetry: Telemetry,
) -> AsyncIterator[Runtime]:
    """Start the MCP servers, refresh the evidence index, and build the agent.

    The application can run without a model API key; investigations then
    report that the model is not configured.
    """
    async with AsyncExitStack() as stack:
        telemetry.start()
        agent: Agent | None = None

        def stop_telemetry() -> None:
            # The agent shuts the observers down itself once it exists.
            if agent is None:
                telemetry.shutdown()

        stack.callback(stop_telemetry)
        mcp = await stack.enter_async_context(
            MCPRuntime(
                registry=backends.mcp_registry,
                max_concurrency=6,
                operation_timeout=60.0,
            )
        )
        for server_id, target in mcp_targets.items():
            await mcp.register_mcp(
                server_id, target, refresh=RefreshPolicy(on_register=True)
            )
        refresh = RefreshEngine(
            WorkspaceSource(settings.workspace),
            IndexOperation(backends.evidence),
            store=backends.refresh,
        )
        stack.push_async_callback(refresh.close)
        await refresh.refresh()
        registry = Registry()
        registry.register(ProvenanceStore, backends.provenance)
        explainability = XAIRuntime(
            registry=registry,
            application_id=APPLICATION_ID,
            tenant_id=TENANT_ID,
            graph_id="engineering-investigation",
        )
        stack.push_async_callback(explainability.close)
        feedback = FeedbackManager(store=backends.feedback)
        stack.push_async_callback(feedback.aclose)
        policy = runtime_policy(
            target_site_url=settings.target_site_url,
            approval_ttl_seconds=settings.approval_ttl_seconds,
        )
        investigator: Investigator | None = None
        if model is not None:
            agent = build_agent(
                model=model,
                embeddings=embeddings,
                backends=backends,
                mcp=mcp,
                policy=policy,
                explainability=explainability,
                telemetry=telemetry,
            )
            stack.callback(agent.close)
            investigator = Investigator(
                agent=agent,
                model=model,
                evidence=backends.evidence,
                investigations=backends.investigations,
                feedback=feedback,
                policy_store=backends.policy,
                workspace=settings.workspace,
                target_site_url=settings.target_site_url,
                timeout=settings.investigation_timeout,
            )
        yield Runtime(
            settings=settings,
            telemetry=telemetry,
            backends=backends,
            mcp=mcp,
            refresh=refresh,
            policy=policy,
            agent=agent,
            investigator=investigator,
        )


def postgres_backends(settings: Settings, embeddings: Embeddings) -> Backends:
    """Open the PostgreSQL stores and create their tables."""
    pool = open_pool(settings.database_url)
    try:
        evidence = EvidenceIndex(pool, embeddings)
        feedback = PostgresFeedbackStore(pool)
        behavior = PostgresBehaviorStateStore(pool)
        registry = PostgresCapabilityRegistry(pool)
        refresh = PostgresRefreshStore(pool)
        provenance = PostgresProvenanceStore(pool)
        investigations = InvestigationLog(pool)
        register = ReleaseRegister(pool)
        for store in (
            evidence,
            feedback,
            behavior,
            registry,
            refresh,
            provenance,
            investigations,
            register,
        ):
            store.initialize()
        policy_store = PostgresRuntimeStore(
            settings.database_url,
            schema="agentbarrier",
            create_schema=True,
            migrate=True,
        )
    except BaseException:
        pool.close()
        raise

    def close() -> None:
        try:
            policy_store.close()
        finally:
            pool.close()

    return Backends(
        evidence=evidence,
        feedback=feedback,
        behavior=behavior,
        mcp_registry=registry,
        refresh=refresh,
        provenance=provenance,
        policy=policy_store,
        investigations=investigations,
        release_register=register,
        counts=lambda: table_counts(pool),
        close=close,
    )


@asynccontextmanager
async def production_runtime() -> AsyncIterator[Runtime]:
    """Run the investigator against PostgreSQL, Ollama, and the npm MCP servers."""
    settings = Settings.from_environment()
    telemetry = build_telemetry(settings)
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model, base_url=settings.ollama_url
    )
    backends = await asyncio.to_thread(postgres_backends, settings, embeddings)
    try:
        async with open_runtime(
            settings,
            backends,
            model=chat_model(settings),
            embeddings=embeddings,
            mcp_targets=server_targets(settings),
            telemetry=telemetry,
        ) as runtime:
            yield runtime
    finally:
        backends.close()
