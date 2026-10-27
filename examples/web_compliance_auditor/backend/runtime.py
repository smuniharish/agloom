"""Composition of the auditor: stores, MCP servers, policy, and the agent.

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
from behaviorweave import BehaviorEngine, InMemoryBehaviorStateStore, InterventionType
from behaviorweave import PolicyRule as BehaviorRule
from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware
from contextsage import IntelligentSummarizationMiddleware
from feedback_manager import FeedbackManager
from feedback_manager.contracts.store import FeedbackStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from mcp_capability_router import MCPRuntime, RefreshPolicy
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

from .auditor import SYSTEM_PROMPT, Auditor, AuditRecords
from .governance import RECORD_AUDIT, policy_manager, runtime_policy
from .mcp_servers import server_targets
from .observability import Telemetry, build_telemetry
from .policies import (
    IndexOperation,
    PolicyCatalog,
    PolicyFileSource,
    PolicyIndex,
    policy_search_tool,
)
from .settings import Settings, model_api_key
from .stores import (
    AuditLog,
    ComplianceRegister,
    PostgresFeedbackStore,
    PostgresRefreshStore,
    open_pool,
    table_counts,
)

BEHAVIOR_RULES = (
    BehaviorRule(
        policy_id="stop-repeated-policy-search",
        pattern_id="repeated_tool_call",
        threshold=3,
        intervention=InterventionType.STOP,
        message="The same policy search repeated three times; stop and report.",
    ),
)


@dataclass
class Backends:
    """The persistence the auditor uses."""

    policies: PolicyCatalog
    feedback: FeedbackStore
    refresh: StateStoreABC
    policy: RuntimeStore
    audits: AuditRecords
    register: Any
    counts: Callable[[], Mapping[str, int]]
    close: Callable[[], None]


@dataclass
class Runtime:
    """A running auditor and the components it owns."""

    settings: Settings
    telemetry: Telemetry
    backends: Backends
    mcp: MCPRuntime
    refresh: RefreshEngine
    policy: RuntimePolicy
    agent: Agent | None
    auditor: Auditor | None


class LocalToolsOnly(CapabilityFilter):
    """Offers the model only the local policy search tool.

    Audited pages are untrusted, so browsing and the other MCP surfaces stay
    with the application, which collects that evidence under the same policy.
    """

    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        """Admit local LangChain tools."""
        del task
        return capability.kind is CapabilityKind.TOOL


def chat_model(settings: Settings) -> ChatOpenAI | None:
    """Return the OpenAI-compatible model, or `None` without an API key."""
    api_key = model_api_key()
    if api_key is None:
        return None
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.model_base_url,
        api_key=api_key,
        timeout=90,
        max_retries=2,
        use_responses_api=False,
    )


def build_agent(
    *,
    model: BaseChatModel,
    backends: Backends,
    mcp: MCPRuntime,
    policy: RuntimePolicy,
    telemetry: Telemetry,
) -> Agent:
    """Create the auditor's ReAct agent."""
    return create_agent(
        model=model,
        tools=[policy_search_tool(backends.policies)],
        system_prompt=SYSTEM_PROMPT,
        pattern="react",
        limits=ExecutionLimits(max_depth=0),
        capabilities={RECORD_AUDIT: backends.register},
        capability_router=DefaultCapabilityRouter(capability_filter=LocalToolsOnly()),
        mcp=mcp,
        middleware=[
            IntelligentSummarizationMiddleware(
                model, trigger=("tokens", 100_000), keep=("messages", 8)
            ),
            BehaviorWeaveMiddleware(
                BehaviorEngine(
                    policies=BEHAVIOR_RULES, state_store=InMemoryBehaviorStateStore()
                ),
                on_decision=behavior_signals,
            ),
        ],
        policy=policy_manager(backends.policy, policy),
        observers=telemetry.observers,
    )


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    backends: Backends,
    *,
    model: BaseChatModel | None,
    mcp_targets: Mapping[str, Any],
    telemetry: Telemetry,
) -> AsyncIterator[Runtime]:
    """Start the MCP servers, refresh the policy index, and build the agent.

    The application can run without a model API key; audits then report that
    the model is not configured.
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
            MCPRuntime(max_concurrency=3, operation_timeout=30.0)
        )
        for server_id, target in mcp_targets.items():
            await mcp.register_mcp(
                server_id, target, refresh=RefreshPolicy(on_register=True)
            )
        refresh = RefreshEngine(
            PolicyFileSource(settings.policies_file),
            IndexOperation(backends.policies),
            store=backends.refresh,
        )
        stack.push_async_callback(refresh.close)
        await refresh.refresh()
        feedback = FeedbackManager(store=backends.feedback)
        stack.push_async_callback(feedback.aclose)
        policy = runtime_policy(approval_ttl_seconds=settings.approval_ttl_seconds)
        auditor: Auditor | None = None
        if model is not None:
            agent = build_agent(
                model=model,
                backends=backends,
                mcp=mcp,
                policy=policy,
                telemetry=telemetry,
            )
            stack.callback(agent.close)
            auditor = Auditor(
                agent=agent,
                model=model,
                policies=backends.policies,
                audits=backends.audits,
                feedback=feedback,
                policy_store=backends.policy,
                audit_counter=telemetry.audits,
                timeout=settings.audit_timeout,
            )
        yield Runtime(
            settings=settings,
            telemetry=telemetry,
            backends=backends,
            mcp=mcp,
            refresh=refresh,
            policy=policy,
            agent=agent,
            auditor=auditor,
        )


def postgres_backends(settings: Settings, embeddings: Embeddings) -> Backends:
    """Open the PostgreSQL stores and create their tables."""
    pool = open_pool(settings.database_url)
    try:
        policies = PolicyIndex(pool, embeddings)
        feedback = PostgresFeedbackStore(pool)
        refresh = PostgresRefreshStore(pool)
        audits = AuditLog(pool)
        register = ComplianceRegister(pool)
        for store in (policies, feedback, refresh, audits, register):
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
        policies=policies,
        feedback=feedback,
        refresh=refresh,
        policy=policy_store,
        audits=audits,
        register=register,
        counts=lambda: table_counts(pool),
        close=close,
    )


@asynccontextmanager
async def production_runtime() -> AsyncIterator[Runtime]:
    """Run the auditor against PostgreSQL, Ollama, and the npm MCP servers."""
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
            mcp_targets=server_targets(settings),
            telemetry=telemetry,
        ) as runtime:
            yield runtime
    finally:
        backends.close()
