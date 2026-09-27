"""Public construction API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agloom.agent.handle import Agent
from agloom.capabilities.integrations import (
    behavior_engine,
    with_structured_output,
)
from agloom.capabilities.integrations import (
    context_middleware as make_context_middleware,
)
from agloom.capabilities.integrations import (
    feedback_manager as make_feedback_manager,
)
from agloom.capabilities.integrations import (
    mcp_refresh_policy as make_mcp_refresh_policy,
)
from agloom.capabilities.integrations import (
    mcp_runtime as make_mcp_runtime,
)
from agloom.capabilities.integrations import (
    refresh_engine as make_refresh_engine,
)
from agloom.capabilities.integrations import (
    xai_runtime as make_xai_runtime,
)
from agloom.capabilities.registry import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRegistry,
)
from agloom.capabilities.router import (
    CapabilityPolicy,
    CapabilityRouter,
    DefaultCapabilityRouter,
)
from agloom.compiler.compiler import ArchitectureCompiler
from agloom.errors import ConfigurationError
from agloom.models import (
    AgentConfig,
    PipelineStage,
    RecursionPolicy,
    Strategy,
    TopologyName,
    WorkerSpec,
    normalize_composition,
    parse_pattern,
)
from agloom.observability import (
    CallableObserver,
    CompositeObserver,
    LangfuseObserver,
    Observer,
)
from agloom.strategy.engine import StrategyEngine


def create_agent(
    *,
    model: BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any]] = (),
    pattern: str | TopologyName | Sequence[str | TopologyName] | None = None,
    recursion: bool = False,
    recursion_policy: RecursionPolicy | None = None,
    composition: Sequence[str | TopologyName] | None = None,
    workers: Sequence[WorkerSpec] = (),
    stages: Sequence[PipelineStage] = (),
    checkpointer: BaseCheckpointSaver | None = None,
    checkpoint_authorizer: Callable[[str], bool] | None = None,
    interrupt_before: Sequence[str] = (),
    interrupt_after: Sequence[str] = (),
    event_sink: Callable[[Any], None] | None = None,
    observers: Sequence[Observer] = (),
    capabilities: Mapping[str, Any] | None = None,
    capability_registry: CapabilityRegistry | None = None,
    capability_router: CapabilityRouter | None = None,
    capability_policy: CapabilityPolicy | None = None,
    capability_embeddings: Embeddings | None = None,
    capability_retriever: BaseRetriever | None = None,
    capability_reranker: BaseDocumentCompressor | None = None,
    max_selected_capabilities: int = 20,
    middleware: Sequence[Any] = (),
    explainability: Any = None,
    feedback_options: Mapping[str, Any] | None = None,
    behavior_options: Mapping[str, Any] | None = None,
    context_model: BaseChatModel | str | None = None,
    context_options: Mapping[str, Any] | None = None,
    mcp_options: Mapping[str, Any] | None = None,
    mcp_client: Any = None,
    mcp_server_name: str | None = None,
    mcp_server_id: str | None = None,
    mcp_client_server_name: str | None = None,
    mcp_discover_resources: bool = False,
    mcp_metadata: Mapping[str, Any] | None = None,
    mcp_refresh: Any = None,
    refresh_source: Any = None,
    refresh_operation: Any = None,
    refresh_options: Mapping[str, Any] | None = None,
    xai_options: Mapping[str, Any] | None = None,
    xai_enabled: bool = True,
    structured_output_schema: Any = None,
    structured_output_options: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    max_reflections: int = 1,
    strategy: Strategy | None = None,
    compiler: ArchitectureCompiler | None = None,
) -> Agent:
    """Create an isolated, LangChain-runnable agent.

    A single explicit topology is compiled without running the analyzer.
    Multiple topology candidates are an allow-list; selection happens per task.
    Integration option mappings are passed directly to their upstream
    constructors. The xstructured wrapper, when configured, is exposed as
    ``agent.structured_output``; MCP client setup is performed asynchronously.
    """

    candidates = parse_pattern(pattern)
    if capability_router is not None and (
        capability_policy is not None
        or capability_embeddings is not None
        or capability_retriever is not None
        or capability_reranker is not None
        or max_selected_capabilities != 20
    ):
        raise ConfigurationError(
            "capability_router cannot be combined with router component options"
        )
    resolved_router = capability_router or DefaultCapabilityRouter(
        policy=capability_policy,
        embeddings=capability_embeddings,
        retriever=capability_retriever,
        reranker=capability_reranker,
        max_results=max_selected_capabilities,
    )
    resolved_observers = list(observers)
    if event_sink is not None:
        resolved_observers.append(CallableObserver(event_sink))
    observability = CompositeObserver(resolved_observers)
    normalized_composition = normalize_composition(composition)
    if candidates and TopologyName.HYBRID in candidates and not normalized_composition:
        raise ConfigurationError(
            "Hybrid requires an explicit composition of at least two topologies"
        )
    if normalized_composition and (
        len(normalized_composition) < 2
        or TopologyName.HYBRID in normalized_composition
        or len(normalized_composition) > 4
    ):
        raise ConfigurationError(
            "Hybrid composition requires two to four non-Hybrid topologies"
        )

    resolved_capabilities = dict(capabilities or {})

    def add_capability(name: str, value: Any) -> None:
        if name in resolved_capabilities:
            raise ConfigurationError(
                f"capability {name!r} is configured more than once"
            )
        resolved_capabilities[name] = value

    resolved_middleware = list(middleware)
    if context_model is not None and context_options is None:
        raise ConfigurationError("context_model requires context_options")
    if feedback_options is not None:
        add_capability("feedback_manager", make_feedback_manager(**feedback_options))
    if behavior_options is not None:
        add_capability("behavior_engine", behavior_engine(**behavior_options))
    if context_options is not None:
        context = make_context_middleware(
            context_model if context_model is not None else model,
            **context_options,
        )
        add_capability("context_middleware", context)
        resolved_middleware.append(context)
    if mcp_client is None and (
        mcp_server_name is not None
        or mcp_server_id is not None
        or mcp_client_server_name is not None
        or mcp_discover_resources
        or mcp_metadata is not None
        or mcp_refresh is not None
    ):
        raise ConfigurationError("MCP client registration options require mcp_client")
    if mcp_options is not None or mcp_client is not None:
        runtime = make_mcp_runtime(**(mcp_options or {}))
        add_capability("mcp_runtime", runtime)
        if mcp_client is not None and (
            not mcp_server_name or not mcp_server_name.strip()
        ):
            raise ConfigurationError(
                "mcp_server_name is required when mcp_client is supplied"
            )
    if (refresh_source is None) != (refresh_operation is None):
        raise ConfigurationError(
            "refresh_source and refresh_operation must be provided together"
        )
    if refresh_options is not None and refresh_source is None:
        raise ConfigurationError(
            "refresh_options require refresh_source and refresh_operation"
        )
    if refresh_source is not None and refresh_operation is not None:
        add_capability(
            "refresh_engine",
            make_refresh_engine(
                refresh_source,
                refresh_operation,
                **(refresh_options or {}),
            ),
        )
    if not xai_enabled and (explainability is not None or xai_options is not None):
        raise ConfigurationError(
            "xai_enabled=False cannot be combined with explainability or xai_options"
        )
    if explainability is not None and xai_options is not None:
        raise ConfigurationError(
            "provide either explainability or xai_options, not both"
        )
    resolved_explainability = None
    if xai_enabled:
        resolved_xai_options = {"application_id": "agloom", **(xai_options or {})}
        resolved_explainability = explainability or make_xai_runtime(
            **resolved_xai_options
        )
        langfuse_observers = tuple(
            observer
            for observer in resolved_observers
            if isinstance(observer, LangfuseObserver)
        )
        if len(langfuse_observers) > 1:
            raise ConfigurationError(
                "only one LangfuseObserver can back the default xAI runtime"
            )
        if langfuse_observers:
            from langgraph_xai import ObservabilityProvider

            resolved_explainability.register(
                ObservabilityProvider,
                langfuse_observers[0].xai_observability(),
            )
    if resolved_explainability is not None:
        add_capability("xai_runtime", resolved_explainability)
    if (
        structured_output_schema is not None
        and "structured_output" in resolved_capabilities
    ):
        raise ConfigurationError(
            "capability 'structured_output' is reserved when a schema is configured"
        )

    config = AgentConfig(
        model=model,
        tools=tuple(tools),
        pattern_candidates=candidates,
        recursion=recursion,
        recursion_policy=recursion_policy or RecursionPolicy(),
        composition=normalized_composition,
        workers=tuple(workers),
        stages=tuple(stages),
        checkpointer=checkpointer,
        checkpoint_authorizer=checkpoint_authorizer,
        interrupt_before=tuple(interrupt_before),
        interrupt_after=tuple(interrupt_after),
        event_sink=event_sink,
        capabilities=resolved_capabilities,
        middleware=tuple(resolved_middleware),
        explainability=resolved_explainability,
        system_prompt=system_prompt,
        max_reflections=max_reflections,
    )
    mcp_registration = None
    if mcp_client is not None:
        mcp_registration = {
            "server_id": mcp_server_id or mcp_server_name,
            "client": mcp_client,
            "client_server_name": mcp_client_server_name,
            "discover_resources": mcp_discover_resources,
            "metadata": dict(mcp_metadata) if mcp_metadata is not None else None,
            "refresh": (
                mcp_refresh
                if mcp_refresh is not None
                else make_mcp_refresh_policy(on_register=True)
            ),
        }
    agent = Agent(
        config=config,
        strategy=strategy or StrategyEngine(),
        compiler=compiler or ArchitectureCompiler(),
        observability=observability,
        capability_registry=capability_registry,
        capability_router=resolved_router,
        mcp_server_name=mcp_server_name,
        mcp_registration=mcp_registration,
    )
    if structured_output_schema is not None:
        structured = with_structured_output(
            agent,
            structured_output_schema,
            **(structured_output_options or {}),
        )
        agent.capabilities.register(
            CapabilityDescriptor(
                name="structured_output",
                description="Validated structured output runnable.",
                kind=CapabilityKind.APPLICATION,
                value=structured,
                provider="xstructured",
            )
        )
        agent.structured_output = structured
    elif structured_output_options is not None:
        raise ConfigurationError(
            "structured_output_schema is required with structured_output_options"
        )
    return agent
