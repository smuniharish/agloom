"""The public construction API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from agloom.agent.agent import Agent, AgentSettings
from agloom.capabilities.providers import (
    application_descriptors,
    to_tool,
    tool_descriptors,
)
from agloom.capabilities.registry import CapabilityRegistry, InMemoryCapabilityRegistry
from agloom.capabilities.router import CapabilityRouter, DefaultCapabilityRouter
from agloom.compiler.compiler import ArchitectureCompiler
from agloom.errors import ConfigurationError
from agloom.loop.contracts import LoopController, LoopEvaluator, LoopObserver
from agloom.loop.controller import DefaultLoopController
from agloom.loop.evaluator import DefaultLoopEvaluator, ModelLoopEvaluator
from agloom.loop.models import LoopPolicy
from agloom.loop.observer import DefaultLoopObserver
from agloom.models import (
    ExecutionLimits,
    PipelineStage,
    Strategy,
    TopologyName,
    WorkerSpec,
    parse_composition,
    parse_pattern,
)
from agloom.observability.base import Observer, compose_observers
from agloom.observability.events import AgentEvent
from agloom.policy._arguments import InvalidArguments, may_be_string, policy_schema
from agloom.policy.manager import PolicyManager
from agloom.strategy.engine import StrategyEngine
from agloom.topology.supervisor import FINISH
from agloom.topology.swarm import STOP

if TYPE_CHECKING:
    from feedback_manager import FeedbackManager
    from langgraph_xai import XAIRuntime
    from mcp_capability_router import MCPRuntime
    from refresh_engine import RefreshEngine

_RESERVED_WORKER_NAMES = frozenset({FINISH, STOP})


def create_agent(
    *,
    model: BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any]] = (),
    system_prompt: str | None = None,
    pattern: str | TopologyName | Sequence[str | TopologyName] | None = None,
    composition: Sequence[str | TopologyName] | None = None,
    workers: Sequence[WorkerSpec] = (),
    stages: Sequence[PipelineStage] = (),
    max_reflections: int = 1,
    limits: ExecutionLimits | None = None,
    capabilities: Mapping[str, Any] | None = None,
    capability_registry: CapabilityRegistry | None = None,
    capability_router: CapabilityRouter | None = None,
    mcp: MCPRuntime | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
    policy: PolicyManager | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    checkpoint_authorizer: Callable[[str], bool] | None = None,
    interrupt_before: Sequence[str] = (),
    interrupt_after: Sequence[str] = (),
    loop_policy: LoopPolicy | None = None,
    loop_observer: LoopObserver | None = None,
    loop_evaluator: LoopEvaluator | None = None,
    loop_controller: LoopController | None = None,
    feedback: FeedbackManager | None = None,
    refresh: RefreshEngine | None = None,
    explainability: XAIRuntime | None = None,
    observers: Sequence[Observer | Callable[[AgentEvent], object]] = (),
    strategy: Strategy | None = None,
    compiler: ArchitectureCompiler | None = None,
) -> Agent:
    """Create an agent that selects DIRECT execution or one of eight topologies.

    Args:
        model: The LangChain chat model. Automatic selection and the routing
            topologies require structured-output support.
        tools: LangChain tools or annotated functions.
        system_prompt: Trusted instructions for every model call that executes
            a request; task analysis and loop evaluation use their own
            instructions.
        pattern: One topology (explicit, no analysis), a list of topologies
            (an allow-list for automatic selection), or `None` for automatic
            selection between DIRECT and all eight topologies.
        composition: Two to four child topologies for Hybrid; required when
            `pattern` names Hybrid.
        workers: Workers for Supervisor, Swarm, and Blackboard.
        stages: Stages for Pipeline.
        max_reflections: Maximum Reflection review rounds (1 to 5).
        limits: Recursion, parallelism, plan, and handoff limits.
        capabilities: Application capability objects by name.
        capability_registry: A custom capability catalog.
        capability_router: A custom router; defaults to `DefaultCapabilityRouter()`.
        mcp: An `mcp_capability_router.MCPRuntime` owned by the application.
        middleware: LangChain agent middleware applied to every tool-capable
            agent, such as ContextSage or BehaviorWeave middleware.
        policy: A [`PolicyManager`][agloom.PolicyManager] governing tool calls.
        checkpointer: A LangGraph checkpointer for durable, resumable runs.
        checkpoint_authorizer: Authorizes the caller's `thread_id`; required
            with a checkpointer.
        interrupt_before: Topology node names to pause before.
        interrupt_after: Topology node names to pause after.
        loop_policy: Loop Engineering bounds; one-shot by default.
        loop_observer: Custom loop observer.
        loop_evaluator: Custom loop evaluator. Multi-iteration policies
            without a custom observer or evaluator use a `ModelLoopEvaluator`.
        loop_controller: Custom loop controller.
        feedback: A `feedback_manager.FeedbackManager` whose human feedback on
            the thread applies to every later iteration.
        refresh: A `refresh_engine.RefreshEngine` used by the REFRESH action.
        explainability: A `langgraph_xai.XAIRuntime` instrumenting model and
            graph runs.
        observers: Observers or plain event callables.
        strategy: A custom execution strategy.
        compiler: A custom architecture compiler.

    Raises:
        ConfigurationError: If the configuration is invalid, for example when
            the policy cannot judge the arguments of a tool it governs.
        CompilationError: If an explicitly selected topology cannot be
            compiled, for example because a breakpoint names an unknown node.
    """
    if not isinstance(model, BaseChatModel):
        raise ConfigurationError("model must be a LangChain BaseChatModel")
    if not 1 <= max_reflections <= 5:
        raise ConfigurationError("max_reflections must be between 1 and 5")
    candidates = parse_pattern(pattern)
    parsed_composition = parse_composition(composition)
    if candidates is not None:
        if TopologyName.HYBRID in candidates and not parsed_composition:
            raise ConfigurationError(
                "the Hybrid topology requires an explicit composition"
            )
        if parsed_composition and TopologyName.HYBRID not in candidates:
            raise ConfigurationError(
                "composition is only valid when Hybrid can be selected"
            )
    resolved_workers = tuple(workers)
    resolved_stages = tuple(stages)
    _require_unique("worker", [worker.name for worker in resolved_workers])
    _require_unique("pipeline stage", [stage.name for stage in resolved_stages])
    reserved = sorted(
        {worker.name for worker in resolved_workers} & _RESERVED_WORKER_NAMES
    )
    if reserved:
        raise ConfigurationError(f"reserved worker names cannot be used: {reserved}")
    resolved_tools = tuple(to_tool(tool) for tool in tools)
    middleware_tools = tuple(
        tool for item in middleware for tool in getattr(item, "tools", ())
    )
    _require_unique(
        "tool", [tool.name for tool in (*resolved_tools, *middleware_tools)]
    )
    if policy is not None:
        configured = {
            *(tool.name for tool in (*resolved_tools, *middleware_tools)),
            *(capabilities or {}),
            *(capability_registry.names if capability_registry is not None else ()),
        }
        unknown = sorted(set(policy.consequential) - configured)
        if unknown and mcp is None:
            raise ConfigurationError(
                f"consequential tools are not configured: {unknown}"
            )
        resolved_tools = tuple(
            tool for tool in resolved_tools if policy.visible(tool.name)
        )
        middleware_tools = tuple(
            tool for tool in middleware_tools if policy.visible(tool.name)
        )
    registry = capability_registry or InMemoryCapabilityRegistry()
    for descriptor in (
        *tool_descriptors(resolved_tools),
        *tool_descriptors(middleware_tools, provider="middleware"),
        *application_descriptors(dict(capabilities or {})),
    ):
        registry.register(descriptor)
    if policy is not None:
        _check_governed_tools(
            policy,
            [
                descriptor.value
                for descriptor in registry.descriptors()
                if isinstance(descriptor.value, BaseTool)
            ],
        )
    if checkpointer is None and (interrupt_before or interrupt_after):
        raise ConfigurationError(
            "interrupt_before and interrupt_after require a checkpointer"
        )
    explicit = candidates is not None and len(candidates) == 1
    if (interrupt_before or interrupt_after) and not explicit:
        raise ConfigurationError(
            "interrupt_before and interrupt_after name topology nodes and require "
            "one explicit pattern"
        )
    if checkpointer is not None and checkpoint_authorizer is None:
        raise ConfigurationError(
            "a checkpointer requires a checkpoint_authorizer that validates thread IDs"
        )
    resolved_loop = loop_policy or LoopPolicy()
    if resolved_loop.allow_refresh and refresh is None:
        raise ConfigurationError(
            "LoopPolicy(allow_refresh=True) requires a refresh engine"
        )
    if resolved_loop.allow_topology_change and explicit:
        raise ConfigurationError(
            "LoopPolicy(allow_topology_change=True) requires automatic selection "
            "or an allow-list of topologies"
        )
    instrument = explainability.instrument if explainability is not None else None
    if loop_evaluator is None:
        loop_evaluator = (
            ModelLoopEvaluator(model, instrument=instrument)
            if resolved_loop.max_iterations > 1 and loop_observer is None
            else DefaultLoopEvaluator()
        )
    settings = AgentSettings(
        model=model,
        tools=resolved_tools,
        system_prompt=system_prompt.strip()
        if system_prompt and system_prompt.strip()
        else None,
        pattern=candidates,
        composition=parsed_composition,
        workers=resolved_workers,
        stages=resolved_stages,
        max_reflections=max_reflections,
        limits=limits or ExecutionLimits(),
        registry=registry,
        router=capability_router or DefaultCapabilityRouter(),
        mcp=mcp,
        middleware=tuple(middleware),
        policy=policy,
        checkpointer=checkpointer,
        checkpoint_authorizer=checkpoint_authorizer,
        interrupt_before=tuple(interrupt_before),
        interrupt_after=tuple(interrupt_after),
        loop_policy=resolved_loop,
        loop_observer=loop_observer or DefaultLoopObserver(),
        loop_evaluator=loop_evaluator,
        loop_controller=loop_controller or DefaultLoopController(),
        feedback=feedback,
        refresh=refresh,
        explainability=explainability,
        observers=compose_observers(observers),
        strategy=strategy or StrategyEngine(),
        compiler=compiler or ArchitectureCompiler(),
    )
    return Agent(settings)


def _require_unique(kind: str, names: Sequence[str]) -> None:
    if len(set(names)) != len(names):
        raise ConfigurationError(f"{kind} names must be unique")


def _check_governed_tools(policy: PolicyManager, tools: Sequence[BaseTool]) -> None:
    for tool in tools:
        try:
            schema = policy_schema(tool)
        except InvalidArguments as error:
            raise ConfigurationError(
                f"the policy cannot judge the arguments of {tool.name!r}: {error}; "
                "declare them with a Pydantic v2 model or a JSON schema"
            ) from error
        selector = policy.consequential.get(tool.name)
        if isinstance(selector, str) and not may_be_string(schema, selector):
            raise ConfigurationError(
                f"the idempotency key {selector!r} of {tool.name!r} must be a string "
                "argument of the tool"
            )
