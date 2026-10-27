"""The Agloom agent: a LangChain Runnable over Loop Engineering and eight topologies."""

from __future__ import annotations

import asyncio
import inspect
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from agentbarrier.errors import RuntimeBarrierError
from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    ensure_config,
)
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphDrained, GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, GraphOutput, Interrupt, StateSnapshot
from pydantic import ValidationError
from typing_extensions import TypedDict

from agloom._runtime import (
    RunContext,
    configured_run,
    scoped,
    with_run,
    write_stream,
)
from agloom.agent._inputs import answer_message, normalize_input, shape_result
from agloom.agent._recursion import RecursionSession
from agloom.analysis.analyzer import TaskAnalyzer
from agloom.capabilities._middleware import ToolRoutingMiddleware
from agloom.capabilities.providers import MCPCapability, mcp_descriptors
from agloom.capabilities.registry import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRegistry,
)
from agloom.capabilities.router import CapabilityRouter, CapabilitySelection
from agloom.compiler.compiler import ArchitectureCompiler
from agloom.errors import (
    AgloomError,
    CancellationError,
    CapabilityResolutionError,
    ConfigurationError,
    ExecutionError,
    ExecutionPaused,
    TopologySelectionError,
)
from agloom.loop.contracts import LoopController, LoopEvaluator, LoopObserver
from agloom.loop.machine import LoopMachine
from agloom.loop.models import LoopPhase, LoopPolicy, LoopResult, LoopState
from agloom.loop.runner import IterationScope, LoopIntegrations, LoopRunner
from agloom.models import (
    ArchitectureSpec,
    ExecutionDecision,
    ExecutionLimits,
    ExecutionMode,
    PipelineStage,
    Strategy,
    TaskAnalysis,
    TopologyName,
    WorkerSpec,
)
from agloom.observability.base import CompositeObserver
from agloom.observability.events import AgentEvent, EventName
from agloom.policy._arguments import InvalidArguments, call_arguments
from agloom.policy._middleware import PolicyMiddleware
from agloom.policy.manager import PolicyManager
from agloom.topology._shared import (
    agent_text,
    topology_answer,
    topology_input,
    with_context,
)
from agloom.topology.base import (
    TopologyBuildContext,
    compose_prompt,
    langchain_agent,
)

if TYPE_CHECKING:
    from feedback_manager import FeedbackManager
    from langgraph_xai import XAIRuntime
    from mcp_capability_router import MCPRuntime
    from refresh_engine import RefreshEngine

_GRAPH_CACHE_SIZE = 64
BREAKPOINT_INTERRUPT = "agloom.breakpoint"


@dataclass(frozen=True)
class AgentSettings:
    """Validated settings of one agent, produced by `create_agent`."""

    model: BaseChatModel
    tools: tuple[BaseTool, ...]
    system_prompt: str | None
    pattern: tuple[TopologyName, ...] | None
    composition: tuple[TopologyName, ...]
    workers: tuple[WorkerSpec, ...]
    stages: tuple[PipelineStage, ...]
    max_reflections: int
    limits: ExecutionLimits
    registry: CapabilityRegistry
    router: CapabilityRouter
    mcp: MCPRuntime | None
    middleware: tuple[AgentMiddleware[Any, Any], ...]
    policy: PolicyManager | None
    checkpointer: BaseCheckpointSaver[Any] | None
    checkpoint_authorizer: Callable[[str], bool] | None
    interrupt_before: tuple[str, ...]
    interrupt_after: tuple[str, ...]
    loop_policy: LoopPolicy
    loop_observer: LoopObserver
    loop_evaluator: LoopEvaluator
    loop_controller: LoopController
    feedback: FeedbackManager | None
    refresh: RefreshEngine | None
    explainability: XAIRuntime | None
    observers: CompositeObserver
    strategy: Strategy
    compiler: ArchitectureCompiler


@dataclass(frozen=True)
class _Plan:
    decision: ExecutionDecision
    spec: ArchitectureSpec | None
    selection: CapabilitySelection
    dynamic_tools: Mapping[str, BaseTool] = field(default_factory=dict)
    notes: str | None = None


@dataclass(frozen=True)
class _PlanRequest:
    """Inputs of one planning decision."""

    task: str
    run: RunContext
    depth: int
    forced: TopologyName | None


class _PlanningState(TypedDict, total=False):
    """State of the planning graph; checkpointed, so JSON values only."""

    task: str
    depth: int
    forced: str | None
    record: dict[str, Any]


def _selection(
    record: Mapping[str, Any], task: str, catalog: Sequence[CapabilityDescriptor]
) -> CapabilitySelection:
    """Return the recorded capability selection, as far as the catalog still has it."""
    available = {capability.name: capability for capability in catalog}
    return CapabilitySelection(
        task,
        tuple(available[name] for name in record["capabilities"] if name in available),
    )


class Agent(Runnable[Any, Any]):
    """A LangChain Runnable executing requests with DIRECT or one of eight topologies.

    Every invocation runs through Loop Engineering. The default loop policy is
    one-shot, so Loop Engineering adds no model calls to a simple request.
    Automatic selection adds one structured task-analysis call; an explicit
    `pattern` skips it. Create agents with [`create_agent`][agloom.create_agent].
    """

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self.id = uuid.uuid4().hex
        self._observers = settings.observers
        self._instrument: Callable[[Any], Any] = (
            settings.explainability.instrument
            if settings.explainability is not None
            else _identity
        )
        policy_middleware: tuple[AgentMiddleware[Any, Any], ...] = (
            (PolicyMiddleware(settings.policy),) if settings.policy is not None else ()
        )
        self._base_context = TopologyBuildContext(
            model=settings.model,
            tools=settings.tools,
            middleware=(
                ToolRoutingMiddleware(),
                *settings.middleware,
                *policy_middleware,
            ),
            workers=settings.workers,
            stages=settings.stages,
            system_prompt=settings.system_prompt,
            limits=settings.limits,
            max_reflections=settings.max_reflections,
            composition=settings.composition,
            interrupt_before=settings.interrupt_before,
            interrupt_after=settings.interrupt_after,
            screen_answers=bool(settings.middleware),
        )
        self._nested_context = replace(
            self._base_context, interrupt_before=(), interrupt_after=()
        )
        self._fixed_spec = self._explicit_spec()
        self._planning = self._planning_graph()
        needs_analysis = self._fixed_spec is None or settings.limits.recursion_enabled
        self._analyzer = (
            TaskAnalyzer(
                settings.model,
                settings.workers,
                settings.stages,
                instrument=self._instrument,
            )
            if needs_analysis
            else None
        )
        self._direct_agent = bool(settings.middleware)
        direct: Runnable[Any, Any] = (
            langchain_agent(
                settings.model,
                name="agloom_direct",
                tools=[],
                system_prompt=settings.system_prompt,
                middleware=list(self._base_context.middleware),
            )
            if self._direct_agent
            else settings.model
        )
        self._direct: Runnable[Any, Any] = cast(
            Runnable[Any, Any], self._instrument(direct)
        )
        self._graphs: OrderedDict[tuple[object, ...], Runnable[Any, Any]] = (
            OrderedDict()
        )
        self._graph_lock = threading.Lock()
        self._active: set[threading.Event] = set()
        self._active_lock = threading.Lock()
        self._runner = LoopRunner(
            policy=settings.loop_policy,
            machine=LoopMachine(
                settings.loop_policy, settings.loop_observer, settings.loop_controller
            ),
            evaluator=settings.loop_evaluator,
            operation=self._execute,
            aoperation=self._aexecute,
            checkpointer=settings.checkpointer,
            integrations=LoopIntegrations(
                refresh=settings.refresh, feedback=settings.feedback
            ),
        )
        if self._fixed_spec is not None:
            self._compiled(self._fixed_spec, nested=False)
        self._observers.on_event(
            AgentEvent(name=EventName.AGENT_CREATED, agent_id=self.id, run_id=self.id)
        )

    @property
    def capabilities(self) -> CapabilityRegistry:
        """The agent's capability catalog (local tools and application capabilities)."""
        return self._settings.registry

    @property
    def policy(self) -> PolicyManager | None:
        """The policy manager governing tool calls, if configured."""
        return self._settings.policy

    @property
    def loop_policy(self) -> LoopPolicy:
        """The Loop Engineering policy."""
        return self._settings.loop_policy

    @property
    def limits(self) -> ExecutionLimits:
        """The execution limits."""
        return self._settings.limits

    @property
    def xai(self) -> XAIRuntime | None:
        """The langgraph-xai runtime, if explainability is configured."""
        return self._settings.explainability

    @property
    def feedback(self) -> FeedbackManager | None:
        """The feedback manager, if configured."""
        return self._settings.feedback

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run one request and return its answer.

        Args:
            input: Text, messages, a mapping, a LangGraph `Command` resuming
                pending interrupts, or `None` to continue a checkpointed
                thread from its last checkpoint.
            config: LangChain run configuration; `configurable.thread_id`
                selects the checkpoint thread.
            context: LangGraph runtime context passed to tools and middleware.
            **kwargs: Not supported; present for `Runnable` compatibility.

        Returns:
            An `AIMessage` for text and message input, or a mapping with the
            conversation `messages` for mapping input.

        Raises:
            ExecutionPaused: The run paused at an interrupt, such as a pending
                policy approval or a configured breakpoint.
        """
        _reject_options(kwargs)
        return _surface(self.invoke_loop(input, config, context=context), config)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously run one request and return its answer. See `invoke`."""
        _reject_options(kwargs)
        return _surface(await self.ainvoke_loop(input, config, context=context), config)

    def invoke_loop(
        self, input: Any, config: RunnableConfig | None = None, *, context: Any = None
    ) -> LoopResult:
        """Run one request and return its answer with the Loop Engineering state.

        Pauses are returned as data in `LoopResult.interrupts` instead of
        raising `ExecutionPaused`.
        """
        self._require_sync()
        prepared = self._prepare(input, config)
        try:
            output = self._runner.graph.invoke(
                prepared.graph_input, prepared.config, context=context, version="v2"
            )
        except BaseException as error:
            self._stopped(prepared, error)
            raise
        finally:
            self._unregister(prepared.run)
        return self._result(prepared, output)

    async def ainvoke_loop(
        self, input: Any, config: RunnableConfig | None = None, *, context: Any = None
    ) -> LoopResult:
        """Asynchronously run one request with its loop state. See `invoke_loop`."""
        prepared = self._prepare(input, config)
        try:
            output = await self._runner.agraph.ainvoke(
                prepared.graph_input, prepared.config, context=context, version="v2"
            )
        except BaseException as error:
            self._stopped(prepared, error)
            raise
        finally:
            self._unregister(prepared.run)
        return self._result(prepared, output)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: Any = "updates",
        subgraphs: bool = False,
        context: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream one run with LangGraph stream modes.

        Use `stream_mode="messages"` for the tokens of every model call (internal
        routing decisions excluded), `"custom"` for Agloom events, and
        `"updates"` for loop progress. `"messages"` and `"custom"` include the
        nested topology graphs; other modes report the top level unless
        `subgraphs=True`, which also yields LangGraph namespaces. Additional
        keyword arguments are passed to LangGraph's `stream`.
        """
        self._require_sync()
        prepared = self._prepare(input, config)
        try:
            for item in self._runner.graph.stream(
                prepared.graph_input,
                prepared.config,
                stream_mode=stream_mode,
                subgraphs=True,
                context=context,
                **kwargs,
            ):
                part = _stream_part(item, stream_mode, subgraphs)
                if part is not _SKIPPED:
                    yield part
        except BaseException as error:
            self._stopped(prepared, error)
            raise
        finally:
            self._unregister(prepared.run)
        self._streamed(
            prepared,
            self._runner.graph.get_state if self._settings.checkpointer else None,
        )

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: Any = "updates",
        subgraphs: bool = False,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronously stream one run. See `stream`."""
        prepared = self._prepare(input, config)
        try:
            async for item in self._runner.agraph.astream(
                prepared.graph_input,
                prepared.config,
                stream_mode=stream_mode,
                subgraphs=True,
                context=context,
                **kwargs,
            ):
                part = _stream_part(item, stream_mode, subgraphs)
                if part is not _SKIPPED:
                    yield part
        except BaseException as error:
            self._stopped(prepared, error)
            raise
        finally:
            self._unregister(prepared.run)
        snapshot = (
            await self._runner.agraph.aget_state(prepared.config)
            if self._settings.checkpointer is not None
            else None
        )
        self._streamed(
            prepared, (lambda _config: snapshot) if snapshot is not None else None
        )

    def transform(
        self,
        input: Iterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Run once on the combined input; used when the agent streams in a chain.

        Inside LangChain compositions the agent emits its final answer, like
        `invoke`, rather than LangGraph stream parts.
        """
        final: Any = _SKIPPED
        for chunk in input:
            final = chunk if final is _SKIPPED else final + chunk
        if final is not _SKIPPED:
            yield self.invoke(final, config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Asynchronously run once on the combined input. See `transform`."""
        final: Any = _SKIPPED
        async for chunk in input:
            final = chunk if final is _SKIPPED else final + chunk
        if final is not _SKIPPED:
            yield await self.ainvoke(final, config, **kwargs)

    def resume(
        self, config: RunnableConfig, value: Any = True, *, context: Any = None
    ) -> Any:
        """Resume a paused checkpoint thread.

        Pending interrupts receive `value`. A thread stopped at an
        `interrupt_before`/`interrupt_after` breakpoint, or interrupted by a
        crash or cancellation, continues from its last checkpoint; completed
        steps are not repeated. For policy approvals, decide the action in
        agentbarrier first; the resumed call then executes exactly once, or
        fails closed when it was rejected.

        Args:
            config: Configuration selecting the checkpoint thread.
            value: Resume value; for loop pauses, human feedback text or
                `{"feedback": ...}`.
            context: LangGraph runtime context.

        Raises:
            ConfigurationError: If the thread has nothing to resume.
        """
        snapshot = self.get_state(config)
        return self.invoke(_resume_input(snapshot, value), config, context=context)

    async def aresume(
        self, config: RunnableConfig, value: Any = True, *, context: Any = None
    ) -> Any:
        """Asynchronously resume a paused checkpoint thread. See `resume`."""
        snapshot = await self.aget_state(config)
        return await self.ainvoke(
            _resume_input(snapshot, value), config, context=context
        )

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        """Return the checkpoint snapshot of a thread, including pending interrupts."""
        self._authorize(config, require=True)
        return self._runner.graph.get_state(config, subgraphs=True)

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Asynchronously return the checkpoint snapshot of a thread."""
        self._authorize(config, require=True)
        return await self._runner.agraph.aget_state(config, subgraphs=True)

    def stop(self) -> None:
        """Request cooperative cancellation of every active invocation of this agent."""
        with self._active_lock:
            events = tuple(self._active)
        for event in events:
            event.set()

    def route_capabilities(self, task: str) -> CapabilitySelection:
        """Select local and application capabilities for a task."""
        return self._settings.router.route(task, self._settings.registry.descriptors())

    async def aroute_capabilities(self, task: str) -> CapabilitySelection:
        """Select local, application, and MCP capabilities for a task."""
        return await self._settings.router.aroute(task, await self._acatalog())

    def execute_capability(self, name: str, input: Any = None) -> Any:
        """Execute one local or application capability under the policy.

        With a policy, tool arguments are validated against the tool's schema
        first, so the policy judges the values the tool receives. Returns what
        the capability produces: a LangChain tool's output or the application
        object's return value.

        Raises:
            CapabilityResolutionError: If the capability is unknown, cannot run
                synchronously, or, with a policy, the arguments do not match the
                tool's schema.
            PolicyViolationError: If the policy denies a call to a capability
                that is not consequential. For consequential capabilities,
                agentbarrier's errors propagate unchanged, for example
                `PolicyDenied`, `ApprovalRequired`, `ActionInProgress`, and
                `ActionOutcomeUnknown`.
        """
        capability = self._settings.registry.descriptor(name).value
        policy = self._settings.policy
        if policy is None:
            return _call(capability, input)
        arguments, value = _governed_input(name, capability, input)
        return policy._execute(
            name, arguments, lambda: _call(capability, value), self._direct_run()
        )

    async def aexecute_capability(self, name: str, input: Any = None) -> Any:
        """Execute one capability under the policy, including MCP capabilities.

        Returns what the capability produces. MCP results come from
        `mcp-capability-router` unchanged: a tool call returns a LangChain
        `ToolMessage`, a resource read returns the list of MCP resource
        contents, and a prompt returns MCP's `GetPromptResult`.

        Raises:
            CapabilityResolutionError: If the capability is unknown, or, with a
                policy, the arguments do not match the tool's schema.
            PolicyViolationError: If the policy denies a call to a capability
                that is not consequential. For consequential capabilities,
                agentbarrier's errors propagate unchanged.
        """
        descriptor = next(
            (item for item in await self._acatalog() if item.name == name), None
        )
        if descriptor is None:
            raise CapabilityResolutionError(f"capability {name!r} is not registered")
        capability = descriptor.value
        policy = self._settings.policy
        if policy is None:
            return await _acall(capability, input)
        schema_tool = (
            await capability.as_tool()
            if isinstance(capability, MCPCapability)
            and capability.capability_type == "tool"
            else capability
        )
        arguments, value = _governed_input(name, schema_tool, input)

        async def operation() -> Any:
            return await _acall(capability, value)

        return await policy._aexecute(name, arguments, operation, self._direct_run())

    def with_structured_output(self, schema: Any, **options: Any) -> Runnable[Any, Any]:
        """Return a runnable that validates and repairs the answer with xstructured.

        The schema instructions are appended to the latest user message, so the
        runnable accepts the agent's inputs: text, messages, or a mapping. It
        returns an `xstructured.XStructuredResult` whose `structured` attribute
        holds the validated value.

        Args:
            schema: A Pydantic model, a `TypeAdapter`, a type annotation, or
                named schemas.
            **options: Keyword arguments for `xstructured.with_xstructured_output`.
                With `inject_instructions=False` the input is passed unchanged,
                so you can place the runnable's `instructions` in your own prompt.
        """
        from xstructured import with_xstructured_output

        inject = options.pop("inject_instructions", True)
        answering = _Answering(self)
        structured = with_xstructured_output(
            answering, schema, inject_instructions=False, **options
        )
        if inject:
            answering.instructions = structured.instructions
        return structured

    def flush(self) -> None:
        """Flush observer and explainability telemetry."""
        if self._settings.explainability is not None:
            xai = self._settings.explainability
            xai.run_sync(xai.flush())
        self._observers.flush()

    def close(self) -> None:
        """Flush telemetry and shut the observers down.

        Injected components (MCP runtime, refresh engine, feedback manager,
        policy store, explainability runtime) are owned by the application.
        """
        self.flush()
        self._observers.shutdown()

    def _require_sync(self) -> None:
        settings = self._settings
        if (
            settings.mcp is not None
            or settings.refresh is not None
            or settings.feedback is not None
        ):
            raise ConfigurationError(
                "this agent uses asynchronous integrations (mcp, refresh, feedback); "
                "use ainvoke, ainvoke_loop, astream, or aresume"
            )

    def _prepare(self, input: Any, config: RunnableConfig | None) -> _Prepared:
        resolved = _graph_config(ensure_config(config), config)
        self._authorize(resolved, require=input is None)
        if input is None or isinstance(input, Command):
            graph_input: Any = input
            original, mapping = None, False
        else:
            task, messages, mapping = normalize_input(input)
            graph_input = {"task": task, "messages": messages}
            original = input
        run_id = uuid.uuid4().hex
        event = threading.Event()
        with self._active_lock:
            self._active.add(event)
        run = RunContext(
            agent_id=self.id,
            run_id=run_id,
            cancel_event=event,
            emit=self._emitter(run_id),
            pausable=self._settings.checkpointer is not None,
            recursion=RecursionSession(self._settings.limits)
            if self._settings.limits.recursion_enabled
            else None,
        )
        handlers = self._observers.callback_handlers
        if handlers:
            resolved = _with_handlers(resolved, handlers)
        run.emit(
            EventName.EXECUTION_STARTED,
            {"resume": input is None or isinstance(input, Command)},
        )
        return _Prepared(
            run=run,
            config=with_run(resolved, run),
            graph_input=graph_input,
            original=original,
            mapping=mapping,
        )

    def _result(self, prepared: _Prepared, output: GraphOutput[Any]) -> LoopResult:
        run = prepared.run
        value = output.value if isinstance(output.value, Mapping) else {}
        interrupts = tuple(output.interrupts)
        if not interrupts and "state" not in value:
            interrupts = (self._breakpoint(),)
        if interrupts:
            run.emit(EventName.EXECUTION_PAUSED, {"interrupts": len(interrupts)})
            return LoopResult(
                result=None,
                state=_waiting(run.scratch.get("state")),
                interrupts=interrupts,
            )
        final = LoopState.model_validate(value["state"])
        raw = value.get("result")
        error = _loop_error(run, final)
        answer = (
            shape_result(answer_message(raw), prepared.original, prepared.mapping)
            if raw is not None
            else None
        )
        if error is not None:
            run.emit(EventName.EXECUTION_FAILED, {"error": type(error).__name__})
        else:
            run.emit(EventName.EXECUTION_COMPLETED, {"iterations": final.iteration})
        return LoopResult(result=answer, state=final, error=error)

    def _breakpoint(self) -> Interrupt:
        settings = self._settings
        return Interrupt(
            value={
                "type": BREAKPOINT_INTERRUPT,
                "interrupt_before": list(settings.interrupt_before),
                "interrupt_after": list(settings.interrupt_after),
            },
            id=BREAKPOINT_INTERRUPT,
        )

    def _stopped(self, prepared: _Prepared, error: BaseException) -> None:
        """Report how a run that raised ended."""
        run = prepared.run
        if isinstance(error, GraphInterrupt):
            # Inside an enclosing graph, a pause propagates as LangGraph's interrupt.
            pending = {interrupt.id for interrupt in error.args[0]}
            run.emit(EventName.EXECUTION_PAUSED, {"interrupts": max(1, len(pending))})
        elif isinstance(error, GraphDrained):
            run.emit(EventName.EXECUTION_PAUSED, {"interrupts": 0})
        elif isinstance(
            error, CancellationError | GeneratorExit | asyncio.CancelledError
        ):
            run.emit(EventName.EXECUTION_CANCELLED, {})
        else:
            run.emit(EventName.EXECUTION_FAILED, {"error": type(error).__name__})

    def _streamed(
        self,
        prepared: _Prepared,
        snapshot: Callable[[RunnableConfig], StateSnapshot | None] | None,
    ) -> None:
        state = snapshot(prepared.config) if snapshot is not None else None
        if state is not None and (state.interrupts or state.next):
            prepared.run.emit(
                EventName.EXECUTION_PAUSED,
                {"interrupts": max(1, len(state.interrupts))},
            )
        else:
            prepared.run.emit(EventName.EXECUTION_COMPLETED, {})

    def _unregister(self, run: RunContext) -> None:
        with self._active_lock:
            self._active.discard(run.cancel_event)

    def _authorize(self, config: RunnableConfig, *, require: bool) -> None:
        settings = self._settings
        if settings.checkpointer is None:
            if require:
                raise ConfigurationError(
                    "checkpoint state requires a configured checkpointer"
                )
            return
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ConfigurationError(
                "a non-empty configurable.thread_id is required with a checkpointer"
            )
        authorizer = settings.checkpoint_authorizer
        if authorizer is None or not authorizer(thread_id):
            raise ConfigurationError("checkpoint access is not authorized")

    def _emitter(self, run_id: str) -> Callable[[str, Mapping[str, Any]], None]:
        def emit(name: str, payload: Mapping[str, Any]) -> None:
            details = dict(payload)
            topology = details.pop("topology", None)
            self._observers.on_event(
                AgentEvent(
                    name=name,
                    agent_id=self.id,
                    run_id=run_id,
                    topology=str(topology) if topology else None,
                    payload=details,
                )
            )
            write_stream({"agloom_event": name, "run_id": run_id, **payload})

        return emit

    def _direct_run(self) -> RunContext:
        run_id = uuid.uuid4().hex
        return RunContext(
            agent_id=self.id,
            run_id=run_id,
            cancel_event=threading.Event(),
            emit=self._emitter(run_id),
        )

    def _execute(self, scope: IterationScope, config: RunnableConfig) -> AIMessage:
        iteration_run = self._iteration_run(_require_run(config), scope)
        plan = self._pinned_plan(
            _PlanRequest(scope.task, iteration_run, 0, scope.requested_topology),
            config,
        )
        scope.topology = plan.decision.topology
        return self._run_plan(
            plan, scope.task, scope.messages, scope.context, iteration_run, config
        )

    async def _aexecute(
        self, scope: IterationScope, config: RunnableConfig
    ) -> AIMessage:
        iteration_run = self._iteration_run(_require_run(config), scope)
        plan = await self._apinned_plan(
            _PlanRequest(scope.task, iteration_run, 0, scope.requested_topology),
            config,
        )
        scope.topology = plan.decision.topology
        return await self._arun_plan(
            plan, scope.task, scope.messages, scope.context, iteration_run, config
        )

    def _iteration_run(self, run: RunContext, scope: IterationScope) -> RunContext:
        iteration_run = run.derive(tool_budget=scope.tool_budget)
        if run.recursion is None:
            return iteration_run
        return iteration_run.derive(
            child=_ChildRunner(self, run.recursion, iteration_run)
        )

    def _planning_graph(self) -> Runnable[_PlanningState, dict[str, Any]]:
        """Compile the graph that records the planning decisions of one execution.

        Routing and task analysis may differ between calls, so a checkpointed run
        makes them once, in this nested graph; a resumed run reads the recorded
        result instead of planning again. A completed nested graph keeps its
        result even where LangGraph re-runs the finished work of an interrupted
        step (langchain-ai/langgraph#9106), which a nested task does not.
        """
        builder = StateGraph[Any](_PlanningState)
        builder.add_node(
            "record",
            RunnableLambda(self._record_node, afunc=self._arecord_node, name="record"),
        )
        builder.add_edge(START, "record")
        builder.add_edge("record", END)
        return cast(
            Runnable[_PlanningState, dict[str, Any]],
            builder.compile(name="agloom_plan"),
        )

    def _pinned_plan(self, request: _PlanRequest, config: RunnableConfig) -> _Plan:
        """Plan an execution once; a resumed run reuses the recorded plan."""
        state = self._planning.invoke(
            _planning_input(request), with_run(config, request.run)
        )
        record = state["record"]
        selection = _selection(
            record, request.task, self._settings.registry.descriptors()
        )
        return self._plan_from(record, selection, {})

    async def _apinned_plan(
        self, request: _PlanRequest, config: RunnableConfig
    ) -> _Plan:
        state = await self._planning.ainvoke(
            _planning_input(request), with_run(config, request.run)
        )
        record = state["record"]
        selection = _selection(record, request.task, await self._acatalog())
        dynamic = {
            capability.name: await capability.value.as_tool()
            for capability in selection.capabilities
            if capability.kind is CapabilityKind.MCP and capability.model_callable
        }
        return self._plan_from(record, selection, dynamic)

    def _record_node(
        self, state: _PlanningState, config: RunnableConfig
    ) -> dict[str, Any]:
        return {"record": self._record_plan(_planned_request(state, config), config)}

    async def _arecord_node(
        self, state: _PlanningState, config: RunnableConfig
    ) -> dict[str, Any]:
        request = _planned_request(state, config)
        return {"record": await self._arecord_plan(request, config)}

    def _record_plan(
        self, request: _PlanRequest, config: RunnableConfig
    ) -> dict[str, Any]:
        """Make the planning decisions of one execution and return them as JSON."""
        catalog = self._settings.registry.descriptors()
        selection = self._settings.router.route(request.task, catalog)
        self._routed(request.run, selection, catalog)
        if self._fixed(request):
            return self._fixed_record(request, selection)
        candidates, allow_direct = self._candidates(request.depth, request.forced)
        request.run.emit(EventName.TOPOLOGY_SELECTION_STARTED, {"depth": request.depth})
        analysis = self._require_analyzer().analyze(
            request.task, selection.capabilities, config
        )
        return self._analyzed_record(
            request, selection, analysis, candidates, allow_direct
        )

    async def _arecord_plan(
        self, request: _PlanRequest, config: RunnableConfig
    ) -> dict[str, Any]:
        catalog = await self._acatalog()
        selection = await self._settings.router.aroute(request.task, catalog)
        self._routed(request.run, selection, catalog)
        if self._fixed(request):
            return self._fixed_record(request, selection)
        candidates, allow_direct = self._candidates(request.depth, request.forced)
        request.run.emit(EventName.TOPOLOGY_SELECTION_STARTED, {"depth": request.depth})
        analysis = await self._require_analyzer().aanalyze(
            request.task, selection.capabilities, config
        )
        return self._analyzed_record(
            request, selection, analysis, candidates, allow_direct
        )

    async def _acatalog(self) -> tuple[CapabilityDescriptor, ...]:
        local = self._settings.registry.descriptors()
        if self._settings.mcp is None:
            return local
        policy = self._settings.policy
        remote = tuple(
            descriptor
            for descriptor in await mcp_descriptors(self._settings.mcp)
            if policy is None
            or not descriptor.model_callable
            or policy.visible(descriptor.name)
        )
        return (*local, *remote)

    @staticmethod
    def _routed(
        run: RunContext,
        selection: CapabilitySelection,
        catalog: Sequence[CapabilityDescriptor],
    ) -> None:
        run.emit(
            EventName.CAPABILITIES_ROUTED,
            {"selected": len(selection.capabilities), "available": len(catalog)},
        )

    def _fixed(self, request: _PlanRequest) -> bool:
        """Whether the developer configured the topology this execution runs."""
        spec = self._fixed_spec
        if request.depth > 0 or spec is None:
            return False
        if request.forced is not None and request.forced is not spec.topology:
            raise ConfigurationError(
                "Loop Engineering cannot override the explicitly configured topology"
            )
        return True

    def _fixed_record(
        self, request: _PlanRequest, selection: CapabilitySelection
    ) -> dict[str, Any]:
        spec = self._require_fixed_spec()
        decision = ExecutionDecision(
            mode=ExecutionMode.TOPOLOGY,
            topology=spec.topology,
            reason="the developer configured one explicit topology",
        )
        self._selected(request, decision)
        return {
            "capabilities": list(selection.names),
            "decision": decision.model_dump(mode="json"),
            "analysis": None,
        }

    @staticmethod
    def _selected(
        request: _PlanRequest, decision: ExecutionDecision, *, missing: int = 0
    ) -> None:
        request.run.emit(
            EventName.TOPOLOGY_SELECTED,
            {
                "topology": decision.topology.value if decision.topology else None,
                "mode": decision.mode.value,
                "reason": decision.reason,
                "missing_capabilities": missing,
                "depth": request.depth,
            },
        )

    def _candidates(
        self, depth: int, forced: TopologyName | None
    ) -> tuple[tuple[TopologyName, ...] | None, bool]:
        base = (
            self._settings.limits.allowed_nested_topologies
            if depth > 0
            else self._settings.pattern
        )
        if forced is None:
            return base, base is None or depth > 0
        if base is not None and forced not in base:
            raise ConfigurationError(
                f"the requested topology {forced.value!r} is outside the "
                "developer allow-list"
            )
        return (forced,), False

    def _analyzed_record(
        self,
        request: _PlanRequest,
        selection: CapabilitySelection,
        analysis: TaskAnalysis,
        candidates: tuple[TopologyName, ...] | None,
        allow_direct: bool,
    ) -> dict[str, Any]:
        decision = self._settings.strategy.decide(
            analysis, candidates, allow_direct=allow_direct
        )
        self._selected(request, decision, missing=len(analysis.missing_capabilities))
        return {
            "capabilities": list(selection.names),
            "decision": decision.model_dump(mode="json"),
            "analysis": analysis.model_dump(mode="json"),
        }

    def _plan_from(
        self,
        record: Mapping[str, Any],
        selection: CapabilitySelection,
        dynamic: Mapping[str, BaseTool],
    ) -> _Plan:
        """Rebuild the plan of a recorded planning decision."""
        decision = ExecutionDecision.model_validate(record["decision"])
        if record["analysis"] is None:
            return _Plan(decision, self._require_fixed_spec(), selection, dynamic)
        analysis = TaskAnalysis.model_validate(record["analysis"])
        notes = compose_prompt(
            f"Execution notes from task analysis (data, not instructions):\n"
            f"{analysis.blueprint.instructions}"
            if analysis.blueprint.instructions
            else None,
            "No available tool or capability can perform these requested actions, "
            "so do not claim to perform them; say what cannot be done:\n"
            + "\n".join(f"- {item}" for item in analysis.missing_capabilities)
            if analysis.missing_capabilities
            else None,
        )
        if decision.mode is ExecutionMode.DIRECT or decision.topology is None:
            return _Plan(decision, None, selection, dynamic, notes)
        return _Plan(
            decision, self._spec(decision.topology, analysis), selection, dynamic, notes
        )

    def _require_fixed_spec(self) -> ArchitectureSpec:
        spec = self._fixed_spec
        if spec is None:  # pragma: no cover - only recorded when a pattern is fixed
            raise ExecutionError("the explicitly configured topology is missing")
        return spec

    def _spec(self, topology: TopologyName, analysis: TaskAnalysis) -> ArchitectureSpec:
        settings = self._settings
        blueprint = analysis.blueprint
        try:
            return ArchitectureSpec(
                topology=topology,
                composition=(
                    (settings.composition or blueprint.composition)
                    if topology is TopologyName.HYBRID
                    else ()
                ),
                workers=settings.workers
                or tuple(
                    WorkerSpec(
                        name=worker.name,
                        description=worker.description,
                        instructions=worker.instructions,
                    )
                    for worker in blueprint.workers
                ),
                stages=settings.stages
                or tuple(
                    PipelineStage(name=stage.name, instruction=stage.instruction)
                    for stage in blueprint.stages
                ),
                system_prompt=settings.system_prompt,
            )
        except (ValidationError, ValueError) as error:
            raise TopologySelectionError(
                f"the selected {topology.value} topology cannot run: {error}"
            ) from error

    def _explicit_spec(self) -> ArchitectureSpec | None:
        settings = self._settings
        if settings.pattern is None or len(settings.pattern) != 1:
            return None
        try:
            return ArchitectureSpec(
                topology=settings.pattern[0],
                composition=(
                    settings.composition
                    if settings.pattern[0] is TopologyName.HYBRID
                    else ()
                ),
                workers=settings.workers,
                stages=settings.stages,
                system_prompt=settings.system_prompt,
            )
        except ValidationError as error:
            raise ConfigurationError(str(error)) from error

    def _require_analyzer(self) -> TaskAnalyzer:
        # Explicit patterns without recursion never analyze, so no analyzer exists.
        if self._analyzer is None:  # pragma: no cover - guarded by _fixed_plan
            raise TopologySelectionError(
                "automatic execution selection is not configured"
            )
        return self._analyzer

    def _compiled(self, spec: ArchitectureSpec, *, nested: bool) -> Runnable[Any, Any]:
        # Breakpoints apply to the top-level topology only, never to recursive children.
        breakpoints = not nested and bool(
            self._settings.interrupt_before or self._settings.interrupt_after
        )
        key = (breakpoints, spec.cache_key)
        with self._graph_lock:
            graph = self._graphs.get(key)
            if graph is not None:
                self._graphs.move_to_end(key)
                return graph
        context = self._base_context if breakpoints else self._nested_context
        graph = self._instrument(self._settings.compiler.compile(spec, context).graph)
        with self._graph_lock:
            graph = self._graphs.setdefault(key, graph)
            while len(self._graphs) > _GRAPH_CACHE_SIZE:
                self._graphs.popitem(last=False)
        return graph

    def _execution(
        self,
        plan: _Plan,
        task: str,
        messages: Sequence[BaseMessage],
        context: str | None,
        run: RunContext,
        config: RunnableConfig,
    ) -> _Execution:
        context_text = compose_prompt(plan.notes, context)
        if plan.spec is None:
            run.emit(EventName.DIRECT_EXECUTION_STARTED, {})
            direct = run.derive(tool_names=frozenset(), dynamic_tools={})
            return _Execution(
                runnable=self._direct,
                input=self._direct_input(messages, context_text),
                config=with_run(config, direct),
                run=direct,
                topology=None,
            )
        scoped_run = run.derive(
            tool_names=plan.selection.tool_names, dynamic_tools=plan.dynamic_tools
        )
        topology = plan.spec.topology
        return _Execution(
            runnable=self._compiled(plan.spec, nested=run.depth > 0),
            input=topology_input(
                topology is TopologyName.REACT, task, messages, context_text
            ),
            config=with_run(config, scoped_run),
            run=scoped_run,
            topology=topology,
        )

    def _run_plan(
        self,
        plan: _Plan,
        task: str,
        messages: Sequence[BaseMessage],
        context: str | None,
        run: RunContext,
        config: RunnableConfig,
    ) -> AIMessage:
        execution = self._execution(plan, task, messages, context, run, config)
        with scoped(execution.run):
            result = execution.runnable.invoke(execution.input, execution.config)
        return self._answer(result, execution.topology)

    async def _arun_plan(
        self,
        plan: _Plan,
        task: str,
        messages: Sequence[BaseMessage],
        context: str | None,
        run: RunContext,
        config: RunnableConfig,
    ) -> AIMessage:
        execution = self._execution(plan, task, messages, context, run, config)
        with scoped(execution.run):
            result = await execution.runnable.ainvoke(execution.input, execution.config)
        return self._answer(result, execution.topology)

    def _direct_input(
        self, messages: Sequence[BaseMessage], context: str | None
    ) -> Any:
        conversation = with_context(messages, context)
        if self._direct_agent:
            return {"messages": conversation}
        system = self._settings.system_prompt
        return ([SystemMessage(content=system)] if system else []) + conversation

    def _answer(self, result: Any, topology: TopologyName | None) -> AIMessage:
        if topology is None:
            if self._direct_agent:
                return _last_ai(result)
            return answer_message(result)
        if topology is TopologyName.REACT:
            return _last_ai(result)
        answer = topology_answer(False, result)
        return AIMessage(
            content=answer["output"],
            additional_kwargs={"loop": answer["loop"]} if answer["loop"] else {},
        )

    def _run_child(
        self,
        task: str,
        depth: int,
        session: RecursionSession,
        parent: RunContext,
        config: RunnableConfig,
    ) -> str:
        child = self._child_context(depth, session, parent)
        plan = self._pinned_plan(_PlanRequest(task, child, depth, None), config)
        answer = self._run_plan(
            plan, task, [HumanMessage(content=task)], None, child, config
        )
        session.check()
        return answer.text

    async def _arun_child(
        self,
        task: str,
        depth: int,
        session: RecursionSession,
        parent: RunContext,
        config: RunnableConfig,
    ) -> str:
        child = self._child_context(depth, session, parent)
        plan = await self._apinned_plan(_PlanRequest(task, child, depth, None), config)
        answer = await self._arun_plan(
            plan, task, [HumanMessage(content=task)], None, child, config
        )
        session.check()
        return answer.text

    def _child_context(
        self, depth: int, session: RecursionSession, parent: RunContext
    ) -> RunContext:
        child = parent.derive(depth=depth, child=None)
        if depth < self._settings.limits.max_depth:
            child = child.derive(child=_ChildRunner(self, session, child))
        return child


@dataclass(frozen=True)
class _Prepared:
    run: RunContext
    config: RunnableConfig
    graph_input: Any
    original: Any
    mapping: bool


@dataclass(frozen=True)
class _Execution:
    runnable: Runnable[Any, Any]
    input: Any
    config: RunnableConfig
    run: RunContext
    topology: TopologyName | None


class _ChildRunner:
    """Runs recursive children of one iteration through the owning agent."""

    def __init__(
        self, agent: Agent, session: RecursionSession, parent: RunContext
    ) -> None:
        self._agent = agent
        self._session = session
        self._parent = parent

    def run(self, task: str, config: RunnableConfig) -> str:
        depth = self._parent.depth + 1
        self._session.admit(depth)
        return self._agent._run_child(task, depth, self._session, self._parent, config)

    async def arun(self, task: str, config: RunnableConfig) -> str:
        depth = self._parent.depth + 1
        self._session.admit(depth)
        return await self._agent._arun_child(
            task, depth, self._session, self._parent, config
        )


class _Answering(Runnable[Any, AIMessage]):
    """Runs an agent with extra trusted instructions and returns its answer message."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self.instructions: str | None = None

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AIMessage:
        result = self._agent.invoke(
            _instructed(input, self.instructions), config, **kwargs
        )
        return _answer_of(result)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AIMessage:
        result = await self._agent.ainvoke(
            _instructed(input, self.instructions), config, **kwargs
        )
        return _answer_of(result)


_SKIPPED: Any = object()
_NESTED_STREAM_MODES = frozenset({"messages", "custom"})


def _stream_part(item: Any, stream_mode: Any, subgraphs: bool) -> Any:
    if subgraphs:
        return item
    namespace, *rest = item
    single = isinstance(stream_mode, str)
    mode = stream_mode if single else rest[0]
    if namespace and mode not in _NESTED_STREAM_MODES:
        return _SKIPPED
    return rest[0] if single else tuple(rest)


def _instructed(value: Any, instructions: str | None) -> Any:
    if not instructions or value is None or isinstance(value, Command):
        return value
    if isinstance(value, str):
        return f"{value}\n\n{instructions}"
    if isinstance(value, Mapping):
        if value.get("messages") is not None:
            return {
                **value,
                "messages": _with_instructions(value["messages"], instructions),
            }
        for key in ("task", "input", "prompt"):
            if value.get(key) is not None:
                return {**value, key: f"{value[key]}\n\n{instructions}"}
        return value
    return _with_instructions(value, instructions)


def _with_instructions(messages: Any, instructions: str) -> Any:
    try:
        resolved = convert_to_messages(
            [messages] if isinstance(messages, BaseMessage) else messages
        )
    except (TypeError, ValueError, NotImplementedError):
        return messages
    return with_context(resolved, instructions)


def _answer_of(result: Any) -> AIMessage:
    if isinstance(result, Mapping):
        return _last_ai(result)
    return answer_message(result)


def _identity(runnable: Runnable[Any, Any]) -> Runnable[Any, Any]:
    return runnable


def _graph_config(
    resolved: RunnableConfig, config: RunnableConfig | None
) -> RunnableConfig:
    """Return the configuration Agloom's graphs run with.

    LangChain Core fills in a recursion limit of 25 that LangGraph would treat
    as explicit, so only a limit set by the caller or an enclosing run is kept
    and LangGraph otherwise applies its own default, as LangChain agents do.
    `max_concurrency` is dropped: LangGraph applies it to nested tasks too, so a
    task waiting on another would wait forever. `ExecutionLimits.max_workers`
    bounds the agent's parallel branches instead.
    """
    ambient = var_child_runnable_config.get() or {}
    if (config or {}).get("recursion_limit") is None and (
        ambient.get("recursion_limit") is None
    ):
        resolved.pop("recursion_limit", None)
    resolved.pop("max_concurrency", None)
    return resolved


def _with_handlers(
    config: RunnableConfig, handlers: Sequence[BaseCallbackHandler]
) -> RunnableConfig:
    # `merge_configs` would re-read the ambient config of an enclosing graph and
    # overwrite the caller's `configurable`; only the callbacks change here.
    callbacks = config.get("callbacks")
    if callbacks is None or isinstance(callbacks, list):
        merged: Any = [*(callbacks or []), *handlers]
    else:
        merged = callbacks.copy()
        for handler in handlers:
            merged.add_handler(handler, inherit=True)
    return cast(RunnableConfig, {**config, "callbacks": merged})


def _require_run(config: RunnableConfig) -> RunContext:
    run = configured_run(config)
    if run is None:  # pragma: no cover - the loop only runs inside Agent invocations
        raise ExecutionError("the agent run context is missing")
    return run


def _planning_input(request: _PlanRequest) -> _PlanningState:
    forced = request.forced
    return {
        "task": request.task,
        "depth": request.depth,
        "forced": forced.value if forced is not None else None,
    }


def _planned_request(state: _PlanningState, config: RunnableConfig) -> _PlanRequest:
    forced = state.get("forced")
    return _PlanRequest(
        state["task"],
        _require_run(config),
        state["depth"],
        TopologyName(forced) if forced is not None else None,
    )


def _reject_options(options: Mapping[str, Any]) -> None:
    if options:
        raise TypeError(f"unexpected invocation options: {', '.join(sorted(options))}")


def _surface(result: LoopResult, config: RunnableConfig | None) -> Any:
    if result.interrupts:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        raise ExecutionPaused(result.interrupts, thread_id)
    error = result.error
    if error is not None:
        if isinstance(error, AgloomError | RuntimeBarrierError):
            raise error
        raise ExecutionError(
            f"agent execution failed with {type(error).__name__}"
        ) from error
    if result.state.phase is LoopPhase.FAILED:
        decision = result.state.decision
        reason = decision.reason if decision is not None else "the loop failed"
        raise ExecutionError(f"agent execution failed: {reason}")
    return result.result


def _waiting(state: LoopState | None) -> LoopState:
    if state is None:  # pragma: no cover - replays record the state before pausing
        raise ExecutionError("the run paused before its loop state was recorded")
    return state.model_copy(update={"phase": LoopPhase.WAITING})


def _loop_error(run: RunContext, state: LoopState) -> BaseException | None:
    observation = state.last_observation
    if observation is None or observation.success:
        return None
    return run.errors.get(state.iteration)


def _resume_input(snapshot: StateSnapshot, value: Any) -> Command[Any] | None:
    interrupts = snapshot.interrupts
    if len(interrupts) == 1:
        return Command(resume=value)
    if interrupts:
        return Command(resume={interrupt.id: value for interrupt in interrupts})
    if snapshot.next:
        return None
    raise ConfigurationError("the thread has nothing to resume")


def _last_ai(result: Any) -> AIMessage:
    messages = result.get("messages") if isinstance(result, Mapping) else None
    for message in reversed(messages or ()):
        if isinstance(message, AIMessage):
            return message
    return AIMessage(content=agent_text(result) if isinstance(result, Mapping) else "")


def _arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"input": value}


def _governed_input(
    name: str, capability: Any, value: Any
) -> tuple[dict[str, Any], Any]:
    """Return the arguments the policy judges and the input that runs a capability.

    Tools are validated against their schema, so both are the values the tool
    receives; other capabilities are judged on the input as given.
    """
    if not isinstance(capability, BaseTool):
        return _arguments(value), value
    try:
        arguments = call_arguments(capability, value)
    except InvalidArguments as invalid:
        raise CapabilityResolutionError(
            f"invalid arguments for {name!r}: {invalid}"
        ) from invalid
    return arguments.policy, arguments.values


def _call(capability: Any, value: Any) -> Any:
    if isinstance(capability, BaseTool):
        result = capability.invoke(value if value is not None else {})
    elif callable(getattr(capability, "invoke", None)):
        result = capability.invoke(value)
    elif callable(capability):
        result = capability(value)
    else:
        raise CapabilityResolutionError("the capability is not executable")
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise CapabilityResolutionError(
            "the capability is asynchronous; use Agent.aexecute_capability"
        )
    return result


async def _acall(capability: Any, value: Any) -> Any:
    if isinstance(capability, BaseTool):
        return await capability.ainvoke(value if value is not None else {})
    if callable(getattr(capability, "ainvoke", None)):
        result = capability.ainvoke(value)
    elif callable(getattr(capability, "invoke", None)):
        result = capability.invoke(value)
    elif callable(capability):
        result = capability(value)
    else:
        raise CapabilityResolutionError("the capability is not executable")
    return await result if inspect.isawaitable(result) else result
