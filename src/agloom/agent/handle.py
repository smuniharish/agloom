"""Runnable agent facade, resolution, and per-instance harness runtime."""

from __future__ import annotations

import asyncio
import inspect
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, cast

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agloom.analysis.analyzer import _CapabilityAwareAnalyzer
from agloom.capabilities.providers import (
    ApplicationCapabilityProvider,
    LocalLangChainToolProvider,
    MCPCapabilityProvider,
)
from agloom.capabilities.registry import (
    CapabilityDescriptor,
    CapabilityRegistry,
    InMemoryCapabilityRegistry,
)
from agloom.capabilities.router import (
    CapabilityRouter,
    CapabilitySelection,
)
from agloom.compiler.compiler import ArchitectureCompiler
from agloom.errors import (
    AgloomError,
    CancellationError,
    CapabilityResolutionError,
    ConfigurationError,
    ExecutionError,
    TopologySelectionError,
)
from agloom.models import (
    AgentConfig,
    ArchitectureSpec,
    ExecutionDecision,
    ExecutionMode,
    PipelineStage,
    Strategy,
    TaskAnalysis,
    TopologyName,
    WorkerSpec,
)
from agloom.observability import Observer
from agloom.runtime.events import AgentEvent
from agloom.runtime.lifecycle import Lifecycle, LifecycleState
from agloom.runtime.recursion import RecursionSession
from agloom.topology.base import TopologyArtifact


class Agent(Runnable[Any, Any]):
    """One isolated LangChain Runnable backed by DIRECT or LangGraph."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        strategy: Strategy,
        compiler: ArchitectureCompiler,
        observability: Observer,
        capability_registry: CapabilityRegistry | None = None,
        capability_router: CapabilityRouter,
        mcp_server_name: str | None = None,
        mcp_registration: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.observability = observability
        self._xai_model = (
            config.explainability.instrument(config.model)
            if config.explainability is not None
            else config.model
        )
        self._analyzer = _CapabilityAwareAnalyzer(
            self._xai_model,
            config.workers,
            config.stages,
            observability.callback_handlers,
        )
        self.strategy = strategy
        self.compiler = compiler
        self.capabilities = capability_registry or InMemoryCapabilityRegistry()
        self.capability_router = capability_router
        self._mcp_provider = MCPCapabilityProvider(
            config.capabilities,
            server_name=mcp_server_name,
        )
        self._register_capabilities(LocalLangChainToolProvider(config.tools).load())
        self._register_capabilities(
            ApplicationCapabilityProvider(config.capabilities).load()
        )
        self._register_capabilities(self._mcp_provider.load())
        self.structured_output: Any = None
        self._mcp_registration = (
            dict(mcp_registration) if mcp_registration is not None else None
        )
        self._mcp_init_lock = asyncio.Lock()
        self.id = str(uuid.uuid4())
        self._cache_lock = threading.RLock()
        self._artifacts: dict[tuple[Any, ...], TopologyArtifact] = {}
        self._active_lock = threading.Lock()
        self._active: set[threading.Event] = set()
        self._fixed_spec: ArchitectureSpec | None = None
        candidates = config.pattern_candidates
        if candidates is not None and len(candidates) == 1:
            topology = candidates[0]
            decision = ExecutionDecision(
                mode=ExecutionMode.TOPOLOGY,
                topology=topology,
                reason="single explicit developer topology",
            )
            self._fixed_spec = ArchitectureSpec(
                decision=decision,
                composition=(
                    config.composition if topology is TopologyName.HYBRID else ()
                ),
                checkpointer=config.checkpointer,
                workers=config.workers,
                stages=config.stages,
                system_prompt=config.system_prompt,
            )
            self._compile(self._fixed_spec)
        self._emit("AgentCreated", str(uuid.uuid4()))

    @property
    def xai(self) -> Any:
        """Return the agent's langgraph-xai runtime, or None when disabled."""

        return self.config.explainability

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise TypeError(f"unexpected invoke options: {', '.join(kwargs)}")
        return self._invoke(input, config, depth=0, session=None, child=False)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise TypeError(f"unexpected ainvoke options: {', '.join(kwargs)}")
        return await self._ainvoke(input, config, depth=0, session=None, child=False)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        if kwargs:
            raise TypeError(f"unexpected stream options: {', '.join(kwargs)}")
        self._initialize_capabilities_sync()
        decision, spec, task, messages, _, selection = self._resolve(input, depth=0)
        execution_config = self._run_config(
            config,
            threading.Event(),
            None,
            0,
            selected_capabilities=selection.names,
        )
        if decision.mode is ExecutionMode.DIRECT:
            yield from self._xai_model.stream(
                self._model_input(
                    messages,
                    spec.system_prompt if spec is not None else None,
                ),
                config=execution_config,
            )
            return
        artifact = self._compile(spec) if spec is not None else None
        graph = artifact.graph if artifact is not None else None
        if graph is None:
            raise ExecutionError("execution graph was not constructed")
        graph_input = (
            {"messages": messages}
            if decision.topology is TopologyName.REACT
            else {
                "task": task,
                "messages": messages,
            }
        )
        yield from graph.stream(
            graph_input,
            config=execution_config,
        )

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        if kwargs:
            raise TypeError(f"unexpected astream options: {', '.join(kwargs)}")
        await self._initialize_capabilities()
        decision, spec, task, messages, _, selection = await self._aresolve(
            input, depth=0
        )
        execution_config = self._run_config(
            config,
            threading.Event(),
            None,
            0,
            async_mode=True,
            selected_capabilities=selection.names,
        )
        if decision.mode is ExecutionMode.DIRECT:
            async for chunk in self._xai_model.astream(
                self._model_input(
                    messages,
                    spec.system_prompt if spec is not None else None,
                ),
                config=execution_config,
            ):
                yield chunk
            return
        artifact = self._compile(spec) if spec is not None else None
        graph = artifact.graph if artifact is not None else None
        if graph is None:
            raise ExecutionError("execution graph was not constructed")
        graph_input = (
            {"messages": messages}
            if decision.topology is TopologyName.REACT
            else {
                "task": task,
                "messages": messages,
            }
        )
        async for chunk in graph.astream(
            graph_input,
            config=execution_config,
        ):
            yield chunk

    def stop(self) -> None:
        """Request cooperative cancellation of this agent's active invocations."""

        with self._active_lock:
            signals = tuple(self._active)
        for signal in signals:
            signal.set()

    def flush_observability(self) -> None:
        """Flush buffered telemetry from every configured observer."""

        if self.config.explainability is not None:
            self.config.explainability.run_sync(self.config.explainability.flush())
        self.observability.flush()

    def close_observability(self) -> None:
        """Flush and release configured observer resources."""

        if self.config.explainability is not None:
            self.config.explainability.run_sync(self.config.explainability.close())
        self.observability.flush()
        self.observability.shutdown()

    async def _initialize_capabilities(self) -> None:
        async with self._mcp_init_lock:
            if self._mcp_registration is None:
                return
            runtime = self.capabilities.resolve("mcp_runtime")
            await runtime.register_mcp_client(**self._mcp_registration)
            self._register_capabilities(await self._mcp_provider.adiscover())
            self._mcp_registration = None

    def route_capabilities(self, task: str) -> CapabilitySelection:
        """Select local, application, and MCP capabilities for one task."""

        return self.capability_router.route(task, self.capabilities)

    async def aroute_capabilities(self, task: str) -> CapabilitySelection:
        """Asynchronously select capabilities for one task."""

        return await self.capability_router.aroute(task, self.capabilities)

    def execute_capability(self, name: str, input: Any = None) -> Any:
        """Execute one named capability through its standard local contract."""

        capability = self.capabilities.resolve(name)
        invoke = getattr(capability, "invoke", None)
        if callable(invoke):
            result = invoke(input)
        elif callable(capability):
            result = capability(input)
        else:
            raise CapabilityResolutionError(
                f"capability {name!r} is not synchronously executable"
            )
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise CapabilityResolutionError(
                f"capability {name!r} requires async execution"
            )
        return result

    async def aexecute_capability(self, name: str, input: Any = None) -> Any:
        """Asynchronously execute one named capability."""

        capability = self.capabilities.resolve(name)
        ainvoke = getattr(capability, "ainvoke", None)
        if callable(ainvoke):
            result = ainvoke(input)
        else:
            invoke = getattr(capability, "invoke", None)
            if callable(invoke):
                result = invoke(input)
            elif callable(capability):
                result = capability(input)
            else:
                raise CapabilityResolutionError(
                    f"capability {name!r} is not executable"
                )
        return await result if inspect.isawaitable(result) else result

    def _initialize_capabilities_sync(self) -> None:
        if self._mcp_registration is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._initialize_capabilities())
        else:
            raise ConfigurationError(
                "sync MCP execution cannot run inside an active event loop; "
                "use await agent.ainvoke(...) or agent.astream(...)"
            )

    def _invoke(
        self,
        input: Any,
        run_config: RunnableConfig | None,
        *,
        depth: int,
        session: RecursionSession | None,
        child: bool,
    ) -> Any:
        self._initialize_capabilities_sync()
        if self.config.recursion and session is None:
            session = RecursionSession(self.config.recursion_policy)
        signal = self._register()
        lifecycle = Lifecycle()
        run_id = str(uuid.uuid4())
        try:
            lifecycle.transition(LifecycleState.INITIALIZING)
            lifecycle.transition(LifecycleState.READY)
            lifecycle.transition(LifecycleState.RUNNING)
            self._emit("HarnessStarted", run_id)
            if isinstance(input, Command):
                spec = self._validate_resume()
                artifact = self._compile(spec)
                config = self._run_config(
                    run_config,
                    signal,
                    session,
                    depth,
                    checkpointer_active=True,
                )
                self._validate_command_input(input)
                if artifact is None:
                    raise ExecutionError("resume requires a compiled topology")
                result = artifact.graph.invoke(input, config=config)
                result = self._mark_checkpoint_interrupt(artifact.graph, config, result)
                return self._finish_result(
                    result, run_id, lifecycle, input, mapping_input=False
                )
            (
                decision,
                spec,
                task,
                messages,
                mapping_input,
                selection,
            ) = self._resolve(input, depth=depth, run_id=run_id)
            artifact = self._compile(spec) if spec is not None else None
            config = self._run_config(
                run_config,
                signal,
                session,
                depth,
                checkpointer_active=(decision.mode is ExecutionMode.TOPOLOGY),
                selected_capabilities=selection.names,
            )
            self._check_cancelled(signal)
            if decision.mode is ExecutionMode.DIRECT:
                self._emit("DirectExecutionStarted", run_id)
                result = self._direct_invoke(
                    messages,
                    config,
                    spec.system_prompt if spec is not None else None,
                )
            else:
                if artifact is None:
                    raise ExecutionError("topology compiler returned no graph")
                self._emit(
                    "TopologySelected",
                    run_id,
                    topology=decision.topology.value if decision.topology else None,
                )
                result = self._invoke_graph(
                    artifact.graph,
                    task,
                    messages,
                    decision.topology,
                    config,
                )
                if self.config.interrupt_before or self.config.interrupt_after:
                    result = self._mark_checkpoint_interrupt(
                        artifact.graph, config, result
                    )
            self._check_cancelled(signal)
            return self._finish_result(
                result, run_id, lifecycle, input, mapping_input, child=child
            )
        except (asyncio.CancelledError, CancellationError):
            if lifecycle.state is LifecycleState.RUNNING:
                lifecycle.transition(LifecycleState.CANCELLED)
            self._emit("ExecutionCancelled", run_id)
            raise
        except Exception as error:
            if lifecycle.state is LifecycleState.RUNNING:
                lifecycle.transition(LifecycleState.FAILED)
            self._emit("ExecutionFailed", run_id, error=type(error).__name__)
            if isinstance(error, AgloomError):
                raise
            raise ExecutionError("agent execution failed") from error
        finally:
            self._unregister(signal)

    async def _ainvoke(
        self,
        input: Any,
        run_config: RunnableConfig | None,
        *,
        depth: int,
        session: RecursionSession | None,
        child: bool,
    ) -> Any:
        await self._initialize_capabilities()
        if self.config.recursion and session is None:
            session = RecursionSession(self.config.recursion_policy)
        signal = self._register()
        lifecycle = Lifecycle()
        run_id = str(uuid.uuid4())
        try:
            lifecycle.transition(LifecycleState.INITIALIZING)
            lifecycle.transition(LifecycleState.READY)
            lifecycle.transition(LifecycleState.RUNNING)
            self._emit("HarnessStarted", run_id)
            if isinstance(input, Command):
                spec = self._validate_resume()
                artifact = self._compile(spec)
                config = self._run_config(
                    run_config,
                    signal,
                    session,
                    depth,
                    async_mode=True,
                    checkpointer_active=True,
                )
                self._validate_command_input(input)
                if artifact is None:
                    raise ExecutionError("resume requires a compiled topology")
                result = await artifact.graph.ainvoke(input, config=config)
                result = await self._amark_checkpoint_interrupt(
                    artifact.graph, config, result
                )
                return self._finish_result(
                    result, run_id, lifecycle, input, mapping_input=False
                )
            (
                decision,
                spec,
                task,
                messages,
                mapping_input,
                selection,
            ) = await self._aresolve(input, depth=depth, run_id=run_id)
            artifact = self._compile(spec) if spec is not None else None
            config = self._run_config(
                run_config,
                signal,
                session,
                depth,
                async_mode=True,
                checkpointer_active=(decision.mode is ExecutionMode.TOPOLOGY),
                selected_capabilities=selection.names,
            )
            self._check_cancelled(signal)
            if decision.mode is ExecutionMode.DIRECT:
                self._emit("DirectExecutionStarted", run_id)
                result = await self._xai_model.ainvoke(
                    self._model_input(
                        messages,
                        spec.system_prompt if spec is not None else None,
                    ),
                    config=config,
                )
            else:
                if artifact is None:
                    raise ExecutionError("topology compiler returned no graph")
                self._emit(
                    "TopologySelected",
                    run_id,
                    topology=decision.topology.value if decision.topology else None,
                )
                graph_input = (
                    {"messages": messages}
                    if decision.topology is TopologyName.REACT
                    else {"task": task, "messages": messages}
                )
                result = await artifact.graph.ainvoke(graph_input, config=config)
                if self.config.interrupt_before or self.config.interrupt_after:
                    result = await self._amark_checkpoint_interrupt(
                        artifact.graph, config, result
                    )
            self._check_cancelled(signal)
            return self._finish_result(
                result, run_id, lifecycle, input, mapping_input, child=child
            )
        except asyncio.CancelledError:
            if lifecycle.state is LifecycleState.RUNNING:
                lifecycle.transition(LifecycleState.CANCELLED)
            self._emit("ExecutionCancelled", run_id)
            raise
        except CancellationError:
            if lifecycle.state is LifecycleState.RUNNING:
                lifecycle.transition(LifecycleState.CANCELLED)
            self._emit("ExecutionCancelled", run_id)
            raise
        except Exception as error:
            if lifecycle.state is LifecycleState.RUNNING:
                lifecycle.transition(LifecycleState.FAILED)
            self._emit("ExecutionFailed", run_id, error=type(error).__name__)
            if isinstance(error, AgloomError):
                raise
            raise ExecutionError("agent execution failed") from error
        finally:
            self._unregister(signal)

    def _resolve(self, input: Any, *, depth: int, run_id: str | None = None) -> tuple[
        ExecutionDecision,
        ArchitectureSpec | None,
        str,
        list[Any],
        bool,
        CapabilitySelection,
    ]:
        task, messages, mapping_input = self._normalize_input(input)
        selection = self.route_capabilities(task)
        if self._fixed_spec is not None and depth == 0:
            return (
                self._fixed_spec.decision,
                self._fixed_spec,
                task,
                messages,
                mapping_input,
                selection,
            )

        candidates = self._analysis_candidates(depth)
        self._emit_analysis_started(run_id)
        analysis = self._analyzer.analyze(task, selection.capabilities)
        return self._resolve_analysis(
            analysis,
            candidates,
            task,
            messages,
            mapping_input,
            selection,
            depth=depth,
        )

    async def _aresolve(
        self, input: Any, *, depth: int, run_id: str | None = None
    ) -> tuple[
        ExecutionDecision,
        ArchitectureSpec | None,
        str,
        list[Any],
        bool,
        CapabilitySelection,
    ]:
        task, messages, mapping_input = self._normalize_input(input)
        selection = await self.aroute_capabilities(task)
        if self._fixed_spec is not None and depth == 0:
            return (
                self._fixed_spec.decision,
                self._fixed_spec,
                task,
                messages,
                mapping_input,
                selection,
            )

        candidates = self._analysis_candidates(depth)
        self._emit_analysis_started(run_id)
        analysis = await self._analyzer.aanalyze(task, selection.capabilities)
        return self._resolve_analysis(
            analysis,
            candidates,
            task,
            messages,
            mapping_input,
            selection,
            depth=depth,
        )

    def _resolve_analysis(
        self,
        analysis: TaskAnalysis,
        candidates: tuple[TopologyName, ...] | None,
        task: str,
        messages: list[Any],
        mapping_input: bool,
        selection: CapabilitySelection,
        *,
        depth: int,
    ) -> tuple[
        ExecutionDecision,
        ArchitectureSpec | None,
        str,
        list[Any],
        bool,
        CapabilitySelection,
    ]:
        if analysis.missing_capabilities:
            missing = ", ".join(analysis.missing_capabilities)
            raise CapabilityResolutionError(
                f"task requires unavailable capabilities: {missing}"
            )
        decision = self.strategy.decide(analysis, candidates, allow_direct=depth > 0)
        if decision.mode is ExecutionMode.TOPOLOGY and decision.topology is None:
            raise ExecutionError("strategy selected topology mode without a topology")
        composition = (
            self.config.composition or analysis.blueprint.composition
            if decision.topology is TopologyName.HYBRID
            else ()
        )
        if decision.topology is TopologyName.HYBRID and not composition:
            raise ConfigurationError(
                "strategy selected Hybrid but no Hybrid composition was configured"
            )
        if decision.mode is ExecutionMode.DIRECT and self.config.middleware:
            raise ConfigurationError(
                "LangChain middleware requires ReAct; set pattern='react' explicitly"
            )
        spec = ArchitectureSpec(
            decision=decision,
            composition=composition,
            checkpointer=self.config.checkpointer,
            workers=self.config.workers
            or tuple(
                WorkerSpec(
                    name=worker.name,
                    description=worker.description,
                    instructions=worker.instructions,
                )
                for worker in analysis.blueprint.workers[
                    : self.config.recursion_policy.max_workers
                ]
            ),
            stages=self.config.stages
            or tuple(
                PipelineStage(name=stage.name, instruction=stage.instruction)
                for stage in analysis.blueprint.stages
            ),
            system_prompt=(
                self.config.system_prompt or analysis.blueprint.instructions or None
            ),
        )
        worker_topologies = {
            TopologyName.SUPERVISOR,
            TopologyName.SWARM,
            TopologyName.BLACKBOARD,
        }
        requires_workers = decision.topology in worker_topologies or (
            decision.topology is TopologyName.HYBRID
            and any(child in worker_topologies for child in composition)
        )
        if requires_workers and not spec.workers:
            raise TopologySelectionError(
                "analyzer selected an execution topology that requires workers "
                "but did not provide a worker blueprint"
            )
        return decision, spec, task, messages, mapping_input, selection

    def _analysis_candidates(self, depth: int) -> tuple[TopologyName, ...] | None:
        if depth > 0 and self.config.recursion:
            return self.config.recursion_policy.allowed_nested_topologies
        return self.config.pattern_candidates

    def _emit_analysis_started(self, run_id: str | None) -> None:
        if (
            self.config.pattern_candidates is None
            or len(self.config.pattern_candidates) > 1
        ):
            self._emit("TopologySelectionStarted", run_id or str(uuid.uuid4()))

    def _compile(self, spec: ArchitectureSpec) -> TopologyArtifact | None:
        topology = spec.decision.topology
        if topology is None:
            return None
        key = (
            topology,
            spec.composition,
            spec.workers,
            tuple((stage.name, stage.instruction) for stage in spec.stages),
            spec.system_prompt,
        )
        with self._cache_lock:
            artifact = self._artifacts.get(key)
            if artifact is None:
                artifact = self.compiler.compile(spec, self.config)
                if artifact is None:
                    raise ExecutionError("topology compiler returned no artifact")
                self._artifacts[key] = artifact
            return artifact

    def _direct_invoke(
        self,
        messages: list[Any],
        config: RunnableConfig | None,
        system_prompt: str | None,
    ):
        return self._xai_model.invoke(
            self._model_input(messages, system_prompt),
            config=config,
        )

    def _model_input(
        self,
        messages: list[Any],
        system_prompt: str | None = None,
    ) -> list[Any]:
        resolved_prompt = self.config.system_prompt or system_prompt
        if resolved_prompt:
            return [SystemMessage(content=resolved_prompt), *messages]
        return messages

    def _invoke_graph(
        self,
        graph: CompiledStateGraph,
        task: str,
        messages: list[Any],
        topology: TopologyName | None,
        config: RunnableConfig,
    ):
        graph_input = (
            {"messages": messages}
            if topology is TopologyName.REACT
            else {"task": task, "messages": messages}
        )
        return graph.invoke(graph_input, config=config)

    def _run_config(
        self,
        caller_config: RunnableConfig | None,
        signal: threading.Event,
        session: RecursionSession | None,
        depth: int,
        *,
        async_mode: bool = False,
        checkpointer_active: bool = False,
        selected_capabilities: tuple[str, ...] = (),
    ) -> RunnableConfig:
        result = cast(RunnableConfig, dict(caller_config or {}))
        raw_configurable = result.get("configurable")
        configurable: dict[str, Any] = (
            dict(raw_configurable) if isinstance(raw_configurable, Mapping) else {}
        )
        configurable["agloom_cancel_event"] = signal
        configurable["agloom_max_handoffs"] = self.config.recursion_policy.max_subtasks
        configurable["agloom_selected_capabilities"] = selected_capabilities
        if checkpointer_active and self.config.checkpointer is not None:
            self._authorize_checkpoint_thread(configurable.get("thread_id"))
        if self.config.recursion and session is not None:
            callback_name = (
                "agloom_achild_executor" if async_mode else "agloom_child_executor"
            )
            configurable[callback_name] = self._child_executor(
                caller_config, session, depth, async_mode=async_mode
            )
        result["configurable"] = configurable
        callbacks = self.observability.callback_handlers
        if callbacks:
            callback_manager = CallbackManager.configure(
                inheritable_callbacks=result.get("callbacks")
            )
            for callback in callbacks:
                callback_manager.add_handler(callback, inherit=True)
            result["callbacks"] = callback_manager
        return cast(RunnableConfig, result)

    def _register_capabilities(
        self,
        descriptors: Sequence[CapabilityDescriptor],
    ) -> None:
        for descriptor in descriptors:
            try:
                self.capabilities.register(descriptor)
            except ValueError as error:
                raise ConfigurationError(str(error)) from error

    def _child_executor(
        self,
        caller_config: RunnableConfig | None,
        session: RecursionSession,
        depth: int,
        *,
        async_mode: bool,
    ) -> Callable[..., Any]:
        def run(task: str):
            with session.child(depth + 1):
                session.check()
                child_config = _child_config(caller_config)
                result = self._invoke(
                    task,
                    child_config,
                    depth=depth + 1,
                    session=session,
                    child=True,
                )
                session.check()
                return result

        async def arun(task: str):
            with session.child(depth + 1):
                session.check()
                child_config = _child_config(caller_config)
                result = await self._ainvoke(
                    task,
                    child_config,
                    depth=depth + 1,
                    session=session,
                    child=True,
                )
                session.check()
                return result

        return arun if async_mode else run

    def _normalize_input(self, input: Any) -> tuple[str, list[Any], bool]:
        if isinstance(input, str):
            task = input.strip()
            if not task:
                raise ConfigurationError("task input must not be empty")
            return task, [HumanMessage(content=task)], False
        if isinstance(input, BaseMessage):
            messages = [input]
            self._validate_input_messages(messages)
            task = _latest_user_content(messages)
            return task, messages, False
        if isinstance(input, Sequence) and not isinstance(input, (str, bytes, dict)):
            messages = list(input)
            self._validate_input_messages(messages)
            task = _latest_user_content(messages)
            return task, messages, False
        if isinstance(input, dict):
            raw_messages = input.get("messages")
            if raw_messages is not None:
                messages = list(raw_messages)
                self._validate_input_messages(messages)
                task = _latest_user_content(messages)
            else:
                task_value = input.get("task", input.get("input", input.get("prompt")))
                if task_value is None:
                    raise ConfigurationError(
                        "input mapping requires messages, task, input, or prompt"
                    )
                task = str(task_value)
                messages = [HumanMessage(content=task)]
            if not task.strip():
                raise ConfigurationError("task input must not be empty")
            return task, messages, True
        raise ConfigurationError(
            f"unsupported input type {type(input).__name__}; expected text or messages"
        )

    @staticmethod
    def _validate_input_messages(messages: list[Any]) -> None:
        for message in messages:
            role, _ = _message_role_and_content(message)
            if role not in {"human", "user", "ai", "assistant"}:
                raise ConfigurationError(
                    "untrusted messages may use only user or assistant roles; "
                    "configure trusted instructions with system_prompt"
                )

    def _validate_command_input(self, command: Command[Any]) -> None:
        update = command.update
        if update is None:
            return
        if not isinstance(update, Mapping):
            raise ConfigurationError("Command state updates must be mappings")
        if "messages" not in update:
            return
        messages = update["messages"]
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ConfigurationError("Command message updates must be a message list")
        self._validate_input_messages(list(messages))

    def _shape_result(self, result: Any, input: Any, mapping_input: bool) -> Any:
        if not mapping_input:
            if isinstance(result, dict) and "output" in result:
                return AIMessage(content=str(result["output"]))
            if isinstance(result, dict) and result.get("messages"):
                return result["messages"][-1]
            return result
        if isinstance(result, dict):
            return result
        if isinstance(result, BaseMessage):
            if isinstance(input, dict) and input.get("messages") is not None:
                return {**input, "messages": [*input["messages"], result]}
            return {"messages": [HumanMessage(content=str(input)), result]}
        return result

    def _finish_result(
        self,
        result: Any,
        run_id: str,
        lifecycle: Lifecycle,
        input: Any,
        mapping_input: bool,
        *,
        child: bool = False,
    ) -> Any:
        if isinstance(result, dict) and result.get("__interrupt__"):
            lifecycle.transition(LifecycleState.PAUSED)
            self._emit("HumanApprovalRequired", run_id)
            self._emit("ExecutionPaused", run_id)
            return result
        lifecycle.transition(LifecycleState.COMPLETED)
        self._emit("ExecutionCompleted", run_id)
        shaped = self._shape_result(result, input, mapping_input)
        if child and isinstance(shaped, AIMessage):
            return shaped.content
        return shaped

    def _validate_resume(self) -> ArchitectureSpec:
        if self._fixed_spec is None:
            raise ConfigurationError(
                "resume requires an agent with one explicit topology"
            )
        if self.config.checkpointer is None:
            raise ConfigurationError("resume requires a configured checkpointer")
        return self._fixed_spec

    def get_state(self, config: RunnableConfig) -> Any:
        spec = self._validate_resume()
        self._authorize_checkpoint_thread(_configurable_thread_id(config))
        artifact = self._compile(spec)
        if artifact is None:
            raise ExecutionError("compiled topology graph is unavailable")
        return artifact.graph.get_state(config)

    def _authorize_checkpoint_thread(self, thread_id: Any) -> None:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ConfigurationError(
                "a nonempty configurable.thread_id is required with a checkpointer"
            )
        authorizer = self.config.checkpoint_authorizer
        if authorizer is None:
            raise ConfigurationError(
                "checkpoint access requires a checkpoint_authorizer that validates "
                "the current caller's thread_id"
            )
        if not authorizer(thread_id):
            raise ConfigurationError("checkpoint access is not authorized")

    def _mark_checkpoint_interrupt(
        self, graph: CompiledStateGraph, config: RunnableConfig, result: Any
    ) -> Any:
        if not self.config.checkpointer:
            return result
        snapshot = graph.get_state(config)
        if snapshot.next and isinstance(result, dict):
            return {**result, "__interrupt__": snapshot.next}
        return result

    async def _amark_checkpoint_interrupt(
        self, graph: CompiledStateGraph, config: RunnableConfig, result: Any
    ) -> Any:
        if not self.config.checkpointer:
            return result
        snapshot = await graph.aget_state(config)
        if snapshot.next and isinstance(result, dict):
            return {**result, "__interrupt__": snapshot.next}
        return result

    def _register(self) -> threading.Event:
        signal = threading.Event()
        with self._active_lock:
            self._active.add(signal)
        return signal

    def _unregister(self, signal: threading.Event) -> None:
        with self._active_lock:
            self._active.discard(signal)

    @staticmethod
    def _check_cancelled(signal: threading.Event) -> None:
        if signal.is_set():
            raise CancellationError("execution was cooperatively cancelled")

    def _emit(self, name: str, run_id: str, **payload: Any) -> None:
        self.observability.on_event(
            AgentEvent(
                name=name,
                agent_id=self.id,
                run_id=run_id,
                topology=payload.pop("topology", None),
                payload=payload,
            )
        )


def _latest_user_content(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        role, content = _message_role_and_content(message)
        if role in {"human", "user"}:
            return str(content)
    raise ConfigurationError("messages input must contain at least one message")


def _message_role_and_content(message: Any) -> tuple[str, Any]:
    if isinstance(message, BaseMessage):
        return message.type, message.content
    if isinstance(message, Mapping):
        role = message.get("role", message.get("type"))
        if isinstance(role, str) and "content" in message:
            return role.lower(), message["content"]
    if isinstance(message, (tuple, list)) and len(message) == 2:
        role, content = message
        if isinstance(role, str):
            return role.lower(), content
    raise ConfigurationError(
        "messages must be LangChain messages, role/content mappings, or "
        "(role, content) pairs"
    )


def _configurable_thread_id(config: RunnableConfig) -> Any:
    configurable = config.get("configurable")
    if isinstance(configurable, Mapping):
        return configurable.get("thread_id")
    return None


def _child_config(config: RunnableConfig | None) -> RunnableConfig | None:
    if config is None:
        return None
    result = cast(RunnableConfig, dict(config))
    raw_configurable = result.get("configurable")
    configurable: dict[str, Any] = (
        dict(raw_configurable) if isinstance(raw_configurable, Mapping) else {}
    )
    if configurable.get("thread_id"):
        configurable["thread_id"] = f"{configurable['thread_id']}:child:{uuid.uuid4()}"
    result["configurable"] = configurable
    return cast(RunnableConfig, result)
