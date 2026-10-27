"""Topology build contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agloom._decisions import structured_decision
from agloom._runtime import tools_withheld
from agloom.models import (
    ArchitectureSpec,
    ExecutionLimits,
    PipelineStage,
    TopologyName,
    WorkerSpec,
)


@dataclass(frozen=True)
class TopologyBuildContext:
    """Everything a topology builder needs to construct its graph.

    Attributes:
        model: The chat model.
        tools: The agent's tools, already filtered by policy visibility.
        middleware: The middleware stack applied to every tool-capable agent.
        workers: Workers for Supervisor, Swarm, and Blackboard.
        stages: Stages for Pipeline.
        system_prompt: Trusted instructions prepended to every model call.
        limits: Execution limits.
        max_reflections: Maximum Reflection review rounds.
        composition: Child topologies for Hybrid.
        interrupt_before: Node names to pause before (requires a checkpointer).
        interrupt_after: Node names to pause after (requires a checkpointer).
        compile_child: Compiles a child topology for Hybrid.
        screen_answers: Whether final answers must pass through the middleware
            stack; true when application middleware is configured.
    """

    model: BaseChatModel
    tools: tuple[BaseTool, ...]
    middleware: tuple[AgentMiddleware[Any, Any], ...]
    workers: tuple[WorkerSpec, ...]
    stages: tuple[PipelineStage, ...]
    system_prompt: str | None
    limits: ExecutionLimits
    max_reflections: int
    composition: tuple[TopologyName, ...] = ()
    interrupt_before: tuple[str, ...] = ()
    interrupt_after: tuple[str, ...] = ()
    compile_child: Callable[[TopologyName], Runnable[Any, Any]] | None = None
    screen_answers: bool = False

    def specialize(
        self,
        spec: ArchitectureSpec,
        compile_child: Callable[[TopologyName], Runnable[Any, Any]] | None,
    ) -> TopologyBuildContext:
        """Return a context for one architecture specification."""
        return replace(
            self,
            workers=spec.workers,
            stages=spec.stages,
            system_prompt=spec.system_prompt,
            composition=spec.composition,
            compile_child=compile_child,
        )

    def child(self) -> TopologyBuildContext:
        """Return the context used to compile a Hybrid child topology."""
        return replace(
            self,
            composition=(),
            interrupt_before=(),
            interrupt_after=(),
            compile_child=None,
        )

    def agent(
        self, instruction: str, *, name: str, response_format: Any = None
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build a tool-capable LangChain agent that uses the middleware stack."""
        return langchain_agent(
            self.model,
            name=name,
            tools=list(self.tools),
            system_prompt=compose_prompt(self.system_prompt, instruction),
            middleware=list(self.middleware),
            response_format=response_format,
        )

    def decision(self, schema: type[Any], *, name: str) -> Runnable[Any, Any]:
        """Return a structured-output runnable for an internal routing decision."""
        return structured_decision(self.model, schema, name=name)

    def writer(self, instruction: str, *, name: str) -> Runnable[str, AIMessage]:
        """Return a runnable that writes a final answer from a prompt.

        With `screen_answers`, a tool-less agent turn writes the answer, so the
        middleware stack (for example PII redaction) covers it; otherwise the
        answer is one plain model call.
        """
        system = compose_prompt(self.system_prompt, instruction)
        if not self.screen_answers:
            prompt_to_messages = RunnableLambda(
                lambda prompt: _messages(system, prompt), name=name
            )
            return cast(Runnable[str, AIMessage], prompt_to_messages | self.model)
        agent = langchain_agent(
            self.model,
            name=name,
            tools=[],
            system_prompt=system,
            middleware=list(self.middleware),
        )

        def write(prompt: str, config: RunnableConfig) -> AIMessage:
            with tools_withheld():
                return _final(agent.invoke(_prompt(prompt), config))

        async def awrite(prompt: str, config: RunnableConfig) -> AIMessage:
            with tools_withheld():
                return _final(await agent.ainvoke(_prompt(prompt), config))

        return RunnableLambda(write, afunc=awrite, name=name)

    def compile_options(self) -> dict[str, Any]:
        """Keyword arguments for `StateGraph.compile`."""
        return {
            "interrupt_before": list(self.interrupt_before) or None,
            "interrupt_after": list(self.interrupt_after) or None,
        }


@dataclass(frozen=True)
class TopologyArtifact:
    """A compiled topology graph."""

    name: TopologyName
    graph: Runnable[Any, Any]


class Topology(Protocol):
    """Builds one topology graph. Runtime concerns stay outside this contract."""

    name: TopologyName

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the topology for one build context."""
        ...


def compose_prompt(*parts: str | None) -> str | None:
    """Join non-empty prompt parts with blank lines."""
    joined = "\n\n".join(part.strip() for part in parts if part and part.strip())
    return joined or None


def langchain_agent(
    model: BaseChatModel, *, name: str, **options: Any
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a LangChain agent that leaves model responses unchanged.

    LangChain copies a named agent's name onto every `AIMessage` it produces,
    and several OpenAI-compatible providers reject `name` on assistant
    messages. The name is kept as LangChain's `lc_agent_name` metadata, which
    identifies the agent in traces and streamed chunks.
    """
    return create_agent(model, **options).with_config(metadata={"lc_agent_name": name})


def _messages(system: str | None, prompt: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system)] if system else []
    messages.append(HumanMessage(content=prompt))
    return messages


def _prompt(prompt: str) -> dict[str, Any]:
    return {"messages": [HumanMessage(content=prompt)]}


def _final(result: Any) -> AIMessage:
    for message in reversed(result.get("messages") or ()):
        if isinstance(message, AIMessage) and message.text:
            return message
    return AIMessage(content="")
