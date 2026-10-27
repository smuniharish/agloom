"""Deterministic test doubles shared by the test suite."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from agloom import AgentEvent

Responder = Callable[["Call"], AIMessage | str]

_IDS = itertools.count(1)


class Call:
    """One model call seen by `ScriptedChatModel`."""

    def __init__(self, messages: Sequence[BaseMessage], tools: Sequence[str]) -> None:
        self.messages = list(messages)
        self.tools = tuple(tools)

    @property
    def system(self) -> str:
        """The text of the system message, if any."""
        for message in self.messages:
            if message.type == "system":
                return message.text
        return ""

    @property
    def prompt(self) -> str:
        """The text of every message, joined."""
        return "\n".join(message.text for message in self.messages)

    @property
    def last(self) -> BaseMessage:
        """The latest message."""
        return self.messages[-1]

    def wants(self, tool: str) -> bool:
        """Whether a tool or structured-output schema with this name is offered."""
        return tool in self.tools


class ScriptedChatModel(BaseChatModel):
    """A thread-safe chat model whose answers come from a responder or a queue."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Responder | None = None
    queue: list[Any] = Field(default_factory=list)
    _calls: list[Call] = PrivateAttr(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[Call]:
        """Every call made so far."""
        with self._lock:
            return list(self._calls)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        names = [
            tool["function"]["name"]
            for tool in kwargs.get("tools") or []
            if isinstance(tool, Mapping) and "function" in tool
        ]
        call = Call(messages, names)
        with self._lock:
            self._calls.append(call)
            if self.responder is None:
                if not self.queue:
                    raise AssertionError("the scripted model has no response left")
                queued = self.queue.pop(0)
        # Responders run outside the lock, so parallel branches really overlap.
        response = self.responder(call) if self.responder is not None else queued
        message = (
            response if isinstance(response, AIMessage) else AIMessage(content=response)
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)


def tool_call(
    name: str, args: Mapping[str, Any] | None = None, content: str = ""
) -> AIMessage:
    """Return an AI message that calls one tool or structured-output schema."""
    return AIMessage(
        content=content,
        tool_calls=[
            {"name": name, "args": dict(args or {}), "id": f"call_{next(_IDS)}"}
        ],
    )


def analysis(**fields: Any) -> AIMessage:
    """Return a structured `TaskAnalysis` response."""
    return tool_call("TaskAnalysis", {"complexity": 1, "estimated_steps": 1, **fields})


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-keywords embeddings.

    A constant component keeps every vector non-zero, like real dense embeddings.
    """

    def __init__(self, vocabulary: Iterable[str]) -> None:
        self.vocabulary = tuple(vocabulary)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        lowered = text.casefold()
        return [float(lowered.count(word)) for word in self.vocabulary] + [0.01]


class EventLog(list[AgentEvent]):
    """Records Agloom events; usable directly as an observer."""

    def __call__(self, event: AgentEvent) -> None:
        self.append(event)

    @property
    def names(self) -> list[str]:
        """The recorded event names in order."""
        return [str(event.name) for event in self]


def allow_threads(thread_id: str) -> bool:
    """A checkpoint authorizer admitting every non-empty thread."""
    return bool(thread_id)
