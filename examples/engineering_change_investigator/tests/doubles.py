"""Deterministic doubles for offline tests: no model, database, or network."""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.governance import EVIDENCE_SEARCH
from backend.mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT
from backend.models import Investigation, ReleaseDecision
from fastmcp import FastMCP
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, PrivateAttr

from agloom import AgentEvent

_IDS = itertools.count(1)


class Call:
    """One model call: the messages and the tools offered with it."""

    def __init__(self, messages: Sequence[BaseMessage], tools: Sequence[str]) -> None:
        self.messages = list(messages)
        self.tools = tuple(tools)

    @property
    def last(self) -> BaseMessage:
        return self.messages[-1]

    @property
    def prompt(self) -> str:
        return "\n".join(message.text for message in self.messages)


class ScriptedChatModel(BaseChatModel):
    """A chat model whose replies come from a responder function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Callable[[Call], AIMessage | str]
    _calls: list[Call] = PrivateAttr(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[Call]:
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
        response = self.responder(call)
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
        del tool_choice
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs
        )


def tool_call(name: str, args: Mapping[str, Any] | None = None) -> AIMessage:
    """An AI message that calls one tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": dict(args or {}), "id": f"call_{next(_IDS)}"}
        ],
    )


def structured(payload: Mapping[str, Any]) -> AIMessage:
    """An AI message carrying an xstructured report envelope."""
    envelope = f"<xstructured>{json.dumps(payload)}</xstructured>"
    return AIMessage(content=f"Assessment complete.\n{envelope}")


def report(
    decision: str = "no_go",
    citations: Sequence[str] = ("[ADR-101]", "[RELEASE-301]"),
) -> dict[str, Any]:
    """A report payload citing the given evidence IDs."""
    return {
        "decision": decision,
        "risk": "high" if decision == "no_go" else "low",
        "summary": "Release 4.9 reduces the pool below the documented minimum.",
        "evidence": [
            {"claim": f"Claim supported by {citation}.", "citation": citation}
            for citation in citations
        ],
        "recommendation": "Hold the release until the load test is approved.",
    }


def investigating(final: Mapping[str, Any]) -> Callable[[Call], AIMessage]:
    """Search the evidence once, then answer with `final`."""

    def respond(call: Call) -> AIMessage:
        if call.last.type == "tool":
            return structured(final)
        return tool_call(EVIDENCE_SEARCH, {"query": "checkout connection pool"})

    return respond


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-keywords embeddings."""

    VOCABULARY = (
        "release",
        "pool",
        "connection",
        "database",
        "load",
        "test",
        "runbook",
        "rollout",
        "file",
        "directory",
        "search",
        "evidence",
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [float(lowered.count(word)) for word in self.VOCABULARY] + [0.01]


class HashEmbeddings(Embeddings):
    """Deterministic 384-dimensional word-hash embeddings for the pgvector table."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 383 + [0.01]
        for word in text.casefold().split():
            digest = hashlib.sha256(word.strip(".,:#").encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % 383] += 1.0
        return vector


class EventLog(list[AgentEvent]):
    """An observer that records Agloom events."""

    def __call__(self, event: AgentEvent) -> None:
        self.append(event)

    @property
    def names(self) -> list[str]:
        return [str(event.name) for event in self]


class MemoryEvidence:
    """An in-memory evidence index ranked by shared words."""

    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, str, str]] = {}
        self.upserts: list[str] = []
        self.deletes: list[str] = []

    def upsert(self, document_id: str, title: str, source: str, content: str) -> None:
        self.documents[document_id] = (title, source, content)
        self.upserts.append(document_id)

    def delete(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        self.deletes.append(document_id)

    def search(self, query: str, limit: int = 5) -> list[Document]:
        terms = set(query.casefold().split())
        ranked = sorted(
            self.documents.items(),
            key=lambda item: -len(terms & set(item[1][2].casefold().split())),
        )
        return [
            Document(
                page_content=content,
                metadata={
                    "document_id": document_id,
                    "title": title,
                    "source": source,
                    "score": 0.5,
                    "capability_name": EVIDENCE_SEARCH,
                },
            )
            for document_id, (title, source, content) in ranked[:limit]
        ]


class MemoryInvestigations:
    """In-memory investigation records."""

    def __init__(self) -> None:
        self.items: dict[UUID, Investigation] = {}

    async def save(self, investigation: Investigation) -> None:
        self.items[investigation.investigation_id] = investigation

    async def get(self, investigation_id: UUID) -> Investigation | None:
        return self.items.get(investigation_id)

    async def update_decision(
        self, investigation_id: UUID, decision: ReleaseDecision
    ) -> Investigation | None:
        current = self.items.get(investigation_id)
        if current is None:
            return None
        updated = current.model_copy(update={"release_decision": decision})
        self.items[investigation_id] = updated
        return updated


class MemoryReleaseRegister:
    """Records every publication, so tests can count side effects."""

    description = "Publish a validated go or no-go release decision."

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.attempts = 0
        self.failure: Exception | None = None

    async def ainvoke(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        if self.failure is not None:
            raise self.failure
        self.published.append(dict(arguments))
        return {
            "investigation_id": arguments["investigation_id"],
            "decision": arguments["decision"],
            "recorded_at": datetime.now(UTC).isoformat(),
        }


class FakeServers:
    """In-process MCP servers that mirror the real servers' tool names."""

    def __init__(self) -> None:
        self.page_url: str | None = None
        self.writes: list[str] = []
        self.clicks: list[str] = []

    def targets(self) -> dict[str, FastMCP]:
        return {
            FILESYSTEM: self._filesystem(),
            EVERYTHING: self._everything(),
            PLAYWRIGHT: self._playwright(),
        }

    def _filesystem(self) -> FastMCP:
        server = FastMCP("filesystem")

        @server.tool
        def list_directory(path: str) -> str:
            """List a directory."""
            return "\n".join(
                f"[FILE] {item.name}" for item in sorted(Path(path).iterdir())
            )

        @server.tool
        def read_text_file(path: str) -> str:
            """Read a text file."""
            return Path(path).read_text(encoding="utf-8")

        @server.tool
        def write_file(path: str, content: str) -> str:
            """Write a file."""
            self.writes.append(path)
            return f"wrote {len(content)} characters"

        return server

    def _everything(self) -> FastMCP:
        server = FastMCP("everything")

        @server.tool
        def echo(message: str) -> str:
            """Echo a message."""
            return f"Echo: {message}"

        @server.tool(name="get-env")
        def get_env() -> str:
            """Print the environment."""
            return "SECRET=unsafe"

        @server.resource("demo://resource/static/document/architecture.md")
        def architecture() -> str:
            """The reference architecture document."""
            return "# Everything Server Architecture"

        @server.prompt(name="simple-prompt")
        def simple_prompt() -> str:
            """A prompt without arguments."""
            return "This is a simple prompt without arguments."

        return server

    def _playwright(self) -> FastMCP:
        server = FastMCP("playwright")

        @server.tool
        def browser_navigate(url: str) -> str:
            """Open a page."""
            self.page_url = url
            return f"### Page\n- Page URL: {url}"

        @server.tool
        def browser_snapshot(filename: str | None = None) -> str:
            """Capture the open page."""
            del filename
            return (
                f"### Page\n- Page URL: {self.page_url}\n- Page Title: Release Status\n"
                '```yaml\n- paragraph: Approval pending\n- paragraph: "Database load '
                'test: missing"\n```'
            )

        @server.tool
        def browser_click(element: str) -> str:
            """Click an element."""
            self.clicks.append(element)
            return "clicked"

        return server


def copy_workspace(source: Path, destination: Path, names: Iterable[str]) -> Path:
    """Copy selected workspace documents into a writable folder."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        (destination / name).write_text(
            (source / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    return destination
