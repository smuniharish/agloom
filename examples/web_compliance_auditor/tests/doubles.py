"""Deterministic doubles for offline tests: no model, database, or network."""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from backend.governance import POLICY_SEARCH
from backend.mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT
from backend.models import Audit, ComplianceRecord, Evidence
from fastmcp import FastMCP
from langchain_core.callbacks import CallbackManagerForLLMRun
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
UNREACHABLE = "unreachable.test"


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
    return AIMessage(content=f"Audit complete.\n{envelope}")


def report(
    verdict: str = "fail",
    policy_ids: Sequence[str] = ("WEB-ACCESS-001", "WEB-PRIVACY-002"),
) -> dict[str, Any]:
    """A report payload with one finding per policy ID."""
    return {
        "verdict": verdict,
        "summary": "The page collects personal data without a notice or label.",
        "findings": [
            {
                "policy_id": policy_id,
                "requirement": f"Requirement of {policy_id}",
                "status": verdict,
                "rationale": "Observed in the page snapshot.",
                "evidence_ids": [policy_id],
            }
            for policy_id in policy_ids
        ],
        "limitations": [],
    }


def auditing(final: Mapping[str, Any]) -> Callable[[Call], AIMessage]:
    """Search the policies once, then answer with `final`."""

    def respond(call: Call) -> AIMessage:
        if call.last.type == "tool":
            return structured(final)
        return tool_call(POLICY_SEARCH, {"query": "form labels privacy notice"})

    return respond


class HashEmbeddings(Embeddings):
    """Deterministic 384-dimensional word-hash embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 383 + [0.01]
        for word in text.casefold().split():
            digest = hashlib.sha256(word.strip(".,:#/").encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % 383] += 1.0
        return vector


class EventLog(list[AgentEvent]):
    """An observer that records Agloom events."""

    def __call__(self, event: AgentEvent) -> None:
        self.append(event)

    @property
    def names(self) -> list[str]:
        return [str(event.name) for event in self]


class MemoryPolicies:
    """An in-memory policy index returning every matching policy."""

    def __init__(self) -> None:
        self.policies: dict[str, tuple[str, str]] = {}
        self.upserts: list[str] = []
        self.deletes: list[str] = []

    def upsert(self, policy_id: str, title: str, body: str) -> None:
        self.policies[policy_id] = (title, body)
        self.upserts.append(policy_id)

    def delete(self, policy_id: str) -> None:
        self.policies.pop(policy_id, None)
        self.deletes.append(policy_id)

    def search(
        self, query: str, policy_ids: Sequence[str] = (), limit: int = 8
    ) -> list[Evidence]:
        del query
        return [
            Evidence(policy_id=policy_id, title=title, excerpt=body, similarity=0.5)
            for policy_id, (title, body) in sorted(self.policies.items())
            if not policy_ids or policy_id in policy_ids
        ][:limit]


class MemoryAudits:
    """In-memory audit records."""

    def __init__(self) -> None:
        self.items: dict[UUID, Audit] = {}

    async def save(self, audit: Audit) -> None:
        self.items[audit.audit_id] = audit

    async def get(self, audit_id: UUID) -> Audit | None:
        return self.items.get(audit_id)

    async def update_record(
        self, audit_id: UUID, record: ComplianceRecord
    ) -> Audit | None:
        current = self.items.get(audit_id)
        if current is None:
            return None
        updated = current.model_copy(update={"record": record})
        self.items[audit_id] = updated
        return updated


class MemoryRegister:
    """Records every publication, so tests can count side effects."""

    description = "Publish a grounded audit verdict to the compliance register."

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
            "audit_id": arguments["audit_id"],
            "verdict": arguments["verdict"],
            "recorded_at": datetime.now(UTC).isoformat(),
        }


class FakeServers:
    """In-process MCP servers that mirror the real servers' tool names."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.writes: list[str] = []

    def targets(self) -> dict[str, FastMCP]:
        return {
            PLAYWRIGHT: self._playwright(),
            FILESYSTEM: self._filesystem(),
            EVERYTHING: self._everything(),
        }

    def _playwright(self) -> FastMCP:
        server = FastMCP("playwright")

        @server.tool
        def browser_navigate(url: str) -> str:
            """Open a page."""
            if UNREACHABLE in url:
                raise ConnectionError(f"net::ERR_NAME_NOT_RESOLVED at {url}")
            self.opened.append(url)
            return f"### Page\n- Page URL: {url}"

        @server.tool
        def browser_snapshot(filename: str | None = None) -> str:
            """Capture the open page."""
            del filename
            return (
                f"### Page\n- Page URL: {self.opened[-1]}\n"
                "- Page Title: Benefits Portal\n"
                '```yaml\n- textbox "Employee email"\n- button "Continue"\n'
                "- paragraph: Analytics cookies are enabled by default.\n```"
            )

        @server.tool
        def browser_run_code_unsafe(code: str) -> str:
            """Run code in the page."""
            return code

        return server

    def _filesystem(self) -> FastMCP:
        server = FastMCP("filesystem")

        @server.tool
        def list_allowed_directories() -> str:
            """List the allowed directories."""
            return "Allowed directories:\n/workspace"

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
