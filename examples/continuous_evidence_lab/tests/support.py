"""Deterministic test doubles: a scripted chat model, an in-memory repository,
in-process MCP servers, and hashing embeddings."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from backend.agent import WorkspaceRefreshSource
from backend.auth import Principal, current_principal
from backend.models import (
    Accuracy,
    Approval,
    Iteration,
    RunRequest,
    RunView,
    Source,
    Synthesis,
    empty_accuracy,
)
from backend.store import Enqueue, RunState
from fastmcp import FastMCP
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

TEST_PRINCIPAL = Principal("test-user", "test-tenant")
AUTHORIZATION = {"Authorization": "Bearer test-token"}

SOURCES = WorkspaceRefreshSource().load()

_IDS = itertools.count(1)
Responder = Callable[[list[BaseMessage], tuple[str, ...]], AIMessage | str]


def tool_call(name: str, args: Mapping[str, Any] | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": dict(args or {}), "id": f"call_{next(_IDS)}"}
        ],
    )


class ScriptedChatModel(BaseChatModel):
    """A chat model whose answers come from a responder; records every call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Responder
    calls: list[tuple[list[BaseMessage], tuple[str, ...]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _respond(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        names = tuple(
            tool["function"]["name"]
            for tool in kwargs.get("tools") or []
            if isinstance(tool, Mapping) and "function" in tool
        )
        self.calls.append((list(messages), names))
        response = self.responder(list(messages), names)
        message = (
            response if isinstance(response, AIMessage) else AIMessage(content=response)
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._respond(messages, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._respond(messages, **kwargs)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs
        )


def synthesis(
    topics: Sequence[str] = ("encryption", "incident response"),
    *,
    invalid_quote: bool = False,
) -> dict[str, Any]:
    """A synthesis payload citing `SOURCES`; optionally with a bad citation."""
    quotes = {
        "encryption": ("C1", "VENDOR-SEC-001", "at rest with AES-256"),
        "incident response": ("C2", "VENDOR-IR-002", "within 72 hours"),
    }
    claims = []
    for topic in topics:
        claim_id, source_id, quote = quotes[topic]
        claims.append(
            {
                "id": claim_id,
                "topic": topic,
                "text": f"Acme evidence covers {topic}.",
                "status": "supported",
                "citations": [
                    {
                        "source_id": source_id,
                        "quote": "an invented quote" if invalid_quote else quote,
                    }
                ],
            }
        )
    return {
        "summary": "Acme evidence summary.",
        "claims": claims,
        "covered_topics": list(topics),
        "limitations": [],
    }


@dataclass
class EvidenceScript:
    """Drives the ReAct loop: feedback -> publish -> answer, one draft per iteration."""

    drafts: list[dict[str, Any]]
    browse: str | None = None
    published: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, messages: list[BaseMessage], tools: tuple[str, ...]
    ) -> AIMessage | str:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            if last.name == "get_validation_feedback":
                if self.browse is not None:
                    url, self.browse = self.browse, None
                    return tool_call("evidence-browser__browser_navigate", {"url": url})
                feedback = json.loads(str(last.content))
                index = min(int(feedback["iteration"]), len(self.drafts) - 1)
                return tool_call("publish_evidence", self.drafts[index])
            if last.name == "publish_evidence":
                draft = self._latest_publication(messages)
                return json.dumps(draft)
            return tool_call("get_validation_feedback")
        return tool_call("get_validation_feedback")

    @staticmethod
    def _latest_publication(messages: list[BaseMessage]) -> dict[str, Any]:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if call["name"] == "publish_evidence":
                        return dict(call["args"])
        return {}


class HashEmbeddings:
    """Deterministic 384-dimensional bag-of-words embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            for word in text.casefold().split():
                digest = hashlib.sha256(word.encode()).digest()
                vector[int.from_bytes(digest[:2], "big") % 384] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def fake_mcp_servers() -> dict[str, FastMCP]:
    """In-process servers exposing the tool names of the real servers."""
    filesystem = FastMCP("filesystem")
    everything = FastMCP("everything")
    browser = FastMCP("browser")
    visits: list[str] = []

    @filesystem.tool
    def list_allowed_directories() -> str:
        """List the directories the server may read."""
        return "Allowed directories: /workspace"

    @filesystem.tool
    def read_text_file(path: str) -> str:
        """Read a text file."""
        return f"contents of {path}"

    @filesystem.tool
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        raise AssertionError("write_file must never run")

    @everything.tool
    def echo(message: str) -> str:
        """Echo a message."""
        return f"Echo: {message}"

    @everything.tool(name="get-env")
    def get_env() -> str:
        """Print the environment."""
        raise AssertionError("get-env must never run")

    @browser.tool
    def browser_navigate(url: str) -> str:
        """Navigate to a URL."""
        visits.append(url)
        return f"Navigated to {url}"

    @browser.tool
    def browser_snapshot() -> str:
        """Snapshot the page."""
        return "Acme trust page: SOC 2 Type II available under NDA."

    @browser.tool
    def browser_click(element: str) -> str:
        """Click an element."""
        raise AssertionError("browser_click must never run")

    browser.visits = visits  # type: ignore[attr-defined]
    return {
        "evidence-filesystem": filesystem,
        "evidence-everything": everything,
        "evidence-browser": browser,
    }


@dataclass
class _Run:
    view: RunView
    tenant_id: str
    user_id: str
    request: RunRequest
    idempotency_key: str | None
    job_id: int | None = None


class MemoryStore:
    """An in-memory repository with the PostgreSQL store's state transitions."""

    def __init__(self, sources: Sequence[Source] = SOURCES) -> None:
        self.sources: dict[tuple[str, str], Source] = {}
        self._seed = [source.model_copy() for source in sources]
        self.runs: dict[UUID, _Run] = {}
        self.iterations: dict[UUID, dict[int, Iteration]] = {}
        self.events: list[tuple[str, UUID, dict[str, Any]]] = []
        self.decisions: list[tuple[UUID, str, str | None, str]] = []
        self.publications: list[UUID] = []
        self.available = True

    def _tenant(self) -> str:
        if not self.available:
            raise ConnectionError("database unavailable")
        return current_principal().tenant_id

    def _run(self, run_id: UUID) -> _Run | None:
        run = self.runs.get(run_id)
        return run if run is not None and run.tenant_id == self._tenant() else None

    async def ping(self) -> None:
        self._tenant()

    async def admit_run(
        self, request: RunRequest, *, idempotency_key: str | None, enqueue: Enqueue
    ) -> tuple[UUID, bool]:
        principal = current_principal()
        self._tenant()
        for run_id, run in self.runs.items():
            if (
                idempotency_key
                and run.idempotency_key == idempotency_key
                and run.tenant_id == principal.tenant_id
                and run.user_id == principal.subject
            ):
                return run_id, False
        run_id = uuid4()
        self.runs[run_id] = _Run(
            view=RunView(
                id=run_id,
                objective=request.objective,
                topics=request.topics,
                mode=request.mode,
                status="running",
                execution_status="queued",
                objective_state="incomplete",
                accuracy=empty_accuracy(request.topics),
                created_at=datetime.now(UTC),
            ),
            tenant_id=principal.tenant_id,
            user_id=principal.subject,
            request=request,
            idempotency_key=idempotency_key,
        )
        self.runs[run_id].job_id = await enqueue(None, run_id)
        return run_id, True

    async def get_run(self, run_id: UUID) -> RunView | None:
        run = self._run(run_id)
        if run is None:
            return None
        view = run.view.model_copy(deep=True)
        view.timeline = [
            self.iterations[run_id][key]
            for key in sorted(self.iterations.get(run_id, {}))
        ]
        cited = {
            citation.source_id
            for claim in (view.synthesis.claims if view.synthesis else [])
            for citation in claim.citations
        }
        view.sources = [
            source
            for (tenant, source_id), source in sorted(self.sources.items())
            if tenant == run.tenant_id and source_id in cited
        ]
        return view

    async def get_request(self, run_id: UUID) -> RunRequest | None:
        run = self._run(run_id)
        return run.request if run else None

    async def run_state(self, run_id: UUID) -> RunState | None:
        run = self._run(run_id)
        if run is None:
            return None
        return RunState(
            status=run.view.status,
            execution_status=run.view.execution_status,
            job_id=run.job_id,
            approval=run.view.approval,
        )

    async def search(self, query: str, limit: int = 8) -> list[Source]:
        tenant = self._tenant()
        if not any(key[0] == tenant for key in self.sources):
            for source in self._seed:
                self.sources[(tenant, source.id)] = source.model_copy()
        return [
            source
            for (owner, _), source in sorted(self.sources.items())
            if owner == tenant
        ][:limit]

    async def upsert_source(self, source: Source) -> bool:
        key = (self._tenant(), source.id)
        current = self.sources.get(key)
        if current is not None and current.revision == source.revision:
            return False
        self.sources[key] = source
        return True

    async def delete_source(self, source_id: str) -> bool:
        return self.sources.pop((self._tenant(), source_id), None) is not None

    async def mark_running(self, run_id: UUID) -> None:
        run = self._run(run_id)
        if run and run.view.execution_status in {"queued", "running", "paused"}:
            run.view.execution_status = "running"
            run.view.error = None

    async def await_reconciliation(self, run_id: UUID, message: str) -> None:
        run = self._run(run_id)
        if run and run.view.execution_status == "running":
            run.view.execution_status = "paused"
            run.view.error = message

    async def save_iteration(
        self, run_id: UUID, item: Iteration, decision: Mapping[str, Any]
    ) -> None:
        if self._run(run_id) is not None:
            self.iterations.setdefault(run_id, {})[item.iteration] = item

    async def latest_iteration(self, run_id: UUID) -> Iteration | None:
        items = self.iterations.get(run_id) if self._run(run_id) else None
        return items[max(items)] if items else None

    async def await_approval(
        self,
        run_id: UUID,
        synthesis: Synthesis,
        accuracy: Accuracy,
        approval: Approval,
    ) -> None:
        run = self._run(run_id)
        if run and run.view.status in {"running", "awaiting_approval"}:
            run.view.status = "awaiting_approval"
            run.view.execution_status = "paused"
            run.view.objective_state = "awaiting_approval"
            run.view.synthesis = synthesis
            run.view.accuracy = accuracy
            run.view.approval = approval

    async def record_decision(
        self, run_id: UUID, decision: str, note: str | None, *, enqueue: Enqueue
    ) -> RunView | None:
        run = self._run(run_id)
        if (
            run is None
            or run.view.status != "awaiting_approval"
            or run.view.approval is None
            or run.view.approval.decision is not None
        ):
            return None
        principal = current_principal()
        run.view.approval = run.view.approval.model_copy(
            update={
                "decision": decision,
                "decided_by": principal.subject,
                "decided_at": datetime.now(UTC),
                "note": note,
            }
        )
        self.decisions.append((run_id, decision, note, principal.subject))
        if decision == "reject":
            run.view.status = "rejected"
            run.view.objective_state = "failed"
        run.view.execution_status = "queued"
        run.job_id = await enqueue(None, run_id)
        return await self.get_run(run_id)

    async def requeue_paused(self, run_id: UUID, *, enqueue: Enqueue) -> bool:
        run = self._run(run_id)
        if run is None or run.view.execution_status != "paused":
            return False
        run.view.execution_status = "queued"
        run.job_id = await enqueue(None, run_id)
        return True

    async def publish(
        self, run_id: UUID, synthesis: Synthesis, accuracy: Accuracy
    ) -> datetime:
        run = self._run(run_id)
        if run is None:
            raise LookupError(run_id)
        if run.view.published_at is None:
            self.publications.append(run_id)
            run.view.published_at = datetime.now(UTC)
            run.view.status = "approved"
            run.view.objective_state = "validated"
            run.view.synthesis = synthesis
            run.view.accuracy = accuracy
        return run.view.published_at

    async def published_synthesis(self, run_id: UUID) -> Synthesis | None:
        run = self._run(run_id)
        return run.view.synthesis if run and run.view.published_at else None

    async def complete_run(
        self,
        run_id: UUID,
        *,
        termination: str | None,
        synthesis: Synthesis | None,
        accuracy: Accuracy | None,
    ) -> None:
        run = self._run(run_id)
        if run is None or run.view.execution_status in {"cancelled", "failed"}:
            return
        view = run.view
        view.execution_status = "completed"
        view.termination = termination
        view.error = None
        if view.status != "approved":
            view.synthesis = synthesis or view.synthesis
            view.accuracy = accuracy or view.accuracy
        view.objective_state = (
            "validated"
            if view.status == "approved"
            else "failed"
            if view.status == "rejected"
            else "incomplete"
        )
        if view.status not in {"approved", "rejected", "cancelled"}:
            view.status = "unvalidated"

    async def reject_run(
        self, run_id: UUID, *, termination: str, error: str | None = None
    ) -> None:
        run = self._run(run_id)
        if run is None or run.view.execution_status == "cancelled":
            return
        if run.view.status != "approved":
            run.view.status = "rejected"
            run.view.objective_state = "failed"
            run.view.execution_status = "completed"
            run.view.termination = termination
            run.view.error = error

    async def fail_run(
        self, run_id: UUID, error: str, *, termination: str = "execution_failed"
    ) -> None:
        run = self._run(run_id)
        if run is None or run.view.execution_status == "cancelled":
            return
        view = run.view
        view.execution_status = "failed"
        view.termination = termination
        view.error = error[:2000]
        view.objective_state = "validated" if view.status == "approved" else "failed"
        if view.status not in {"approved", "rejected", "cancelled"}:
            view.status = "failed"

    async def cancel_run(self, run_id: UUID, reason: str) -> bool:
        run = self._run(run_id)
        if (
            run is None
            or run.view.execution_status not in {"queued", "running", "paused"}
            or run.view.status in {"approved", "rejected"}
        ):
            return False
        run.view.status = "cancelled"
        run.view.execution_status = "cancelled"
        run.view.objective_state = "failed"
        run.view.termination = "cancelled"
        run.view.error = reason
        return True

    async def set_job(self, run_id: UUID, job_id: int) -> None:
        run = self._run(run_id)
        if run is not None:
            run.job_id = job_id

    async def append_event(
        self, run_id: UUID, event_type: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        tenant = self._tenant()
        if self._run(run_id) is None:
            raise LookupError(run_id)
        sequence = len(self.events) + 1
        event = {
            **json.loads(json.dumps(dict(payload), default=str)),
            "id": sequence,
            "type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.events.append((tenant, run_id, event))
        return event

    async def read_events(self, run_id: UUID, after: int = 0) -> list[dict[str, Any]]:
        tenant = self._tenant()
        return [
            event
            for owner, event_run, event in self.events
            if owner == tenant and event_run == run_id and event["id"] > after
        ]

    def event_types(self, run_id: UUID) -> list[str]:
        return [
            event["type"] for _, event_run, event in self.events if event_run == run_id
        ]
