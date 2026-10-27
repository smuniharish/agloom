"""Offline fixtures: the real runtime composed with in-memory stores, a SQLite
agentbarrier store, in-process MCP servers, and a scripted model."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from agentbarrier.runtime import SQLiteRuntimeStore
from backend.app import create_app
from backend.observability import build_telemetry
from backend.runtime import Backends, Runtime, open_runtime
from backend.settings import Settings
from behaviorweave import InMemoryBehaviorStateStore
from fastapi import FastAPI
from feedback_manager.storage import InMemoryFeedbackStore
from langchain_core.messages import AIMessage
from langgraph_xai import InMemoryProvenanceStore
from mcp_capability_router import InMemoryRegistry
from refresh_engine import InMemoryStateStore

from .doubles import (
    Call,
    EventLog,
    FakeServers,
    KeywordEmbeddings,
    MemoryEvidence,
    MemoryInvestigations,
    MemoryReleaseRegister,
    ScriptedChatModel,
    copy_workspace,
    investigating,
    report,
)

APP_ROOT = Path(__file__).resolve().parent.parent
TARGET_SITE = "http://release-status.test"
DOCUMENTS = ("ADR-101.md", "RELEASE-301.md", "RUNBOOK-201.md")
NO_GO = investigating(report())


@dataclass
class Harness:
    """An offline investigator and everything a test inspects."""

    settings: Settings
    workspace: Path
    policy_store: SQLiteRuntimeStore
    responder: Callable[[Call], AIMessage | str] | None
    evidence: MemoryEvidence = field(default_factory=MemoryEvidence)
    investigations: MemoryInvestigations = field(default_factory=MemoryInvestigations)
    register: MemoryReleaseRegister = field(default_factory=MemoryReleaseRegister)
    feedback: InMemoryFeedbackStore = field(default_factory=InMemoryFeedbackStore)
    provenance: InMemoryProvenanceStore = field(default_factory=InMemoryProvenanceStore)
    servers: FakeServers = field(default_factory=FakeServers)
    events: EventLog = field(default_factory=EventLog)
    model: ScriptedChatModel | None = None

    def __post_init__(self) -> None:
        if self.responder is not None:
            self.model = ScriptedChatModel(responder=self.responder)

    def backends(self) -> Backends:
        return Backends(
            evidence=self.evidence,
            feedback=self.feedback,
            behavior=InMemoryBehaviorStateStore(),
            mcp_registry=InMemoryRegistry(),
            refresh=InMemoryStateStore(),
            provenance=self.provenance,
            policy=self.policy_store,
            investigations=self.investigations,
            release_register=self.register,
            counts=lambda: {"documents": len(self.evidence.documents)},
            close=lambda: None,
        )

    @asynccontextmanager
    async def runtime(self) -> AsyncIterator[Runtime]:
        telemetry = build_telemetry(self.settings, environ={})
        telemetry = dataclasses.replace(
            telemetry, observers=(*telemetry.observers, self.events)
        )
        async with open_runtime(
            self.settings,
            self.backends(),
            model=self.model,
            embeddings=KeywordEmbeddings(),
            mcp_targets=self.servers.targets(),
            telemetry=telemetry,
        ) as runtime:
            yield runtime

    def app(self) -> FastAPI:
        return create_app(self.runtime)


HarnessFactory = Callable[..., Harness]


@pytest.fixture
def make_harness(tmp_path: Path) -> Iterator[HarnessFactory]:
    """Build harnesses; the default model searches once and answers no-go."""
    stores: list[SQLiteRuntimeStore] = []

    def build(
        responder: Callable[[Call], AIMessage | str] | None = NO_GO,
    ) -> Harness:
        workspace = copy_workspace(
            APP_ROOT / "workspace", tmp_path / "workspace", DOCUMENTS
        )
        settings = Settings.from_environment(
            {
                "DATABASE_URL": "postgresql://tests.invalid/unused",
                "TARGET_SITE_URL": TARGET_SITE,
                "WORKSPACE_DIR": str(workspace),
                "BROWSER_OUTPUT_DIR": str(tmp_path / "browser"),
                "APPROVAL_TTL_SECONDS": "3600",
            }
        )
        store = SQLiteRuntimeStore(tmp_path / f"agentbarrier-{len(stores)}.db")
        stores.append(store)
        return Harness(
            settings=settings,
            workspace=workspace,
            policy_store=store,
            responder=responder,
        )

    yield build
    for store in stores:
        store.close()
