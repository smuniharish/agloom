"""Offline fixtures: the real runtime composed with in-memory stores, a SQLite
agentbarrier store, in-process MCP servers, and a scripted model."""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from agentbarrier.runtime import SQLiteRuntimeStore
from backend.app import create_app
from backend.observability import METRICS_NAMESPACE, Telemetry, build_telemetry
from backend.runtime import Backends, Runtime, open_runtime
from backend.settings import Settings
from fastapi import FastAPI
from feedback_manager.storage import InMemoryFeedbackStore
from langchain_core.messages import AIMessage
from refresh_engine import InMemoryStateStore

from .doubles import (
    Call,
    EventLog,
    FakeServers,
    MemoryAudits,
    MemoryPolicies,
    MemoryRegister,
    ScriptedChatModel,
    auditing,
    report,
)

APP_ROOT = Path(__file__).resolve().parent.parent
FAILING = auditing(report())


@dataclass
class Harness:
    """An offline auditor and everything a test inspects."""

    settings: Settings
    policies_file: Path
    policy_store: SQLiteRuntimeStore
    responder: Callable[[Call], AIMessage | str] | None
    policies: MemoryPolicies = field(default_factory=MemoryPolicies)
    audits: MemoryAudits = field(default_factory=MemoryAudits)
    register: MemoryRegister = field(default_factory=MemoryRegister)
    feedback: InMemoryFeedbackStore = field(default_factory=InMemoryFeedbackStore)
    servers: FakeServers = field(default_factory=FakeServers)
    events: EventLog = field(default_factory=EventLog)
    model: ScriptedChatModel | None = None
    telemetry: Telemetry | None = None

    def __post_init__(self) -> None:
        if self.responder is not None:
            self.model = ScriptedChatModel(responder=self.responder)

    def backends(self) -> Backends:
        return Backends(
            policies=self.policies,
            feedback=self.feedback,
            refresh=InMemoryStateStore(),
            policy=self.policy_store,
            audits=self.audits,
            register=self.register,
            counts=lambda: {"policies": len(self.policies.policies)},
            close=lambda: None,
        )

    @asynccontextmanager
    async def runtime(self) -> AsyncIterator[Runtime]:
        telemetry = build_telemetry(self.settings, environ={})
        self.telemetry = dataclasses.replace(
            telemetry, observers=(*telemetry.observers, self.events)
        )
        async with open_runtime(
            self.settings,
            self.backends(),
            model=self.model,
            mcp_targets=self.servers.targets(),
            telemetry=self.telemetry,
        ) as runtime:
            yield runtime

    def app(self) -> FastAPI:
        return create_app(self.runtime)

    def audits_counted(self, verdict: str) -> float:
        assert self.telemetry is not None
        value = self.telemetry.registry.get_sample_value(
            f"{METRICS_NAMESPACE}_audits_total", {"verdict": verdict}
        )
        return value or 0.0


HarnessFactory = Callable[..., Harness]


@pytest.fixture
def make_harness(tmp_path: Path) -> Iterator[HarnessFactory]:
    """Build harnesses; the default model searches once and answers fail."""
    stores: list[SQLiteRuntimeStore] = []

    def build(
        responder: Callable[[Call], AIMessage | str] | None = FAILING,
    ) -> Harness:
        policies_file = tmp_path / "policies.json"
        shutil.copyfile(APP_ROOT / "data" / "policies.json", policies_file)
        settings = Settings.from_environment(
            {
                "DATABASE_URL": "postgresql://tests.invalid/unused",
                "POLICIES_FILE": str(policies_file),
                "WORKSPACE_DIR": str(APP_ROOT / "workspace"),
                "BROWSER_OUTPUT_DIR": str(tmp_path / "browser"),
                "APPROVAL_TTL_SECONDS": "3600",
            }
        )
        store = SQLiteRuntimeStore(tmp_path / f"agentbarrier-{len(stores)}.db")
        stores.append(store)
        return Harness(
            settings=settings,
            policies_file=policies_file,
            policy_store=store,
            responder=responder,
        )

    yield build
    for store in stores:
        store.close()
