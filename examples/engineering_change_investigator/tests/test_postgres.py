"""PostgreSQL implementations of the package persistence contracts.

These tests need a disposable PostgreSQL database with pgvector; they drop and
recreate the application tables. Set `ENGINEERING_TEST_DATABASE_URL`, for
example to a `pgvector/pgvector:pg18` container.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from agentbarrier import Decision
from agentbarrier.errors import ApprovalRequired
from agentbarrier.runtime import PostgresRuntimeStore
from backend.app import create_app
from backend.evidence import EvidenceIndex, IndexOperation, WorkspaceSource
from backend.governance import RELEASE_DECISION, policy_manager, runtime_policy
from backend.models import Investigation, ReleaseDecision
from backend.observability import build_telemetry
from backend.runtime import Runtime, open_runtime, postgres_backends
from backend.settings import Settings
from backend.stores import (
    TABLES,
    InvestigationLog,
    PostgresBehaviorStateStore,
    PostgresCapabilityRegistry,
    PostgresFeedbackStore,
    PostgresProvenanceStore,
    PostgresRefreshStore,
    ReleaseRegister,
    open_pool,
    table_counts,
)
from behaviorweave import BehaviorState
from fastapi.testclient import TestClient
from feedback_manager import (
    FeedbackCategory,
    FeedbackConflictError,
    FeedbackManager,
    FeedbackQuery,
    FeedbackSource,
    FeedbackStatus,
    FeedbackTarget,
    FeedbackTargetType,
)
from langgraph_xai import Execution, ProvenanceLink, Registry, XAIRuntime
from langgraph_xai import ProvenanceStore as ProvenanceStoreContract
from langgraph_xai.storage import StoreFilter
from mcp_capability_router import CapabilityType, MCPRuntime, RefreshPolicy
from psycopg_pool import ConnectionPool
from refresh_engine import RefreshEngine

from agloom import create_agent

from .conftest import APP_ROOT, DOCUMENTS, TARGET_SITE
from .doubles import (
    FakeServers,
    HashEmbeddings,
    ScriptedChatModel,
    copy_workspace,
    investigating,
    report,
)

DATABASE_URL = os.environ.get("ENGINEERING_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not DATABASE_URL, reason="set ENGINEERING_TEST_DATABASE_URL to run"
    ),
]


@pytest.fixture
def pool() -> Iterator[ConnectionPool]:
    with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
        for table in TABLES.values():
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute("DROP SCHEMA IF EXISTS agentbarrier CASCADE")
    shared = open_pool(DATABASE_URL, max_size=4)
    yield shared
    shared.close()


def test_evidence_index_ranks_updates_and_deletes(pool: ConnectionPool) -> None:
    index = EvidenceIndex(pool, HashEmbeddings())
    index.initialize()
    index.upsert("ADR-101", "ADR-101", "ADR-101.md", "connection pool minimum forty")
    index.upsert("RUNBOOK-201", "RUNBOOK-201", "RUNBOOK-201.md", "stop the rollout")

    best = index.search("connection pool minimum", limit=1)[0]
    assert best.metadata["document_id"] == "ADR-101"
    assert best.metadata["capability_name"] == "semantic_change_search"
    assert 0 < best.metadata["score"] <= 1

    index.upsert("ADR-101", "ADR-101", "ADR-101.md", "updated rollout guidance")
    index.delete("RUNBOOK-201")
    [only] = index.search("anything", limit=5)
    assert only.page_content == "updated rollout guidance"
    assert table_counts(pool)["documents"] == 1


async def test_feedback_store_honours_the_feedback_manager_contract(
    pool: ConnectionPool,
) -> None:
    store = PostgresFeedbackStore(pool)
    store.initialize()
    manager = FeedbackManager(store=store)
    target = FeedbackTarget(type=FeedbackTargetType.THREAD, id="investigation-1")
    try:
        first = await manager.submit(
            source=FeedbackSource.HUMAN,
            category=FeedbackCategory.RATING,
            target=target,
            payload={"rating": "down"},
            idempotency_key="investigation-1:down",
        )
        again = await manager.submit(
            source=FeedbackSource.HUMAN,
            category=FeedbackCategory.RATING,
            target=target,
            payload={"rating": "down"},
            idempotency_key="investigation-1:down",
        )
        assert again.feedback_id == first.feedback_id

        received = await store.query(
            FeedbackQuery(
                target_type="thread",
                target_id="investigation-1",
                status=FeedbackStatus.RECEIVED,
            )
        )
        assert [event.feedback_id for event in received] == [first.feedback_id]

        acknowledged = await manager.acknowledge(first.feedback_id)
        assert acknowledged.status is FeedbackStatus.ACKNOWLEDGED
        stored = await store.get(first.feedback_id)
        assert stored is not None
        assert stored.status is FeedbackStatus.ACKNOWLEDGED
        with pytest.raises(FeedbackConflictError):
            await store.transition(
                first.feedback_id,
                FeedbackStatus.HANDLED,
                expected=FeedbackStatus.RECEIVED,
            )
        assert await store.get(uuid4()) is None
    finally:
        await manager.aclose()


def test_behavior_state_updates_are_durable(pool: ConnectionPool) -> None:
    store = PostgresBehaviorStateStore(pool)
    store.initialize()

    def bump(state: BehaviorState | None) -> BehaviorState:
        current = state or BehaviorState(pattern_id="repeated_tool_call", scope="s")
        return dataclasses.replace(
            current,
            count=current.count + 1,
            last_seen=datetime.now(UTC),
            version=current.version + 1,
        )

    store.update("scope-1", bump)
    updated = store.update("scope-1", bump)
    restored = store.get("scope-1")

    assert restored == updated
    assert restored is not None
    assert restored.count == 2
    store.delete("scope-1")
    assert store.get("scope-1") is None


async def test_capability_registry_persists_mcp_discovery(pool: ConnectionPool) -> None:
    registry = PostgresCapabilityRegistry(pool)
    registry.initialize()
    servers = FakeServers().targets()
    async with MCPRuntime(registry=registry) as runtime:
        for server_id, target in servers.items():
            await runtime.register_mcp(
                server_id, target, refresh=RefreshPolicy(on_register=True)
            )
        echo = await runtime.execute(
            "engineering-everything:tool:echo", {"message": "x"}
        )
        assert echo.text == "Echo: x"

    stored = await registry.list()
    assert {item.server_id for item in stored} == set(servers)
    prompt = await registry.get("engineering-everything:prompt:simple-prompt")
    assert prompt is not None
    assert prompt.type is CapabilityType.PROMPT
    resources = await registry.list(type=CapabilityType.RESOURCE)
    assert [item.uri for item in resources] == [
        "demo://resource/static/document/architecture.md"
    ]
    tools = await registry.list(
        server_id="engineering-filesystem", type=CapabilityType.TOOL
    )
    assert {tool.name for tool in tools} == {
        "list_directory",
        "read_text_file",
        "write_file",
    }

    await registry.remove("engineering-everything:tool:echo")
    await registry.remove_server("engineering-playwright")
    remaining = {item.capability_id for item in await registry.list()}
    assert "engineering-everything:tool:echo" not in remaining
    assert not any(item.startswith("engineering-playwright") for item in remaining)


async def test_refresh_state_survives_restarts(
    pool: ConnectionPool, tmp_path: Path
) -> None:
    index = EvidenceIndex(pool, HashEmbeddings())
    index.initialize()
    store = PostgresRefreshStore(pool)
    store.initialize()
    workspace = copy_workspace(
        APP_ROOT / "workspace", tmp_path / "workspace", DOCUMENTS
    )

    def engine() -> RefreshEngine:
        return RefreshEngine(
            WorkspaceSource(workspace), IndexOperation(index), store=store
        )

    first = engine()
    try:
        initial = await first.refresh()
    finally:
        await first.close()
    restarted = engine()
    try:
        unchanged = await restarted.refresh()
        (workspace / "ADR-101.md").write_text(
            "# ADR-101\n\nNew minimum.", encoding="utf-8"
        )
        modified = await restarted.refresh()
    finally:
        await restarted.close()

    assert initial.added_count == 3
    assert unchanged.refreshed_count == 0
    assert unchanged.unchanged_count == 3
    assert modified.modified_count == 1
    assert len(await store.load_all()) == 3


async def test_provenance_store_records_an_explained_agent_run(
    pool: ConnectionPool,
) -> None:
    store = PostgresProvenanceStore(pool)
    store.initialize()
    assert isinstance(store, ProvenanceStoreContract)
    registry = Registry()
    registry.register(ProvenanceStoreContract, store)
    explainability = XAIRuntime(registry=registry, application_id="eci", tenant_id="t")
    agent = create_agent(
        model=ScriptedChatModel(responder=lambda call: "The pool is too small."),
        pattern="react",
        explainability=explainability,
    )
    try:
        await agent.ainvoke("Summarize the release risk.")
    finally:
        agent.close()
        await explainability.close()

    records = [
        item
        async for item in store.query(StoreFilter(application_id="eci", tenant_id="t"))
    ]
    executions = [item for item in records if isinstance(item, Execution)]
    assert executions
    fetched = await store.get(str(executions[0].id))
    assert isinstance(fetched, Execution)
    assert fetched.id == executions[0].id
    links = [item for item in records if isinstance(item, ProvenanceLink)]
    for link in links:
        parents = await store.parents(str(link.target_id), context=link.context)
        children = await store.children(str(link.source_id), context=link.context)
        assert link in parents
        assert link in children
    only_executions = [
        item
        async for item in store.query(
            StoreFilter(application_id="eci", tenant_id="t", item_type=Execution)
        )
    ]
    assert only_executions == executions
    assert [
        item
        async for item in store.query(StoreFilter(application_id="eci", tenant_id="x"))
    ] == []


async def test_investigations_and_release_register(pool: ConnectionPool) -> None:
    log = InvestigationLog(pool)
    log.initialize()
    register = ReleaseRegister(pool)
    register.initialize()
    investigation = Investigation(
        investigation_id=uuid4(),
        question="Ready?",
        report=report(),
        mcp_checks={"filesystem": "verified"},
        release_decision=ReleaseDecision(status="awaiting_approval", action_id="a-1"),
        created_at=datetime.now(UTC),
    )
    await log.save(investigation)
    updated = await log.update_decision(
        investigation.investigation_id, ReleaseDecision(status="rejected")
    )

    assert updated is not None
    assert (await log.get(investigation.investigation_id)) == updated
    assert (
        await log.update_decision(uuid4(), ReleaseDecision(status="rejected")) is None
    )

    arguments = {
        "investigation_id": str(investigation.investigation_id),
        "decision": "no_go",
        "risk": "high",
        "summary": "Pool below the minimum.",
        "citations": ["ADR-101"],
    }
    first = await register.ainvoke(arguments)
    second = await register.ainvoke(arguments)
    assert first == second
    assert table_counts(pool)["release_decisions"] == 1


async def test_postgres_policy_store_publishes_approved_decisions_once(
    pool: ConnectionPool,
) -> None:
    store = PostgresRuntimeStore(
        DATABASE_URL, schema="agentbarrier", create_schema=True, migrate=True
    )
    manager = policy_manager(
        store, runtime_policy(target_site_url=TARGET_SITE, approval_ttl_seconds=600)
    )
    published: list[str] = []

    async def publish() -> dict[str, str]:
        published.append("go")
        return {"recorded_at": datetime.now(UTC).isoformat()}

    arguments = {
        "investigation_id": "inv-pg",
        "decision": "go",
        "risk": "low",
        "summary": "Evidence complete.",
        "citations": ["RELEASE-301"],
    }
    try:
        with pytest.raises(ApprovalRequired) as pending:
            await manager.aexecute(RELEASE_DECISION, arguments, publish)
        store.decide(
            pending.value.action.action_id, Decision.APPROVE, decided_by="manager"
        )
        await manager.aexecute(RELEASE_DECISION, arguments, publish)
        await manager.aexecute(RELEASE_DECISION, arguments, publish)
    finally:
        store.close()

    assert published == ["go"]


def test_production_wiring_persists_an_investigation(
    pool: ConnectionPool, tmp_path: Path
) -> None:
    workspace = copy_workspace(
        APP_ROOT / "workspace", tmp_path / "workspace", DOCUMENTS
    )
    settings = Settings.from_environment(
        {
            "DATABASE_URL": DATABASE_URL,
            "TARGET_SITE_URL": TARGET_SITE,
            "WORKSPACE_DIR": str(workspace),
            "BROWSER_OUTPUT_DIR": str(tmp_path / "browser"),
        }
    )
    embeddings = HashEmbeddings()

    @asynccontextmanager
    async def runtime() -> AsyncIterator[Runtime]:
        backends = postgres_backends(settings, embeddings)
        try:
            async with open_runtime(
                settings,
                backends,
                model=ScriptedChatModel(responder=investigating(report("go"))),
                embeddings=embeddings,
                mcp_targets=FakeServers().targets(),
                telemetry=build_telemetry(settings, environ={}),
            ) as started:
                yield started
        finally:
            backends.close()

    with TestClient(create_app(runtime)) as client:
        body = client.post("/api/investigate", json={"question": "Ready to deploy?"})
        assert body.status_code == 200, body.text
        investigation_id = body.json()["investigation_id"]
        assert body.json()["release_decision"]["status"] == "awaiting_approval"
        approved = client.post(
            f"/api/investigations/{investigation_id}/approval",
            json={"approve": True, "reviewer": "release-manager"},
        )
        health = client.get("/api/health").json()

    assert approved.json()["release_decision"]["status"] == "recorded"
    counts = health["database"]
    assert counts["documents"] == 3
    assert counts["investigations"] == 1
    assert counts["release_decisions"] == 1
    assert counts["mcp_capabilities"] > 0
    assert counts["refresh_states"] == 3
    assert counts["xai_records"] > 0
