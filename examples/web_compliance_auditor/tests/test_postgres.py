"""PostgreSQL persistence of the auditor.

These tests need a disposable PostgreSQL database with pgvector; they drop and
recreate the application tables. Set `COMPLIANCE_TEST_DATABASE_URL`, for
example to a `pgvector/pgvector:pg18` container.
"""

from __future__ import annotations

import json
import os
import shutil
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
from backend.governance import RECORD_AUDIT, policy_manager, runtime_policy
from backend.models import Audit, ComplianceRecord
from backend.observability import build_telemetry
from backend.policies import IndexOperation, PolicyFileSource, PolicyIndex
from backend.runtime import Runtime, open_runtime, postgres_backends
from backend.settings import Settings
from backend.stores import (
    TABLES,
    AuditLog,
    ComplianceRegister,
    PostgresFeedbackStore,
    PostgresRefreshStore,
    open_pool,
    table_counts,
)
from fastapi.testclient import TestClient
from feedback_manager import (
    FeedbackCategory,
    FeedbackManager,
    FeedbackQuery,
    FeedbackSource,
    FeedbackStatus,
    FeedbackTarget,
    FeedbackTargetType,
)
from psycopg_pool import ConnectionPool
from refresh_engine import RefreshEngine

from .conftest import APP_ROOT
from .doubles import FakeServers, HashEmbeddings, ScriptedChatModel, auditing, report

DATABASE_URL = os.environ.get("COMPLIANCE_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not DATABASE_URL, reason="set COMPLIANCE_TEST_DATABASE_URL to run"
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


def policies_copy(tmp_path: Path) -> Path:
    path = tmp_path / "policies.json"
    shutil.copyfile(APP_ROOT / "data" / "policies.json", path)
    return path


def test_policy_index_ranks_filters_and_deletes(pool: ConnectionPool) -> None:
    index = PolicyIndex(pool, HashEmbeddings())
    index.initialize()
    index.upsert("P-LABELS", "Labels", "inputs need meaningful labels")
    index.upsert("P-HTTPS", "Transport", "forms must use https transport")

    best = index.search("meaningful labels for inputs", limit=1)[0]
    assert best.policy_id == "P-LABELS"
    assert 0 <= best.similarity <= 1
    filtered = index.search("anything", policy_ids=["P-HTTPS"])
    assert [item.policy_id for item in filtered] == ["P-HTTPS"]

    index.delete("P-HTTPS")
    assert [item.policy_id for item in index.search("anything")] == ["P-LABELS"]


async def test_policy_refresh_survives_restarts(
    pool: ConnectionPool, tmp_path: Path
) -> None:
    index = PolicyIndex(pool, HashEmbeddings())
    index.initialize()
    store = PostgresRefreshStore(pool)
    store.initialize()
    path = policies_copy(tmp_path)

    def engine() -> RefreshEngine:
        return RefreshEngine(PolicyFileSource(path), IndexOperation(index), store=store)

    first = engine()
    try:
        initial = await first.refresh()
    finally:
        await first.close()
    restarted = engine()
    try:
        unchanged = await restarted.refresh()
        policies = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(policies[1:]), encoding="utf-8")
        removed = await restarted.refresh()
    finally:
        await restarted.close()

    assert initial.added_count == 3
    assert unchanged.unchanged_count == 3
    assert removed.deleted_count == 1
    assert table_counts(pool)["policies"] == 2
    assert len(await store.load_all()) == 2


async def test_feedback_store_honours_the_feedback_manager_contract(
    pool: ConnectionPool,
) -> None:
    store = PostgresFeedbackStore(pool)
    store.initialize()
    manager = FeedbackManager(store=store)
    target = FeedbackTarget(type=FeedbackTargetType.THREAD, id="audit-1")
    try:
        first = await manager.submit(
            source=FeedbackSource.HUMAN,
            category=FeedbackCategory.CORRECTION,
            target=target,
            payload={"rating": "down", "correction": "Labels are present."},
            idempotency_key="audit-1:down",
        )
        again = await manager.submit(
            source=FeedbackSource.HUMAN,
            category=FeedbackCategory.CORRECTION,
            target=target,
            payload={"rating": "down", "correction": "Labels are present."},
            idempotency_key="audit-1:down",
        )
        assert again.feedback_id == first.feedback_id
        acknowledged = await manager.acknowledge(first.feedback_id)
        assert acknowledged.status is FeedbackStatus.ACKNOWLEDGED
        found = await store.query(
            FeedbackQuery(target_type="thread", target_id="audit-1", newest_first=True)
        )
        assert [event.status for event in found] == [FeedbackStatus.ACKNOWLEDGED]
    finally:
        await manager.aclose()


async def test_audit_log_and_compliance_register(pool: ConnectionPool) -> None:
    log = AuditLog(pool)
    log.initialize()
    register = ComplianceRegister(pool)
    register.initialize()
    audit = Audit(
        audit_id=uuid4(),
        url="http://target-site/",
        report=report(),
        evidence=[],
        mcp_checks={"playwright": "verified"},
        record=ComplianceRecord(status="awaiting_approval", action_id="a-1"),
        created_at=datetime.now(UTC),
    )
    await log.save(audit)
    updated = await log.update_record(
        audit.audit_id, ComplianceRecord(status="rejected")
    )
    assert updated is not None
    assert (await log.get(audit.audit_id)) == updated

    arguments = {
        "audit_id": str(audit.audit_id),
        "url": audit.url,
        "verdict": "fail",
        "summary": "Labels are missing.",
        "policy_ids": ["WEB-ACCESS-001"],
    }
    assert await register.ainvoke(arguments) == await register.ainvoke(arguments)
    assert table_counts(pool)["register"] == 1


async def test_postgres_policy_store_records_approved_passes_once(
    pool: ConnectionPool,
) -> None:
    store = PostgresRuntimeStore(
        DATABASE_URL, schema="agentbarrier", create_schema=True, migrate=True
    )
    manager = policy_manager(store, runtime_policy(approval_ttl_seconds=600))
    published: list[str] = []

    async def publish() -> dict[str, str]:
        published.append("pass")
        return {"recorded_at": datetime.now(UTC).isoformat()}

    arguments = {
        "audit_id": "audit-pg",
        "url": "https://example.com/",
        "verdict": "pass",
        "summary": "Compliant.",
        "policy_ids": ["WEB-ACCESS-001"],
    }
    try:
        with pytest.raises(ApprovalRequired) as pending:
            await manager.aexecute(RECORD_AUDIT, arguments, publish)
        store.decide(
            pending.value.action.action_id, Decision.APPROVE, decided_by="officer"
        )
        await manager.aexecute(RECORD_AUDIT, arguments, publish)
        await manager.aexecute(RECORD_AUDIT, arguments, publish)
    finally:
        store.close()

    assert published == ["pass"]


def test_production_wiring_persists_an_audit(
    pool: ConnectionPool, tmp_path: Path
) -> None:
    settings = Settings.from_environment(
        {
            "DATABASE_URL": DATABASE_URL,
            "POLICIES_FILE": str(policies_copy(tmp_path)),
            "BROWSER_OUTPUT_DIR": str(tmp_path / "browser"),
        }
    )

    @asynccontextmanager
    async def runtime() -> AsyncIterator[Runtime]:
        backends = postgres_backends(settings, HashEmbeddings())
        try:
            async with open_runtime(
                settings,
                backends,
                model=ScriptedChatModel(responder=auditing(report("pass"))),
                mcp_targets=FakeServers().targets(),
                telemetry=build_telemetry(settings, environ={}),
            ) as started:
                yield started
        finally:
            backends.close()

    with TestClient(create_app(runtime)) as client:
        body = client.post("/api/audits", json={"url": "http://target-site/"})
        assert body.status_code == 200, body.text
        audit_id = body.json()["audit_id"]
        assert body.json()["record"]["status"] == "awaiting_approval"
        approved = client.post(
            f"/api/audits/{audit_id}/approval",
            json={"approve": True, "reviewer": "compliance-officer"},
        )
        client.post(
            f"/api/audits/{audit_id}/feedback",
            json={"rating": "up", "correction": None},
        )
        health = client.get("/api/health").json()

    assert approved.json()["record"]["status"] == "recorded"
    assert health["database"] == {
        "policies": 3,
        "refresh_states": 3,
        "audits": 1,
        "register": 1,
        "feedback": 1,
    }
