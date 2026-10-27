"""End-to-end behavior on PostgreSQL: pgvector retrieval, transactional admission,
procrastinate workers, the PostgreSQL checkpointer, and agentbarrier."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from agentbarrier.runtime import PostgresRuntimeStore, RuntimeStatus
from backend import runtime as runtime_module
from backend.app import create_app
from backend.auth import Principal, StaticAuthenticator, principal_scope
from backend.config import Settings
from backend.jobs import (
    RECOVER_TASK,
    RUN_TASK,
    RUNNER_CONTEXT,
    create_job_app,
    enqueue_run,
    run_lock,
)
from backend.models import RunRequest, Source
from backend.policy import ApprovalDesk
from backend.runtime import (
    ApiResources,
    WorkerRuntime,
    job_connector,
    open_worker_runtime,
)
from backend.setup_database import setup_database
from backend.store import EvidenceStore
from procrastinate import App, JobContext
from procrastinate.jobs import Job
from psycopg import sql

from tests.support import (
    AUTHORIZATION,
    SOURCES,
    TEST_PRINCIPAL,
    EvidenceScript,
    HashEmbeddings,
    ScriptedChatModel,
    fake_mcp_servers,
    synthesis,
)
from tests.test_runner import blocking

pytestmark = pytest.mark.postgres

RUN = {
    "objective": "Verify Acme security and incident response evidence.",
    "topics": ["encryption", "incident response"],
}


@dataclass
class Platform:
    settings: Settings
    store: EvidenceStore
    jobs: App
    barrier: PostgresRuntimeStore

    def worker(self, model: ScriptedChatModel):
        return open_worker_runtime(
            self.settings,
            store=self.store,
            jobs=self.jobs,
            barrier=self.barrier,
            model_factory=lambda: model,
            servers={},
        )


async def drain(worker: WorkerRuntime) -> None:
    await worker.jobs.run_worker_async(
        queues=[worker.settings.queue],
        wait=False,
        additional_context={RUNNER_CONTEXT: worker.runner},
        install_signal_handlers=False,
    )


@pytest.fixture
async def platform(schema_dsn: str) -> AsyncIterator[Platform]:
    barrier_schema = f"ab_{uuid4().hex[:12]}"
    settings = Settings(
        database_url=schema_dsn,
        barrier_schema=barrier_schema,
        max_attempts=2,
        retry_backoff_seconds=1,
    )
    await setup_database(settings)
    store = EvidenceStore(schema_dsn, embeddings=HashEmbeddings())
    await store.open()
    jobs = create_job_app(job_connector(schema_dsn), settings)
    barrier = PostgresRuntimeStore(schema_dsn, schema=barrier_schema)
    try:
        async with jobs.open_async():
            yield Platform(settings, store, jobs, barrier)
    finally:
        barrier.close()
        await store.close()
        with psycopg.connect(schema_dsn, autocommit=True) as db:
            db.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(barrier_schema)
                )
            )


def enqueuer(jobs: App):
    async def enqueue(connection: Any, run_id: UUID) -> int:
        return await enqueue_run(
            jobs,
            run_id,
            tenant_id=TEST_PRINCIPAL.tenant_id,
            user_id=TEST_PRINCIPAL.subject,
            connection=connection,
        )

    return enqueue


async def test_store_admission_retrieval_and_events(platform: Platform):
    store, jobs = platform.store, platform.jobs
    request = RunRequest(**RUN)
    run_id, created = await store.admit_run(
        request, idempotency_key="key-1", enqueue=enqueuer(jobs)
    )
    duplicate = await store.admit_run(
        request, idempotency_key="key-1", enqueue=enqueuer(jobs)
    )
    assert created
    assert duplicate == (run_id, False)
    queued = await jobs.job_manager.list_jobs_async(queueing_lock=run_lock(run_id))
    assert [job.task_name for job in queued] == [RUN_TASK]
    state = await store.run_state(run_id)
    assert state is not None
    assert state.job_id == queued[0].id

    async def broken(connection: Any, run_id: UUID) -> int:
        raise RuntimeError("queue unavailable")

    with pytest.raises(RuntimeError):
        await store.admit_run(request, idempotency_key="key-2", enqueue=broken)
    async with store.connection() as db:
        count = await (await db.execute("SELECT count(*) FROM loop_runs")).fetchone()
    assert count == (1,)

    security = next(source for source in SOURCES if source.id == "VENDOR-SEC-001")
    sources = await store.search(security.body, limit=8)
    assert {source.id for source in sources} == {
        "VENDOR-SEC-001",
        "VENDOR-IR-002",
        "VENDOR-COMP-003",
    }
    assert sources[0].id == "VENDOR-SEC-001"
    assert sources[0].similarity == pytest.approx(1.0)
    unchanged = sources[0]
    assert await store.upsert_source(unchanged) is False
    revised = Source(**{**unchanged.model_dump(), "revision": "2026-10-01"})
    assert await store.upsert_source(revised) is True
    assert await store.delete_source("VENDOR-COMP-003") is True

    first = await store.append_event(run_id, "queued", {"message": "accepted"})
    await store.append_event(run_id, "started", {"message": "running"})
    assert [
        event["type"] for event in await store.read_events(run_id, after=first["id"])
    ] == ["started"]
    with principal_scope(Principal("other-user", "other-tenant")):
        assert await store.get_run(run_id) is None
        assert await store.read_events(run_id) == []
        assert await store.search("encryption") != []


async def test_review_flow_with_workers_checkpoints_and_agentbarrier(
    platform: Platform,
):
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    app = create_app(
        settings=platform.settings,
        resources=ApiResources(
            store=platform.store,
            jobs=platform.jobs,
            approvals=ApprovalDesk(platform.barrier),
        ),
        authenticator=StaticAuthenticator(TEST_PRINCIPAL),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://evidence.test",
            headers=AUTHORIZATION,
        ) as client,
        platform.worker(model) as worker,
    ):
        run_id = (await client.post("/api/runs", json=RUN)).json()["run_id"]
        await drain(worker)
        paused = (await client.get(f"/api/runs/{run_id}")).json()
        assert paused["status"] == "awaiting_approval"
        action_id = paused["approval"]["action_id"]
        assert platform.barrier.get_action(action_id).status is RuntimeStatus.PENDING

        decided = await client.post(
            f"/api/runs/{run_id}/decision", json={"decision": "approve"}
        )
        assert decided.status_code == 200
        await drain(worker)

        done = (await client.get(f"/api/runs/{run_id}")).json()
        assert done["status"] == "approved"
        assert done["execution_status"] == "completed"
        assert done["published_at"] is not None
        assert platform.barrier.get_action(action_id).status is RuntimeStatus.SUCCEEDED
        events = [
            json.loads(line.removeprefix("data: "))["type"]
            for line in (
                await client.get(f"/api/runs/{run_id}/events")
            ).text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[0] == "queued"
        assert events[-1] == "awaiting_approval"

    async with platform.store.connection() as db:
        claims = await (
            await db.execute(
                "SELECT count(*) FROM claims WHERE run_id=%s", (UUID(run_id),)
            )
        ).fetchone()
        evidence = await (
            await db.execute(
                "SELECT bool_and(valid), count(*) FROM evidence WHERE run_id=%s",
                (UUID(run_id),),
            )
        ).fetchone()
        decisions = await (
            await db.execute(
                "SELECT decision, action_id FROM decisions WHERE run_id=%s",
                (UUID(run_id),),
            )
        ).fetchall()
    assert claims == (2,)
    assert evidence == (True, 2)
    assert decisions == [("approve", action_id)]
    statuses = [
        job.status
        for job in await platform.jobs.job_manager.list_jobs_async(task=RUN_TASK)
    ]
    assert statuses == ["succeeded", "succeeded"]


async def test_failed_publication_is_reconciled_against_the_ledger(
    platform: Platform,
):
    store = platform.store
    publish = store.publish
    calls = {"publish": 0}

    async def flaky(run_id: UUID, synthesis: Any, accuracy: Any) -> Any:
        calls["publish"] += 1
        if calls["publish"] == 1:
            raise ConnectionError("connection lost before the commit")
        return await publish(run_id, synthesis, accuracy)

    store.publish = flaky  # type: ignore[method-assign]
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    app = create_app(
        settings=platform.settings,
        resources=ApiResources(
            store=store, jobs=platform.jobs, approvals=ApprovalDesk(platform.barrier)
        ),
        authenticator=StaticAuthenticator(TEST_PRINCIPAL),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://evidence.test",
            headers=AUTHORIZATION,
        ) as client,
        platform.worker(model) as worker,
    ):
        run_id = (await client.post("/api/runs", json=RUN)).json()["run_id"]
        await drain(worker)
        approve = {"decision": "approve"}
        decided = await client.post(f"/api/runs/{run_id}/decision", json=approve)
        assert decided.status_code == 200
        await drain(worker)

        held = (await client.get(f"/api/runs/{run_id}")).json()
        assert held["execution_status"] == "paused"
        assert "publication outcome is unknown" in held["error"]
        action_id = held["approval"]["action_id"]
        assert platform.barrier.get_action(action_id).status is RuntimeStatus.UNKNOWN

        async with store.connection() as db:
            await db.execute(
                "UPDATE procrastinate_jobs SET scheduled_at=now() WHERE status='todo'"
            )
        await drain(worker)
        again = (await client.get(f"/api/runs/{run_id}")).json()
        assert again["status"] == "awaiting_approval"
        assert again["approval"]["decision"] is None
        assert platform.barrier.get_action(action_id).status is RuntimeStatus.PENDING

        decided = await client.post(f"/api/runs/{run_id}/decision", json=approve)
        assert decided.status_code == 200
        await drain(worker)
        done = (await client.get(f"/api/runs/{run_id}")).json()
        assert done["status"] == "approved"
        assert done["published_at"] is not None
        assert calls["publish"] == 2
        assert platform.barrier.get_action(action_id).status is RuntimeStatus.SUCCEEDED

    async with store.connection() as db:
        decisions = await (
            await db.execute(
                "SELECT decision, action_id FROM decisions WHERE run_id=%s",
                (UUID(run_id),),
            )
        ).fetchall()
    assert decisions == [("approve", action_id), ("approve", action_id)]


async def test_crashed_worker_resumes_from_the_postgres_checkpoint(platform: Platform):
    model = blocking(
        EvidenceScript(drafts=[synthesis(invalid_quote=True), synthesis()]), block_at=4
    )
    run_id, _ = await platform.store.admit_run(
        RunRequest(**RUN), idempotency_key=None, enqueue=enqueuer(platform.jobs)
    )
    async with platform.worker(model) as first:
        crashed = asyncio.create_task(first.runner.execute(run_id))
        await asyncio.wait_for(model.entered.wait(), 60)
        crashed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await crashed
    assert len(model.calls) == 3
    model.release.set()

    async with platform.worker(model) as second:
        outcome = await second.runner.execute(run_id, attempt=1)

    assert outcome == "awaiting_approval"
    assert len(model.calls) == 5
    view = await platform.store.get_run(run_id)
    assert view is not None
    assert [item.action for item in view.timeline] == ["replan"]


async def test_api_lifespan_runs_a_worker_with_registered_mcp_servers(
    platform: Platform, monkeypatch
):
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    monkeypatch.setattr(runtime_module, "chat_model", lambda: model)
    monkeypatch.setattr(runtime_module, "mcp_servers", fake_mcp_servers)
    app = create_app(
        settings=replace(platform.settings, worker_in_api=True),
        resources=ApiResources(
            store=platform.store,
            jobs=platform.jobs,
            approvals=ApprovalDesk(platform.barrier),
        ),
        authenticator=StaticAuthenticator(TEST_PRINCIPAL),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://evidence.test",
            headers=AUTHORIZATION,
        ) as client,
    ):
        run_id = (await client.post("/api/runs", json=RUN)).json()["run_id"]
        await wait_for(client, run_id, "awaiting_approval")
        prompt = "\n".join(str(message.content) for message in model.calls[0][0])
        assert "Echo: continuous-evidence-ok" in prompt
        await client.post(f"/api/runs/{run_id}/decision", json={"decision": "approve"})
        done = await wait_for(client, run_id, "approved", execution_status="completed")
    assert done["published_at"] is not None


async def wait_for(
    client: httpx.AsyncClient,
    run_id: str,
    status: str,
    *,
    execution_status: str | None = None,
    within: float = 60,
) -> dict[str, Any]:
    deadline = time.monotonic() + within
    while True:
        view = (await client.get(f"/api/runs/{run_id}")).json()
        if view["status"] == status and execution_status in (
            None,
            view["execution_status"],
        ):
            return view
        if time.monotonic() > deadline:
            raise AssertionError(
                f"run stayed {view['status']}/{view['execution_status']}"
            )
        await asyncio.sleep(0.2)


async def test_stalled_jobs_are_recovered_and_run(platform: Platform):
    jobs = platform.jobs
    run_id, _ = await platform.store.admit_run(
        RunRequest(**RUN), idempotency_key=None, enqueue=enqueuer(jobs)
    )
    dead_worker = await jobs.job_manager.register_worker()
    stalled = await jobs.job_manager.fetch_job(
        queues=[platform.settings.queue], worker_id=dead_worker
    )
    assert stalled is not None
    async with platform.store.connection() as db:
        await db.execute(
            """UPDATE procrastinate_workers
               SET last_heartbeat = now() - interval '10 minutes' WHERE id = %s""",
            (dead_worker,),
        )
    model = ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()]))
    async with platform.worker(model) as worker:
        context = JobContext(
            app=jobs,
            job=Job(
                id=0,
                queue=platform.settings.queue,
                lock=None,
                queueing_lock=RECOVER_TASK,
                task_name=RECOVER_TASK,
            ),
            start_timestamp=time.time(),
            abort_reason=lambda: None,
            additional_context={RUNNER_CONTEXT: worker.runner},
        )
        assert await jobs.tasks[RECOVER_TASK](context, timestamp=0) == 1
        await drain(worker)

    view = await platform.store.get_run(run_id)
    assert view is not None
    assert view.status == "awaiting_approval"
