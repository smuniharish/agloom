from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import pytest

from backend.auth import Principal, principal_scope
from backend.models import Approval, ProjectCreate, TaskCreate
from backend.store import MemoryStudioStore
from tests.conftest import TEST_PRINCIPAL
from tests.support import KeywordEmbedder


def project_request() -> ProjectCreate:
    return ProjectCreate.model_validate(
        {
            "name": "Demo",
            "objective": "Build a complete frontend and backend",
            "acceptance_criteria": [
                {
                    "id": "AC-1",
                    "description": "Frontend exists",
                    "kind": "file",
                    "target": "frontend/src/App.tsx",
                }
            ],
        }
    )


class Queue:
    def __init__(self) -> None:
        self.jobs: list[tuple[UUID, Principal]] = []

    async def __call__(
        self, run_id: UUID, principal: Principal, connection: Any
    ) -> int:
        del connection
        self.jobs.append((run_id, principal))
        return len(self.jobs)


def test_memory_store_records_the_full_run_lifecycle() -> None:
    store = MemoryStudioStore()
    queue = Queue()
    project = store.create_project(project_request())
    task = store.create_task(
        project.id,
        TaskCreate(title="Build UI", objective="Build the responsive dashboard"),
    )

    run, created = asyncio.run(
        store.schedule_run(task.id, idempotency_key=None, enqueue=queue)
    )
    assert created
    assert (run.status, store.get_task(task.id).status) == ("queued", "queued")
    assert store.run_job_id(run.id) == 1
    assert store.mark_running(run.id).status == "running"
    store.add_evidence(run.id, 1, "AC-1", "filesystem", True, "file exists")
    approval = Approval(
        action_id="a-1", tool="publish_release", policy_rule="release review"
    )
    paused = store.record_pause(
        run.id,
        "awaiting_approval",
        [approval],
        run.proof.model_copy(update={"iterations": 1}),
    )
    assert paused.status == "awaiting_approval"
    assert store.get_task(task.id).status == "awaiting_approval"
    requeued = asyncio.run(store.requeue_run(run.id, enqueue=queue))
    assert requeued.status == "queued"
    assert queue.jobs[-1] == (
        run.id,
        Principal(TEST_PRINCIPAL.subject, TEST_PRINCIPAL.tenant_id),
    )
    store.mark_running(run.id)
    finished = store.finish_run(
        run.id,
        "completed",
        run.proof.model_copy(update={"iterations": 1}),
        {"ok": True},
    )
    assert finished.status == "completed"
    assert store.finish_run(run.id, "failed", run.proof, {}) is None
    assert store.mark_running(run.id) is None
    assert store.get_task(task.id).status == "completed"
    assert store.evidence_for_run(run.id)[0].criterion_id == "AC-1"
    assert store.counts() == {
        "projects": 1,
        "tasks": 1,
        "loop_runs": 1,
        "artifacts": 0,
        "evidence": 1,
    }


def test_requeue_requires_a_paused_run() -> None:
    store = MemoryStudioStore()
    project = store.create_project(project_request())
    task = store.create_task(
        project.id, TaskCreate(title="Build", objective="Build once only")
    )
    run, _ = asyncio.run(
        store.schedule_run(task.id, idempotency_key=None, enqueue=Queue())
    )
    with pytest.raises(ValueError, match="not waiting for a person"):
        asyncio.run(store.requeue_run(run.id, enqueue=Queue()))


def test_a_pause_records_only_waiting_states() -> None:
    store = MemoryStudioStore()
    project = store.create_project(project_request())
    task = store.create_task(
        project.id, TaskCreate(title="Build", objective="Build once only")
    )
    run, _ = asyncio.run(
        store.schedule_run(task.id, idempotency_key=None, enqueue=Queue())
    )
    store.mark_running(run.id)
    unknown = Approval(
        action_id="a-1", tool="publish_release", kind="reconciliation", status="unknown"
    )
    with pytest.raises(ValueError, match="not a waiting state"):
        store.record_pause(run.id, "completed", [unknown], run.proof)

    paused = store.record_pause(run.id, "awaiting_reconciliation", [unknown], run.proof)
    requeued = asyncio.run(store.requeue_run(run.id, enqueue=Queue()))

    assert paused.status == "awaiting_reconciliation"
    assert paused.approvals[0].kind == "reconciliation"
    assert requeued.status == "queued"


def test_vector_retrieval_orders_semantically_closest_artifact() -> None:
    store = MemoryStudioStore(KeywordEmbedder())
    project = store.create_project(project_request())
    store.index_artifact(
        project.id, "checkout.tsx", "tsx", "Checkout conversion dashboard"
    )
    store.index_artifact(project.id, "weather.tsx", "tsx", "Weather forecast panel")
    results = store.search_artifacts(project.id, "checkout flow")
    assert results[0]["path"] == "checkout.tsx"
    assert results[0]["score"] > results[1]["score"]


def test_concurrent_scheduling_creates_exactly_one_active_run() -> None:
    store = MemoryStudioStore()
    project = store.create_project(project_request())
    task = store.create_task(
        project.id, TaskCreate(title="Build", objective="Build once")
    )
    queue = Queue()

    async def claim() -> object:
        try:
            return await store.schedule_run(
                task.id, idempotency_key=None, enqueue=queue
            )
        except ValueError:
            return None

    async def race() -> list[object]:
        return await asyncio.gather(*(claim() for _ in range(8)))

    results = asyncio.run(race())
    assert sum(result is not None for result in results) == 1
    assert len(queue.jobs) == 1


def test_events_are_ordered_and_tenant_scoped() -> None:
    store = MemoryStudioStore()
    project = store.create_project(project_request())
    task = store.create_task(
        project.id, TaskCreate(title="Build", objective="Build and observe")
    )
    run, _ = asyncio.run(
        store.schedule_run(task.id, idempotency_key=None, enqueue=Queue())
    )
    first = store.append_event(run.id, "queued", {"step": 1})
    store.append_event(run.id, "running", {"step": 2})

    assert [event["type"] for event in store.read_events(run.id)] == [
        "queued",
        "running",
    ]
    assert [
        event["step"] for event in store.read_events(run.id, after=first["id"])
    ] == [2]
    with principal_scope(Principal("other", "other-tenant")):
        assert store.read_events(run.id) == []
        assert store.get_run(run.id) is None
        assert store.evidence_for_run(run.id) == []
        with pytest.raises(LookupError):
            store.add_evidence(run.id, 1, "AC-1", "filesystem", True, "forged")
