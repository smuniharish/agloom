"""PostgreSQL-backed integration: procrastinate, the LangGraph checkpointer, and agentbarrier."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from procrastinate import PsycopgConnector
from psycopg_pool import AsyncConnectionPool

from backend.app import create_app
from backend.auth import Principal, StaticAuthenticator
from backend.jobs import JobQueue, job_app, recover_stalled_runs
from backend.models import ProjectCreate, TaskCreate
from backend.observability import JobQueueCollector
from backend.safety import RunWorkspace
from backend.services import StudioServices, open_services
from backend.settings import APP_ROOT, StudioSettings
from backend.store import StudioStore
from backend.worker import run_worker
from tests.conftest import AUTHORIZATION, TEST_PRINCIPAL
from tests.support import PREVIEW_URL, RecordingBoundary, scripted, tool_call
from tests.test_runs import (
    BACKEND_MANIFEST,
    CRITERIA,
    WORKER_WAIT_SECONDS,
    approve_pending,
    build_and_publish,
    start_run,
    wait_for,
)

pytestmark = pytest.mark.postgres
SELECTOR_LOOP = {"loop_factory": asyncio.SelectorEventLoop} if os.name == "nt" else {}


class FlatEmbedder:
    """Deterministic 384-dimensional embeddings for the pgvector columns."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 383 + [1.0] for _ in texts]


def postgres_settings(
    database_url: str, tmp_path: Path, **changes: Any
) -> StudioSettings:
    values: dict[str, Any] = {
        "database_url": database_url,
        "projects_root": tmp_path / "projects",
        "preview_url": PREVIEW_URL,
        "browser_validation": False,
        "embedded_worker": True,
        "worker_poll_interval": 0.1,
        "shutdown_grace_seconds": 5,
    }
    values.update(changes)
    return StudioSettings(**values)


def postgres_services(
    settings: StudioSettings, model: Any, boundary: Any = None
) -> Callable[[], Any]:
    @asynccontextmanager
    async def factory() -> AsyncIterator[StudioServices]:
        async with open_services(settings, run_agents=True) as services:
            services.model_factory = lambda: model
            services.store.embedder = FlatEmbedder()
            services.execution_boundary = boundary
            yield services

    return factory


def postgres_client(services: Callable[[], Any]) -> TestClient:
    app = create_app(services, StaticAuthenticator(TEST_PRINCIPAL))
    return TestClient(app, headers=AUTHORIZATION, backend_options=SELECTOR_LOOP)


def rows(database_url: str, query: str, *params: Any) -> list[tuple[Any, ...]]:
    with psycopg.connect(database_url) as db:
        return db.execute(query, params).fetchall()


@pytest.mark.anyio
async def test_scheduling_defers_the_job_in_the_same_transaction(
    database_url: str,
) -> None:
    async with AsyncConnectionPool(
        database_url, open=False, check=AsyncConnectionPool.check_connection
    ) as pool:
        store = StudioStore(database_url, FlatEmbedder(), async_pool=pool)
        store.open()
        try:
            with job_app.replace_connector(PsycopgConnector()):
                await _admission_scenario(database_url, store, pool)
        finally:
            store.close()


async def _admission_scenario(
    database_url: str, store: StudioStore, pool: AsyncConnectionPool
) -> None:
    async with job_app.open_async(pool=pool):
        queue = JobQueue(job_app)
        project = store.create_project(
            ProjectCreate.model_validate(
                {
                    "name": "Queue",
                    "objective": "Verify transactional admission",
                    "acceptance_criteria": CRITERIA,
                }
            )
        )
        task = store.create_task(
            project.id, TaskCreate(title="Build", objective="Queue admission")
        )

        async def failing(run_id: UUID, principal: Principal, connection: Any) -> int:
            await queue.enqueue_run(run_id, principal, connection)
            raise ConnectionError("broker lost after deferring")

        with pytest.raises(ConnectionError):
            await store.schedule_run(task.id, idempotency_key="k", enqueue=failing)
        assert rows(database_url, "SELECT count(*) FROM studio_loop_runs") == [(0,)]
        assert rows(database_url, "SELECT count(*) FROM procrastinate_jobs") == [(0,)]

        run, created = await store.schedule_run(
            task.id, idempotency_key="k", enqueue=queue.enqueue_run
        )
        again, created_again = await store.schedule_run(
            task.id, idempotency_key="k", enqueue=queue.enqueue_run
        )
        with pytest.raises(ValueError, match="active run"):
            await store.schedule_run(
                task.id, idempotency_key=None, enqueue=queue.enqueue_run
            )
        job_id = store.run_job_id(run.id)
        store.add_evidence(run.id, 1, "AC-1", "filesystem", False, "missing")
        store.add_evidence(run.id, 1, "AC-1", "filesystem", True, "present")
        evidence = store.evidence_for_run(run.id)
        store.append_event(run.id, "queued", {"step": 1})
        store.append_event(run.id, "running", {"step": 2})
        events = store.read_events(run.id)

    assert created
    assert not created_again
    assert again.id == run.id
    [(job_args, lock, queueing_lock, status)] = rows(
        database_url,
        "SELECT args, lock, queueing_lock, status::text FROM procrastinate_jobs WHERE id=%s",
        job_id,
    )
    assert job_args == {
        "run_id": str(run.id),
        "tenant_id": "test-tenant",
        "user_id": "test-user",
    }
    assert lock == queueing_lock == f"run:{run.id}"
    assert status == "todo"
    assert [(item.iteration, item.passed) for item in evidence] == [(1, True)]
    assert [event["type"] for event in events] == ["queued", "running"]
    assert events[0]["id"] < events[1]["id"]


def test_release_approval_flow_on_postgres(database_url: str, tmp_path: Path) -> None:
    model = scripted(*build_and_publish(), AIMessage(content="Release published."))
    settings = postgres_settings(database_url, tmp_path)
    with postgres_client(postgres_services(settings, model)) as client:
        project, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_approval"}, timeout=60)
        [approval] = client.get(f"/api/runs/{run_id}/approvals").json()
        decided = client.post(
            f"/api/runs/{run_id}/approvals/{approval['action_id']}",
            json={"decision": "approve"},
        )
        assert decided.status_code == 200, decided.text
        done = wait_for(client, run_id, {"completed", "failed"}, timeout=60)
        searched = client.get(
            f"/api/projects/{project['id']}/artifacts/search", params={"q": "health"}
        ).json()

    assert done["status"] == "completed", done
    jobs = rows(
        database_url,
        "SELECT task_name, status::text FROM procrastinate_jobs WHERE args->>'run_id' = %s ORDER BY id",
        run_id,
    )
    assert jobs[:2] == [
        ("studio.execute_run", "succeeded"),
        ("studio.execute_run", "succeeded"),
    ]
    assert rows(
        database_url,
        "SELECT status FROM agentbarrier.runtime_actions WHERE tool_name='publish_release'",
    ) == [("succeeded",)]
    assert rows(
        database_url, "SELECT count(*) FROM checkpoints WHERE thread_id=%s", run_id
    ) == [(0,)]
    assert {item["path"] for item in searched} == {
        "frontend/src/App.tsx",
        "backend/app.py",
    }


def test_release_reconciliation_flow_on_postgres(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str] = []
    publish = RunWorkspace.publish

    def disk_full_once(self: RunWorkspace, guard: Any) -> list[str]:
        attempts.append("publish")
        if len(attempts) == 1:
            raise OSError("no space left on device")
        return publish(self, guard)

    monkeypatch.setattr(RunWorkspace, "publish", disk_full_once)
    model = scripted(*build_and_publish(), AIMessage(content="Release published."))
    settings = postgres_settings(database_url, tmp_path)
    with postgres_client(postgres_services(settings, model)) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_approval"})
        approve_pending(client, run_id)
        unknown = wait_for(client, run_id, {"awaiting_reconciliation", "failed"})
        stored = rows(
            database_url,
            "SELECT r.status, t.status FROM studio_loop_runs r "
            "JOIN studio_tasks t ON t.id = r.task_id WHERE r.id = %s",
            run_id,
        )
        [item] = client.get(f"/api/runs/{run_id}/approvals").json()
        reconciled = client.post(
            f"/api/runs/{run_id}/reconciliations/{item['action_id']}",
            json={"outcome": "not_committed", "reason": "the workspace is unchanged"},
        )
        assert reconciled.status_code == 200, reconciled.text
        wait_for(client, run_id, {"awaiting_approval"})
        approve_pending(client, run_id)
        done = wait_for(client, run_id, {"completed", "failed"})

    assert unknown["status"] == "awaiting_reconciliation", unknown
    assert stored == [("awaiting_reconciliation", "awaiting_reconciliation")]
    assert done["status"] == "completed", done
    assert attempts == ["publish", "publish"]
    assert rows(
        database_url,
        "SELECT status FROM agentbarrier.runtime_actions WHERE tool_name='publish_release'",
    ) == [("succeeded",)]
    assert ("reconciliation_not_committed",) in rows(
        database_url,
        "SELECT event FROM agentbarrier.runtime_receipts r "
        "JOIN agentbarrier.runtime_actions a USING (action_id) "
        "WHERE a.tool_name='publish_release'",
    )


class BlockingBoundary(RecordingBoundary):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def run(
        self, workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict:
        self.started.set()
        assert self.release.wait(timeout=WORKER_WAIT_SECONDS)
        return super().run(workspace, commands, timeout)


def test_cancellation_aborts_the_job_and_stops_the_agent(
    database_url: str, tmp_path: Path
) -> None:
    boundary = BlockingBoundary()
    model = scripted(
        tool_call(
            "write_workspace_file",
            path="backend/pyproject.toml",
            content=BACKEND_MANIFEST,
        ),
        tool_call("run_allowlisted_check", name="backend-test"),
        AIMessage(content="never requested"),
    )
    settings = postgres_settings(database_url, tmp_path)
    with postgres_client(postgres_services(settings, model, boundary)) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        assert boundary.started.wait(timeout=WORKER_WAIT_SECONDS)
        cancelling = client.post(f"/api/runs/{run_id}/cancel")
        deadline = time.monotonic() + WORKER_WAIT_SECONDS
        while not rows(
            database_url,
            "SELECT 1 FROM procrastinate_jobs WHERE args->>'run_id'=%s AND abort_requested",
            run_id,
        ):
            assert time.monotonic() < deadline
            time.sleep(0.1)
        time.sleep(0.5)
        boundary.release.set()
        run = wait_for(client, run_id, {"cancelled", "completed", "failed"}, timeout=60)

    assert cancelling.json() == {"status": "cancelling"}
    assert run["status"] == "cancelled"
    assert run["proof"]["model_calls"] == 2
    assert rows(
        database_url,
        "SELECT status::text FROM procrastinate_jobs WHERE args->>'run_id'=%s AND task_name='studio.execute_run'",
        run_id,
    ) == [("aborted",)]


CRASHING_WORKER = textwrap.dedent(
    """
    import asyncio, os, sys
    from pathlib import Path
    from langchain_core.messages import AIMessage
    from backend.safety import SafeWorkspace
    from backend.services import open_services
    from backend.settings import StudioSettings
    from backend.worker import event_loop_factory, run_worker
    from tests.support import RecordingBoundary, scripted, tool_call

    LOG = Path(os.environ["STUDIO_TEST_LOG"])
    original = SafeWorkspace.write

    def logged_write(self, relative, content):
        with LOG.open("a", encoding="utf-8") as stream:
            stream.write(f"write {relative}\\n")
        return original(self, relative, content)

    SafeWorkspace.write = logged_write

    class CrashingBoundary(RecordingBoundary):
        def run(self, workspace, commands, timeout):
            with LOG.open("a", encoding="utf-8") as stream:
                stream.write("check crashed\\n")
            os._exit(73)

    async def main():
        settings = StudioSettings(
            database_url=os.environ["DATABASE_URL"],
            projects_root=Path(os.environ["STUDIO_PROJECTS_ROOT"]),
            policy_file=Path(os.environ["STUDIO_POLICY_FILE"]),
            browser_validation=False,
            worker_poll_interval=0.1,
        )
        model = scripted(
            tool_call("write_workspace_file", path="backend/pyproject.toml", content=os.environ["MANIFEST"]),
            tool_call("run_allowlisted_check", name="backend-test"),
        )
        async with open_services(settings, run_agents=True) as services:
            services.model_factory = lambda: model
            services.execution_boundary = CrashingBoundary()
            await run_worker(services, wait=False, install_signal_handlers=False)

    asyncio.run(main(), loop_factory=event_loop_factory())
    sys.exit(0)
    """
)


@pytest.mark.anyio
async def test_hard_killed_worker_resumes_from_its_checkpoint(
    database_url: str, tmp_path: Path
) -> None:
    log = tmp_path / "effects.log"
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "version": "studio-automatic-release",
                "rules": [
                    {
                        "name": "automatic release",
                        "effect": "allow",
                        "tool": "publish_release",
                    },
                    {"name": "workspace", "effect": "allow", "tool": "*"},
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = postgres_settings(
        database_url, tmp_path, policy_file=policy_file, embedded_worker=False
    )
    criteria = [
        {
            "id": "AC-1",
            "description": "Manifest exists",
            "kind": "file",
            "target": "backend/pyproject.toml",
        },
        {
            "id": "AC-2",
            "description": "Backend tests pass",
            "kind": "command",
            "target": "backend-test",
        },
    ]
    resumed_model = scripted(
        tool_call("publish_release", summary="recovered release"),
        AIMessage(content="Release published."),
    )
    boundary = RecordingBoundary()

    def logged_check(
        workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict:
        with log.open("a", encoding="utf-8") as stream:
            stream.write("check passed\n")
        return RecordingBoundary.run(boundary, workspace, commands, timeout)

    boundary.run = logged_check  # type: ignore[method-assign]
    async with postgres_services(settings, resumed_model, boundary)() as services:
        from backend.auth import principal_scope

        with principal_scope(TEST_PRINCIPAL):
            project = services.store.create_project(
                ProjectCreate.model_validate(
                    {
                        "name": "Crash",
                        "objective": "Survive a worker crash",
                        "acceptance_criteria": criteria,
                    }
                )
            )
            task = services.store.create_task(
                project.id, TaskCreate(title="Build", objective="Build the backend")
            )
            run, _ = await services.store.schedule_run(
                task.id, idempotency_key=None, enqueue=services.jobs.enqueue_run
            )

        environment = {
            **os.environ,
            "DATABASE_URL": database_url,
            "STUDIO_TEST_LOG": str(log),
            "STUDIO_PROJECTS_ROOT": str(settings.projects_root),
            "STUDIO_POLICY_FILE": str(policy_file),
            "MANIFEST": BACKEND_MANIFEST,
            "LANGGRAPH_STRICT_MSGPACK": "true",
        }
        crashed = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", CRASHING_WORKER],
            cwd=APP_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert crashed.returncode == 73, crashed.stderr[-4000:]
        assert rows(
            database_url,
            "SELECT status::text FROM procrastinate_jobs WHERE args->>'run_id'=%s",
            str(run.id),
        ) == [("doing",)]

        # The dead worker stops sending heartbeats; age them instead of waiting.
        with psycopg.connect(database_url) as db:
            db.execute(
                "UPDATE procrastinate_workers SET last_heartbeat = now() - interval '10 minutes'"
            )
        recovered = await recover_stalled_runs(SimpleNamespace(app=job_app), 0)
        assert recovered == 1
        await run_worker(services, wait=False, install_signal_handlers=False)
        with principal_scope(TEST_PRINCIPAL):
            finished = services.store.get_run(run.id)
            events = [event["type"] for event in services.store.read_events(run.id)]

    assert finished.status == "completed", finished
    assert log.read_text(encoding="utf-8").splitlines() == [
        "write backend/pyproject.toml",
        "check crashed",
        "check passed",
    ]
    assert finished.result["release"]["summary"] == "recovered release"
    assert "resumed" in events
    assert rows(
        database_url,
        "SELECT status::text FROM procrastinate_jobs WHERE args->>'run_id'=%s",
        str(run.id),
    ) == [("succeeded",)]


@pytest.mark.anyio
async def test_queue_metrics_report_job_states(database_url: str) -> None:
    async with AsyncConnectionPool(
        database_url, open=False, check=AsyncConnectionPool.check_connection
    ) as pool:
        with job_app.replace_connector(PsycopgConnector()):
            async with job_app.open_async(pool=pool):
                await JobQueue(job_app).enqueue_run(uuid4(), TEST_PRINCIPAL)
    families = {
        family.name: family for family in JobQueueCollector(database_url).collect()
    }
    jobs = families["agloom_autonomous_product_studio_jobs"]
    assert [(sample.labels, sample.value) for sample in jobs.samples] == [
        ({"status": "todo"}, 1.0)
    ]
    assert (
        families["agloom_autonomous_product_studio_oldest_queued_job_seconds"]
        .samples[0]
        .value
        >= 0
    )
