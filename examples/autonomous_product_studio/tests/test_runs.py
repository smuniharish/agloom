"""End-to-end run lifecycle through the API, the queue, and the embedded worker."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from agentbarrier.runtime import PolicyEffect, PolicyRule, RuntimePolicy
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend.app import create_app
from backend.auth import StaticAuthenticator
from backend.governance import RELEASE_TOOL, default_policy
from backend.safety import RunWorkspace, SafeWorkspace
from backend.services import StudioServices
from tests.conftest import AUTHORIZATION, TEST_PRINCIPAL
from tests.support import (
    PREVIEW_URL,
    RecordingBoundary,
    memory_services,
    scripted,
    tool_call,
)

APP_SOURCE = "export default function App() { return <h1>Pulseboard</h1>; }"
BACKEND_SOURCE = "@app.get('/api/health')\ndef health(): return {'ok': True}\n"
# Ceiling for waits on the background worker; slow or busy CI runners need slack,
# and every wait returns as soon as its condition holds.
WORKER_WAIT_SECONDS = 60
CRITERIA = [
    {
        "id": "AC-1",
        "description": "React entry point exists",
        "kind": "file",
        "target": "frontend/src/App.tsx",
    },
    {
        "id": "AC-2",
        "description": "Backend exposes a health endpoint",
        "kind": "content",
        "target": "backend/app.py",
        "expected": "/api/health",
    },
]


def build_and_publish(summary: str = "Pulseboard skeleton") -> list[Any]:
    return [
        tool_call("read_validation_feedback"),
        tool_call(
            "write_workspace_file", path="frontend/src/App.tsx", content=APP_SOURCE
        ),
        tool_call(
            "write_workspace_file", path="backend/app.py", content=BACKEND_SOURCE
        ),
        tool_call("publish_release", summary=summary),
    ]


def client_for(services: Any) -> TestClient:
    app = create_app(services, StaticAuthenticator(TEST_PRINCIPAL))
    return TestClient(app, headers=AUTHORIZATION)


def start_run(
    client: TestClient, criteria: list[dict[str, Any]] = CRITERIA
) -> tuple[dict, dict]:
    project = client.post(
        "/api/projects",
        json={
            "name": "Pulseboard",
            "objective": "Build a project health dashboard and API",
            "acceptance_criteria": criteria,
        },
    ).json()
    task = client.post(
        f"/api/projects/{project['id']}/tasks",
        json={"title": "Build", "objective": "Build and validate the product"},
    ).json()
    accepted = client.post(f"/api/tasks/{task['id']}/run")
    assert accepted.status_code == 202, accepted.text
    return project, accepted.json()


def wait_for(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    timeout: float = WORKER_WAIT_SECONDS,
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in statuses:
            return run
        if time.monotonic() > deadline:
            raise AssertionError(f"run stayed {run['status']}: {run}")
        time.sleep(0.05)


def event_types(client: TestClient, run_id: str) -> list[str]:
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        frames = [line for line in response.iter_lines() if line.startswith("data: ")]
    return [json.loads(frame.removeprefix("data: "))["type"] for frame in frames]


def test_release_waits_for_approval_then_publishes_exactly_once(tmp_path: Path) -> None:
    model = scripted(*build_and_publish(), AIMessage(content="Release published."))
    with client_for(memory_services(tmp_path, model)) as client:
        project, accepted = start_run(client)
        assert accepted["status"] == "queued"
        run_id = accepted["id"]
        paused = wait_for(client, run_id, {"awaiting_approval", "failed"})
        assert paused["status"] == "awaiting_approval", paused
        assert paused["proof"]["passed_criteria"] == 2
        assert paused["proof"]["tool_calls"] == 3
        assert paused["proof"]["model_calls"] == 4
        active = tmp_path / "projects" / project["id"]
        assert not (active / "frontend").exists()

        [approval] = client.get(f"/api/runs/{run_id}/approvals").json()
        assert approval["tool"] == "publish_release"
        assert approval["status"] == "pending"
        assert approval["policy_rule"] == "release review"
        assert approval["arguments"] == {"summary": "Pulseboard skeleton"}

        decided = client.post(
            f"/api/runs/{run_id}/approvals/{approval['action_id']}",
            json={"decision": "approve", "reason": "looks good"},
        )
        assert decided.status_code == 200, decided.text
        done = wait_for(client, run_id, {"completed", "failed"})
        types = event_types(client, run_id)
        [executed] = client.get(f"/api/runs/{run_id}/approvals").json()

    assert done["status"] == "completed", done
    assert done["proof"]["passed_criteria"] == 2
    assert done["proof"]["validation_pass_rate"] == 1
    assert done["proof"]["iterations"] == 1
    assert done["proof"]["model_calls"] == 5
    assert done["proof"]["tool_calls"] == 4
    assert done["proof"]["elapsed_seconds"] > paused["proof"]["elapsed_seconds"] > 0
    assert done["proof"]["termination_reason"] == "validated"
    assert done["result"]["release"]["files"] == 2
    assert (active / "frontend" / "src" / "App.tsx").read_text() == APP_SOURCE
    assert not (active / ".agloom").exists()
    assert executed["status"] == "succeeded"
    assert executed["decided_by"] == TEST_PRINCIPAL.subject
    for expected in (
        "queued",
        "running",
        "awaiting_approval",
        "approval_decided",
        "resumed",
        "published",
        "completed",
    ):
        assert expected in types, types
    assert types.count("published") == 1


def approve_pending(client: TestClient, run_id: str) -> dict[str, Any]:
    [approval] = client.get(f"/api/runs/{run_id}/approvals").json()
    assert (approval["kind"], approval["status"]) == ("approval", "pending")
    decided = client.post(
        f"/api/runs/{run_id}/approvals/{approval['action_id']}",
        json={"decision": "approve"},
    )
    assert decided.status_code == 200, decided.text
    return approval


def test_a_failed_release_that_did_not_take_effect_runs_once_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    with client_for(memory_services(tmp_path, model)) as client:
        project, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_approval"})
        approval = approve_pending(client, run_id)
        unknown = wait_for(client, run_id, {"awaiting_reconciliation", "failed"})
        [item] = client.get(f"/api/runs/{run_id}/approvals").json()
        refused_cancel = client.post(f"/api/runs/{run_id}/cancel")
        wrong_endpoint = client.post(
            f"/api/runs/{run_id}/approvals/{item['action_id']}",
            json={"decision": "approve"},
        )
        reconciled = client.post(
            f"/api/runs/{run_id}/reconciliations/{item['action_id']}",
            json={"outcome": "not_committed", "reason": "the workspace is unchanged"},
        )
        # agentbarrier puts a release that did not happen back up for approval.
        wait_for(client, run_id, {"awaiting_approval"})
        approve_pending(client, run_id)
        done = wait_for(client, run_id, {"completed", "failed"})
        types = event_types(client, run_id)

    assert unknown["status"] == "awaiting_reconciliation", unknown
    assert item["kind"] == "reconciliation"
    assert item["status"] == "unknown"
    assert item["action_id"] == approval["action_id"]
    assert item["error"]
    assert refused_cancel.status_code == 409
    assert wrong_endpoint.status_code == 404
    assert reconciled.status_code == 200, reconciled.text
    assert done["status"] == "completed", done
    assert attempts == ["publish", "publish"]
    assert types.count("published") == 1
    assert "awaiting_reconciliation" in types
    assert "reconciled" in types
    active = tmp_path / "projects" / project["id"]
    assert (active / "frontend" / "src" / "App.tsx").read_text() == APP_SOURCE


def test_a_failed_release_that_took_effect_replays_the_operator_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failures: list[str] = []
    write_marker = SafeWorkspace.write_release_marker

    def marker_then_failure(self: SafeWorkspace, payload: Mapping[str, Any]) -> Path:
        path = write_marker(self, payload)
        if not failures:
            failures.append("fsync")
            raise OSError("fsync failed after the release")
        return path

    monkeypatch.setattr(SafeWorkspace, "write_release_marker", marker_then_failure)
    model = scripted(*build_and_publish(), AIMessage(content="Release published."))
    with client_for(memory_services(tmp_path, model)) as client:
        project, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_approval"})
        approve_pending(client, run_id)
        wait_for(client, run_id, {"awaiting_reconciliation"})
        [item] = client.get(f"/api/runs/{run_id}/approvals").json()
        reconciled = client.post(
            f"/api/runs/{run_id}/reconciliations/{item['action_id']}",
            json={
                "outcome": "committed",
                "reason": "the project workspace contains the release",
                "result": {"published": True, "files": 2},
            },
        )
        again = client.post(
            f"/api/runs/{run_id}/reconciliations/{item['action_id']}",
            json={"outcome": "committed", "reason": "a second time"},
        )
        done = wait_for(client, run_id, {"completed", "failed"})
        [after] = client.get(f"/api/runs/{run_id}/approvals").json()

    assert reconciled.status_code == 200, reconciled.text
    assert again.status_code == 409
    assert done["status"] == "completed", done
    assert failures == ["fsync"]
    assert after["status"] == "succeeded"
    active = tmp_path / "projects" / project["id"]
    assert (active / "backend" / "app.py").read_text() == BACKEND_SOURCE


def allow_releases() -> RuntimePolicy:
    base = default_policy(PREVIEW_URL)
    rules = tuple(
        PolicyRule("automatic release", PolicyEffect.ALLOW, tool=RELEASE_TOOL)
        if rule.tool == RELEASE_TOOL
        else rule
        for rule in base.rules
    )
    return RuntimePolicy(version="studio-automatic-release", rules=rules)


def with_policy(
    services: Callable[[], Any], policy: RuntimePolicy
) -> Callable[[], Any]:
    @asynccontextmanager
    async def factory() -> AsyncIterator[StudioServices]:
        async with services() as opened:
            opened.policy = policy
            yield opened

    return factory


def test_the_requester_cannot_reconcile_their_own_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(self: RunWorkspace, guard: Any) -> list[str]:
        raise OSError("no space left on device")

    monkeypatch.setattr(RunWorkspace, "publish", disk_full)
    model = scripted(*build_and_publish(), AIMessage(content="unused"))
    services = memory_services(tmp_path, model, require_separate_approver=True)
    with client_for(with_policy(services, allow_releases())) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_reconciliation"})
        [item] = client.get(f"/api/runs/{run_id}/approvals").json()
        refused = client.post(
            f"/api/runs/{run_id}/reconciliations/{item['action_id']}",
            json={"outcome": "not_committed", "reason": "nothing was published"},
        )
        missing = client.post(
            f"/api/runs/{run_id}/reconciliations/not-an-action",
            json={"outcome": "not_committed", "reason": "nothing was published"},
        )
        still = client.get(f"/api/runs/{run_id}").json()

    assert refused.status_code == 403
    assert "requester" in refused.json()["detail"]
    assert missing.status_code == 404
    assert still["status"] == "awaiting_reconciliation"


def test_rejected_release_ends_the_run_without_publishing(tmp_path: Path) -> None:
    model = scripted(*build_and_publish())
    with client_for(memory_services(tmp_path, model)) as client:
        project, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_approval"})
        [approval] = client.get(f"/api/runs/{run_id}/approvals").json()
        decided = client.post(
            f"/api/runs/{run_id}/approvals/{approval['action_id']}",
            json={"decision": "reject", "reason": "the dashboard needs a chart"},
        )
        assert decided.status_code == 200, decided.text
        rejected = decided.json()
        [after] = client.get(f"/api/runs/{run_id}/approvals").json()
        deadline = time.monotonic() + WORKER_WAIT_SECONDS
        runs = tmp_path / "projects" / ".runs" / project["id"] / run_id
        while runs.exists() and time.monotonic() < deadline:
            time.sleep(0.05)

    assert rejected["status"] == "rejected"
    assert rejected["result"] == {
        "rejected_by": TEST_PRINCIPAL.subject,
        "reason": "the dashboard needs a chart",
    }
    assert after["status"] == "rejected"
    assert after["decision_reason"] == "the dashboard needs a chart"
    assert not (tmp_path / "projects" / project["id"]).exists()
    assert not runs.exists()


def test_release_gate_asks_for_approval_only_when_criteria_pass(tmp_path: Path) -> None:
    model = scripted(
        tool_call("publish_release", summary="too early"),
        *build_and_publish("complete build"),
    )
    services = memory_services(tmp_path, model)
    with client_for(services) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        paused = wait_for(client, run_id, {"awaiting_approval", "failed"})
        approvals = client.get(f"/api/runs/{run_id}/approvals").json()
        actions = client.app.state.services.policy_store.list_actions()

    assert paused["status"] == "awaiting_approval"
    assert [item["arguments"] for item in approvals] == [{"summary": "complete build"}]
    assert len(actions) == 1


def test_cancelling_a_queued_run_cancels_its_job(tmp_path: Path) -> None:
    services = memory_services(tmp_path, scripted(), embedded_worker=False)
    with client_for(services) as client:
        project, accepted = start_run(client)
        run_id = accepted["id"]
        cancelled = client.post(f"/api/runs/{run_id}/cancel")
        run = client.get(f"/api/runs/{run_id}").json()
        again = client.post(f"/api/runs/{run_id}/cancel")
        jobs = client.app.state.services.jobs.app.connector.jobs

    assert cancelled.json() == {"status": "cancelled"}
    assert run["status"] == "cancelled"
    assert run["proof"]["termination_reason"] == "cancelled"
    assert again.status_code == 409
    statuses = {job["task_name"]: job["status"] for job in jobs.values()}
    assert statuses["studio.execute_run"] == "cancelled"
    assert not (tmp_path / "projects" / ".runs" / project["id"] / run_id).exists()


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


BACKEND_MANIFEST = """
[project]
name = "generated-api"
version = "0.1.0"
dependencies = ["fastapi>=0.116,<1"]
""".strip()


def test_cancelling_a_running_run_stops_the_agent_at_its_next_step(
    tmp_path: Path,
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
    with client_for(memory_services(tmp_path, model, boundary=boundary)) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        assert boundary.started.wait(timeout=WORKER_WAIT_SECONDS)
        cancelling = client.post(f"/api/runs/{run_id}/cancel")
        time.sleep(0.3)
        boundary.release.set()
        run = wait_for(client, run_id, {"cancelled", "completed", "failed"})
        jobs = client.app.state.services.jobs.app.connector.jobs

    assert cancelling.json() == {"status": "cancelling"}
    assert run["status"] == "cancelled"
    assert run["proof"]["model_calls"] == 2
    assert len(boundary.calls) == 1
    statuses = {job["task_name"]: job["status"] for job in jobs.values()}
    assert statuses["studio.execute_run"] == "aborted"


def test_cancelling_a_paused_run_rejects_its_pending_release(tmp_path: Path) -> None:
    with client_for(
        memory_services(tmp_path, scripted(*build_and_publish()))
    ) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        wait_for(client, run_id, {"awaiting_approval"})
        cancelled = client.post(f"/api/runs/{run_id}/cancel")
        [approval] = client.get(f"/api/runs/{run_id}/approvals").json()
        run = client.get(f"/api/runs/{run_id}").json()

    assert cancelled.json() == {"status": "cancelled"}
    assert run["status"] == "cancelled"
    assert approval["status"] == "rejected"
    assert approval["decision_reason"] == "the run was cancelled"


def test_idempotency_key_returns_the_existing_run(tmp_path: Path) -> None:
    services = memory_services(tmp_path, scripted(), embedded_worker=False)
    with client_for(services) as client:
        project = client.post(
            "/api/projects",
            json={
                "name": "Keyed",
                "objective": "Idempotent run admission",
                "acceptance_criteria": CRITERIA,
            },
        ).json()
        task = client.post(
            f"/api/projects/{project['id']}/tasks",
            json={"title": "Build", "objective": "Build and validate the product"},
        ).json()
        headers = {"Idempotency-Key": "build-1"}
        first = client.post(f"/api/tasks/{task['id']}/run", headers=headers)
        second = client.post(f"/api/tasks/{task['id']}/run", headers=headers)
        conflict = client.post(f"/api/tasks/{task['id']}/run")
        missing = client.post("/api/tasks/00000000-0000-0000-0000-000000000000/run")
        jobs = client.app.state.services.jobs.app.connector.jobs

    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert conflict.status_code == 409
    assert missing.status_code == 404
    assert len(jobs) == 1


def test_loop_budget_without_a_release_fails_the_run(tmp_path: Path) -> None:
    model = scripted(*[AIMessage(content=f"attempt {index}") for index in range(5)])
    with client_for(memory_services(tmp_path, model)) as client:
        _, accepted = start_run(client)
        run = wait_for(client, accepted["id"], {"completed", "failed"})
        evidence = client.get(f"/api/runs/{accepted['id']}/evidence").json()

    assert run["status"] == "failed"
    # Three iterations without progress stagnate; the policy allows one replan first.
    assert run["proof"]["iterations"] == 4
    assert run["proof"]["termination_reason"] == "stagnation"
    assert run["proof"]["passed_criteria"] == 0
    assert {item["iteration"] for item in evidence} == {1, 2, 3, 4}


def test_event_stream_resumes_after_the_last_event_id(tmp_path: Path) -> None:
    model = scripted(*build_and_publish(), AIMessage(content="done"))
    with client_for(memory_services(tmp_path, model)) as client:
        _, accepted = start_run(client)
        run_id = accepted["id"]
        paused = wait_for(client, run_id, {"awaiting_approval"})
        del paused
        [approval] = client.get(f"/api/runs/{run_id}/approvals").json()
        client.post(
            f"/api/runs/{run_id}/approvals/{approval['action_id']}",
            json={"decision": "approve"},
        )
        wait_for(client, run_id, {"completed"})
        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            frames = [line for line in response.iter_lines() if line.startswith("id: ")]
        ids = [int(frame.removeprefix("id: ")) for frame in frames]
        with client.stream(
            "GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(ids[-2])}
        ) as response:
            tail = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert ids == sorted(ids)
    assert [event["id"] for event in tail] == [ids[-1]]
    assert tail[0]["type"] == "completed"
