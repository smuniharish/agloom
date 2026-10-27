from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
import pytest
from agentbarrier.runtime import RuntimeStatus, SQLiteRuntimeStore
from backend import app as application
from backend import observability
from backend.app import create_app, ensure_safe_target
from backend.auth import Principal, StaticAuthenticator
from backend.config import Settings
from backend.jobs import RUN_TASK, RUNNER_CONTEXT
from backend.policy import ApprovalDesk
from backend.runtime import ApiResources
from fastapi import HTTPException
from langgraph.checkpoint.memory import InMemorySaver
from procrastinate import App

from tests.support import (
    AUTHORIZATION,
    TEST_PRINCIPAL,
    EvidenceScript,
    MemoryStore,
    ScriptedChatModel,
    synthesis,
)
from tests.test_runner import make_retries_due, runner_for

RUN = {
    "objective": "Verify Acme security and incident response evidence.",
    "topics": ["encryption", "incident response"],
}


class TokenMapAuthenticator:
    def verify_configuration(self) -> None:
        return None

    def authenticate(self, token: str) -> Principal:
        if token not in {"tenant-a", "tenant-b"}:
            raise HTTPException(401, "invalid bearer token")
        return Principal("user-1", token)


@dataclass
class Api:
    client: httpx.AsyncClient
    store: MemoryStore
    jobs: App
    barrier: SQLiteRuntimeStore
    saver: InMemorySaver

    async def work(self, model: ScriptedChatModel) -> None:
        runner = runner_for(self.store, model, self.barrier, saver=self.saver)
        await self.jobs.run_worker_async(
            queues=["evidence"],
            wait=False,
            additional_context={RUNNER_CONTEXT: runner},
            install_signal_handlers=False,
        )


async def open_api(
    settings: Settings,
    jobs: App,
    barrier: SQLiteRuntimeStore,
    *,
    authenticator: Any = None,
    headers: dict[str, str] | None = None,
    approvals: ApprovalDesk | None = None,
) -> AsyncIterator[Api]:
    store = MemoryStore()
    app = create_app(
        settings=settings,
        resources=ApiResources(
            store=store, jobs=jobs, approvals=approvals or ApprovalDesk(barrier)
        ),
        authenticator=authenticator or StaticAuthenticator(TEST_PRINCIPAL),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://evidence.test",
            headers=AUTHORIZATION if headers is None else headers,
        ) as client,
    ):
        yield Api(client, store, jobs, barrier, InMemorySaver())


@pytest.fixture
async def api(settings, memory_jobs, barrier) -> AsyncIterator[Api]:
    async for opened in open_api(settings, memory_jobs, barrier):
        yield opened


async def paused_run(api: Api) -> UUID:
    started = await api.client.post("/api/runs", json=RUN)
    assert started.status_code == 202
    run_id = UUID(started.json()["run_id"])
    await api.work(ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()])))
    body = (await api.client.get(f"/api/runs/{run_id}")).json()
    assert body["status"] == "awaiting_approval"
    return run_id


def sse(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


async def test_governance_exposes_the_policy_mcp_servers_and_loop(api: Api):
    body = (await api.client.get("/api/governance")).json()
    assert {"evidence-filesystem", "evidence-everything"} <= set(body["mcp_servers"])
    assert body["embedding_model"] == "all-minilm"
    assert body["loop"]["max_iterations"] == 5
    assert body["approval_required"] is True
    assert body["consequential_tools"] == ["publish_evidence"]
    assert body["policy"]["default"] == "deny"
    first = body["policy"]["rules"][0]
    assert first == {
        "name": "reviewer approves every evidence publication",
        "effect": "require_approval",
        "tool": "publish_evidence",
    }


async def test_health_and_readiness(api: Api):
    health = await api.client.get("/api/health", headers={})
    assert health.status_code == 200
    assert health.json()["observability"]["metrics_port"] == 9470
    assert (await api.client.get("/api/ready")).json() == {"status": "ready"}
    api.store.available = False
    unavailable = await api.client.get("/api/ready")
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"] == "database unavailable"


def test_langfuse_tracing_uses_the_sdk_base_url(monkeypatch):
    configured: dict[str, Any] = {}

    class Prometheus:
        def __init__(self, **options: Any) -> None: ...

        def start_http_server(self, port: int, address: str) -> None: ...

    class Langfuse:
        def __init__(self, **options: Any) -> None:
            configured.update(options)

    monkeypatch.setattr(observability, "_PROCESS_TELEMETRY", None)
    monkeypatch.setattr(observability, "PrometheusObserver", Prometheus)
    monkeypatch.setattr(observability, "LangfuseObserver", Langfuse)
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.test:3000")

    observability.process_telemetry("worker")

    assert configured["base_url"] == "http://langfuse.test:3000"
    assert configured["event_metadata"]["component"] == "worker"


async def test_api_process_serves_metrics_for_prometheus(
    settings, memory_jobs, barrier, monkeypatch
):
    served: list[tuple[int, str]] = []
    stopped: list[bool] = []

    class Prometheus:
        def __init__(self, **options: Any) -> None: ...

        def start_http_server(self, port: int, address: str) -> None:
            served.append((port, address))

        def flush(self) -> None: ...

        def shutdown(self) -> None:
            stopped.append(True)

    monkeypatch.setattr(observability, "_PROCESS_TELEMETRY", None)
    monkeypatch.setattr(observability, "PrometheusObserver", Prometheus)
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("METRICS_HOST", "0.0.0.0")
    for name in ("METRICS_PORT", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    async for api in open_api(settings, memory_jobs, barrier):
        assert served == [(9470, "0.0.0.0")]
        health = (await api.client.get("/api/health", headers={})).json()
        assert health["observability"]["prometheus"] is True
    assert stopped == [True]


async def test_every_business_endpoint_requires_a_bearer_token(api: Api):
    anonymous = await api.client.get("/api/governance", headers={"Authorization": ""})
    assert anonymous.status_code == 401
    invalid = await api.client.get(
        "/api/governance", headers={"Authorization": "Bearer wrong"}
    )
    assert invalid.status_code == 401


async def test_admission_queues_one_job_per_idempotency_key(api: Api):
    headers = {"Idempotency-Key": "submit-1"}
    first = await api.client.post("/api/runs", json=RUN, headers=headers)
    assert first.status_code == 202
    body = first.json()
    assert body["status"] == "running"
    assert body["execution_status"] == "queued"
    duplicate = await api.client.post("/api/runs", json=RUN, headers=headers)
    assert duplicate.json()["run_id"] == body["run_id"]
    jobs = list(api.jobs.connector.jobs.values())
    assert len(jobs) == 1
    assert jobs[0]["task_name"] == RUN_TASK
    assert jobs[0]["args"] == {
        "run_id": body["run_id"],
        "tenant_id": TEST_PRINCIPAL.tenant_id,
        "user_id": TEST_PRINCIPAL.subject,
    }
    assert api.store.event_types(UUID(body["run_id"])) == ["queued"]


async def test_admission_reports_unavailable_storage(api: Api):
    api.store.available = False
    response = await api.client.post("/api/runs", json=RUN)
    assert response.status_code == 503
    assert "unable to schedule evidence run" in response.json()["detail"]


async def test_unsafe_targets_are_rejected(api: Api, monkeypatch):
    monkeypatch.setattr(
        application.socket,
        "getaddrinfo",
        lambda *_: [(None, None, None, None, ("127.0.0.1", 0))],
    )
    for url in (
        "http://localhost/admin",
        "https://user@example.com",
        "https://:password@example.com",
        "http://target-site:443/",
    ):
        with pytest.raises(ValueError, match="target"):
            ensure_safe_target(url)
    ensure_safe_target("http://target-site")
    response = await api.client.post(
        "/api/runs", json={**RUN, "target_url": "http://localhost/admin"}
    )
    assert response.status_code == 400


async def test_review_flow_through_the_api(api: Api):
    run_id = await paused_run(api)
    events = sse((await api.client.get(f"/api/runs/{run_id}/events")).text)
    assert events[0]["type"] == "queued"
    assert events[-1]["type"] == "awaiting_approval"
    last_id = events[-1]["id"]

    decided = await api.client.post(
        f"/api/runs/{run_id}/decision", json={"decision": "approve", "note": "ok"}
    )
    assert decided.status_code == 200
    assert decided.json()["run"]["approval"]["decision"] == "approve"
    assert decided.json()["run"]["execution_status"] == "queued"
    again = await api.client.post(
        f"/api/runs/{run_id}/decision", json={"decision": "reject"}
    )
    assert again.status_code == 409

    await api.work(ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()])))
    view = (await api.client.get(f"/api/runs/{run_id}")).json()
    assert view["status"] == "approved"
    assert view["objective_state"] == "validated"
    assert view["approval"]["decided_by"] == TEST_PRINCIPAL.subject
    replay = await api.client.get(
        f"/api/runs/{run_id}/events",
        headers={**AUTHORIZATION, "Last-Event-ID": str(last_id)},
    )
    replayed = [event["type"] for event in sse(replay.text)]
    assert replayed[0] == "decision"
    assert "published" in replayed
    assert replayed[-1] == "completed"


async def test_separate_approvers_can_be_required(settings, memory_jobs, barrier):
    approvals = ApprovalDesk(barrier, require_separate_approver=True)
    async for api in open_api(settings, memory_jobs, barrier, approvals=approvals):
        run_id = await paused_run(api)
        response = await api.client.post(
            f"/api/runs/{run_id}/decision", json={"decision": "approve"}
        )
        assert response.status_code == 403
        view = (await api.client.get(f"/api/runs/{run_id}")).json()
        assert view["status"] == "awaiting_approval"


async def test_decisions_require_a_pending_review(api: Api):
    started = (await api.client.post("/api/runs", json=RUN)).json()
    response = await api.client.post(
        f"/api/runs/{started['run_id']}/decision", json={"decision": "approve"}
    )
    assert response.status_code == 409


async def test_cancel_queued_run(api: Api):
    started = (await api.client.post("/api/runs", json=RUN)).json()
    cancelled = await api.client.post(f"/api/runs/{started['run_id']}/cancel")
    assert cancelled.json() == {"status": "cancelled"}
    job = next(iter(api.jobs.connector.jobs.values()))
    assert job["status"] == "cancelled"
    view = (await api.client.get(f"/api/runs/{started['run_id']}")).json()
    assert view["status"] == "cancelled"
    assert (
        await api.client.post(f"/api/runs/{started['run_id']}/cancel")
    ).status_code == 409


async def test_cancel_running_run_requests_abort(api: Api):
    started = (await api.client.post("/api/runs", json=RUN)).json()
    job = next(iter(api.jobs.connector.jobs.values()))
    job["status"] = "doing"
    api.store.runs[UUID(started["run_id"])].view.execution_status = "running"
    response = await api.client.post(f"/api/runs/{started['run_id']}/cancel")
    assert response.json() == {"status": "cancelling"}
    assert job["abort_requested"] is True


async def test_cancel_paused_run_withdraws_the_approval(api: Api):
    run_id = await paused_run(api)
    action_id = (await api.client.get(f"/api/runs/{run_id}")).json()["approval"][
        "action_id"
    ]
    response = await api.client.post(f"/api/runs/{run_id}/cancel")
    assert response.json() == {"status": "cancelled"}
    assert api.barrier.get_action(action_id).status is RuntimeStatus.REJECTED
    view = (await api.client.get(f"/api/runs/{run_id}")).json()
    assert view["status"] == "cancelled"
    assert api.store.event_types(run_id)[-1] == "cancelled"


async def run_with_failed_publication(api: Api, *, committed: bool) -> UUID:
    """Approve a paused run whose publication raises once."""
    publish = api.store.publish
    calls = {"publish": 0}

    async def flaky(run_id: UUID, synthesis: Any, accuracy: Any) -> Any:
        calls["publish"] += 1
        if calls["publish"] == 1:
            if committed:
                await publish(run_id, synthesis, accuracy)
            raise ConnectionError("connection lost during publication")
        return await publish(run_id, synthesis, accuracy)

    api.store.publish = flaky  # type: ignore[method-assign]
    run_id = await paused_run(api)
    decided = await api.client.post(
        f"/api/runs/{run_id}/decision", json={"decision": "approve"}
    )
    assert decided.status_code == 200
    await api.work(ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()])))
    view = (await api.client.get(f"/api/runs/{run_id}")).json()
    assert view["execution_status"] == "paused"
    assert "publication outcome is unknown" in view["error"]
    return run_id


async def test_cancel_run_while_its_publication_is_reconciled(api: Api):
    run_id = await run_with_failed_publication(api, committed=False)

    response = await api.client.post(f"/api/runs/{run_id}/cancel")

    assert response.json() == {"status": "cancelled"}
    view = (await api.client.get(f"/api/runs/{run_id}")).json()
    assert view["status"] == "cancelled"
    assert api.store.publications == []


async def test_published_run_is_not_cancelled_while_reconciled(api: Api):
    run_id = await run_with_failed_publication(api, committed=True)

    response = await api.client.post(f"/api/runs/{run_id}/cancel")

    assert response.status_code == 409
    # The retried job is still queued and settles the publication.
    make_retries_due(api.jobs)
    await api.work(ScriptedChatModel(responder=EvidenceScript(drafts=[synthesis()])))
    view = (await api.client.get(f"/api/runs/{run_id}")).json()
    assert view["status"] == "approved"
    assert view["execution_status"] == "completed"
    assert api.store.publications == [run_id]


async def test_runs_and_events_are_tenant_scoped(settings, memory_jobs, barrier):
    async for api in open_api(
        settings,
        memory_jobs,
        barrier,
        authenticator=TokenMapAuthenticator(),
        headers={},
    ):
        first = {"Authorization": "Bearer tenant-a"}
        second = {"Authorization": "Bearer tenant-b"}
        run_id = (await api.client.post("/api/runs", json=RUN, headers=first)).json()[
            "run_id"
        ]
        assert (
            await api.client.get(f"/api/runs/{run_id}", headers=first)
        ).status_code == 200
        for path, method in (
            (f"/api/runs/{run_id}", "GET"),
            (f"/api/runs/{run_id}/events", "GET"),
            (f"/api/runs/{run_id}/cancel", "POST"),
        ):
            response = await api.client.request(method, path, headers=second)
            assert response.status_code == 404
        denied = await api.client.post(
            f"/api/runs/{run_id}/decision", json={"decision": "approve"}, headers=second
        )
        assert denied.status_code == 404
