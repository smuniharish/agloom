"""Run execution: continuation from checkpoints and failure handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from agentbarrier.runtime import (
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from backend.agent import MODEL_CALLS_PER_ITERATION
from backend.auth import Principal
from backend.execution import RunExecution
from backend.governance import RELEASE_TOOL, default_policy
from backend.jobs import JobQueue, job_app
from backend.models import ProjectCreate, TaskCreate
from backend.safety import RunWorkspace, SafeWorkspace
from backend.services import StudioServices
from backend.store import MemoryStudioStore
from backend.streaming import RunEventBroker
from tests.conftest import TEST_PRINCIPAL
from tests.support import (
    PREVIEW_URL,
    RecordingBoundary,
    memory_settings,
    scripted,
    tool_call,
)

pytestmark = pytest.mark.anyio

MANIFEST = (
    '[project]\nname = "api"\nversion = "0.1.0"\ndependencies = ["fastapi>=0.116,<1"]\n'
)


class Crash(BaseException):
    """Stands in for the worker process dying in the middle of a tool call."""


class CrashOnceBoundary(RecordingBoundary):
    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    def run(
        self, workspace: Path, commands: list[tuple[str, ...]], timeout: int
    ) -> dict:
        if not self.crashed:
            self.crashed = True
            raise Crash("worker died")
        return super().run(workspace, commands, timeout)


def allow_releases() -> RuntimePolicy:
    base = default_policy(PREVIEW_URL)
    rules = tuple(
        PolicyRule("automatic release", PolicyEffect.ALLOW, tool=RELEASE_TOOL)
        if rule.tool == RELEASE_TOOL
        else rule
        for rule in base.rules
    )
    return RuntimePolicy(version="studio-test", rules=rules)


@contextmanager
def services_for(
    tmp_path: Path, model: Any, boundary: Any, policy: RuntimePolicy
) -> Iterator[StudioServices]:
    store = MemoryStudioStore()
    policy_store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
    try:
        yield StudioServices(
            settings=memory_settings(tmp_path),
            store=store,
            events=RunEventBroker(store),
            jobs=JobQueue(job_app),
            policy=policy,
            policy_store=policy_store,
            checkpointer=InMemorySaver(),
            model_factory=lambda: model,
            execution_boundary=boundary,
        )
    finally:
        policy_store.close()


async def queued_run(services: StudioServices) -> UUID:
    store = services.store
    project = store.create_project(
        ProjectCreate.model_validate(
            {
                "name": "Resumable",
                "objective": "Build a backend that survives worker crashes",
                "acceptance_criteria": [
                    {
                        "id": "AC-1",
                        "description": "Backend manifest exists",
                        "kind": "file",
                        "target": "backend/pyproject.toml",
                    },
                    {
                        "id": "AC-2",
                        "description": "Backend tests pass",
                        "kind": "command",
                        "target": "backend-test",
                    },
                ],
            }
        )
    )
    task = store.create_task(
        project.id, TaskCreate(title="Build", objective="Build the backend")
    )

    async def enqueue(run_id: UUID, principal: Principal, connection: Any) -> int:
        del run_id, principal, connection
        return 1

    run, _ = await store.schedule_run(task.id, idempotency_key=None, enqueue=enqueue)
    return run.id


async def test_crashed_run_continues_from_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[str] = []
    original = SafeWorkspace.write

    def counting_write(self: SafeWorkspace, relative: str, content: str) -> Path:
        writes.append(relative)
        return original(self, relative, content)

    monkeypatch.setattr(SafeWorkspace, "write", counting_write)
    boundary = CrashOnceBoundary()
    model = scripted(
        tool_call(
            "write_workspace_file", path="backend/pyproject.toml", content=MANIFEST
        ),
        tool_call("run_allowlisted_check", name="backend-test"),
        tool_call(RELEASE_TOOL, summary="resumed release"),
        AIMessage(content="Release published."),
    )
    with services_for(tmp_path, model, boundary, allow_releases()) as services:
        run_id = await queued_run(services)
        with pytest.raises(Crash):
            await RunExecution(services, run_id, TEST_PRINCIPAL).run()
        assert services.store.get_run(run_id).status == "running"

        outcome = await RunExecution(services, run_id, TEST_PRINCIPAL).run()
        run = services.store.get_run(run_id)
        events = [event["type"] for event in services.store.read_events(run_id)]
        [release] = services.policy_store.list_actions()

    assert outcome == "completed"
    assert run.status == "completed"
    assert writes == ["backend/pyproject.toml"]
    assert len(boundary.calls) == 1
    assert run.result["release"]["files"] == 1
    assert release.status.value == "succeeded"
    assert events.count("published") == 1
    assert "resumed" in events
    assert not list((tmp_path / "projects" / ".runs").glob("*/*"))


async def test_one_iteration_can_spend_its_whole_model_call_budget(
    tmp_path: Path,
) -> None:
    inspections = MODEL_CALLS_PER_ITERATION - 4
    model = scripted(
        *(tool_call("inspect_workspace") for _ in range(inspections)),
        tool_call(
            "write_workspace_file", path="backend/pyproject.toml", content=MANIFEST
        ),
        tool_call("run_allowlisted_check", name="backend-test"),
        tool_call(RELEASE_TOOL, summary="thorough release"),
        AIMessage(content="Release published."),
    )
    with services_for(
        tmp_path, model, RecordingBoundary(), allow_releases()
    ) as services:
        run_id = await queued_run(services)
        outcome = await RunExecution(services, run_id, TEST_PRINCIPAL).run()
        run = services.store.get_run(run_id)

    assert outcome == "completed"
    assert run.proof.iterations == 1
    assert run.proof.tool_calls == inspections + 3
    assert run.proof.model_calls == MODEL_CALLS_PER_ITERATION


async def test_finished_runs_are_not_executed_again(tmp_path: Path) -> None:
    model = scripted(AIMessage(content="unused"))
    with services_for(
        tmp_path, model, RecordingBoundary(), default_policy(PREVIEW_URL)
    ) as services:
        run_id = await queued_run(services)
        services.store.finish_run(
            run_id, "cancelled", services.store.get_run(run_id).proof, {}
        )
        outcome = await RunExecution(services, run_id, TEST_PRINCIPAL).run()

    assert outcome == "skipped"


async def test_failures_outside_the_loop_retry_then_fail_the_run(
    tmp_path: Path,
) -> None:
    with services_for(
        tmp_path, scripted(), RecordingBoundary(), default_policy(PREVIEW_URL)
    ) as services:
        run_id = await queued_run(services)
        services.store.mark_running(run_id)
        execution = RunExecution(services, run_id, TEST_PRINCIPAL)
        await execution.record_failure(
            ConnectionError("database restarted"), retrying=True, attempt=1
        )
        assert services.store.get_run(run_id).status == "running"
        await execution.record_failure(
            ConnectionError("database restarted"), retrying=False, attempt=5
        )
        run = services.store.get_run(run_id)
        events = services.store.read_events(run_id)

    assert run.status == "failed"
    assert run.result == {"error": "ConnectionError: database restarted"}
    assert run.proof.termination_reason == "execution_failed"
    assert [event["type"] for event in events] == ["retrying", "failed"]
    assert events[0]["attempt"] == 2


async def test_an_unsupported_pause_fails_the_run(tmp_path: Path) -> None:
    with services_for(
        tmp_path, scripted(), RecordingBoundary(), default_policy(PREVIEW_URL)
    ) as services:
        run_id = await queued_run(services)
        run = services.store.mark_running(run_id)
        task = services.store.get_task(run.task_id)
        workspace = RunWorkspace.open(
            services.settings.projects_root, task.project_id, run_id
        )
        breakpoint_pause = SimpleNamespace(
            interrupts=[SimpleNamespace(value={"type": "agloom.breakpoint"})]
        )
        status = await RunExecution(services, run_id, TEST_PRINCIPAL)._pause(
            run, workspace, breakpoint_pause, run.proof
        )
        finished = services.store.get_run(run_id)

    assert status == "failed"
    assert finished.status == "failed"
    assert finished.proof.termination_reason == "unsupported_pause"
    assert "agloom.breakpoint" in finished.result["error"]
    assert not workspace.run_root.exists()
