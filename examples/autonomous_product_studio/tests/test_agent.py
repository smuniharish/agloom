"""Agent wiring, the studio policy, and the governed tools."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from agentbarrier.runtime import PolicyEffect, SQLiteRuntimeStore
from agloom import RECONCILIATION_INTERRUPT
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from backend.agent import (
    LOOP_POLICY,
    ReleaseGate,
    RunScope,
    TenantRunAuthorizer,
    browser_observation,
    build_agent,
)
from backend.auth import Principal, principal_scope
from backend.governance import (
    CHECK_NAMES,
    POLICY_NAMESPACE,
    RELEASE_TOOL,
    approvals_from_interrupts,
    default_policy,
    load_policy,
    reconciliation_refusal,
    reconciliations_from_interrupts,
    release_key,
    unsupported_pauses,
)
from backend.mcp import RUN_CODE_CAPABILITY
from backend.models import AcceptanceCriterion, LoopRun, Project, RunProof, Task
from backend.safety import RunWorkspace, SafeWorkspace
from backend.store import MemoryStudioStore
from backend.validation import ObjectiveValidator
from tests.conftest import TEST_PRINCIPAL
from tests.support import PREVIEW_URL, scripted


@pytest.mark.parametrize(
    ("tool", "arguments", "effect"),
    [
        (RELEASE_TOOL, {"summary": "v1"}, PolicyEffect.REQUIRE_APPROVAL),
        ("write_workspace_file", {"path": "frontend/src/App.tsx"}, PolicyEffect.ALLOW),
        (
            "write_workspace_file",
            {"path": ".agloom/playwright.json"},
            PolicyEffect.DENY,
        ),
        (
            "write_workspace_file",
            {"path": "a/../.agloom/release.json"},
            PolicyEffect.DENY,
        ),
        ("read_workspace_file", {"path": "README.md"}, PolicyEffect.ALLOW),
        ("inspect_workspace", {}, PolicyEffect.ALLOW),
        ("read_validation_feedback", {}, PolicyEffect.ALLOW),
        *[
            ("run_allowlisted_check", {"name": name}, PolicyEffect.ALLOW)
            for name in CHECK_NAMES
        ],
        ("run_allowlisted_check", {"name": "rm -rf"}, PolicyEffect.DENY),
        ("validate_preview", {"url": PREVIEW_URL}, PolicyEffect.ALLOW),
        ("playwright__browser_navigate", {"url": PREVIEW_URL}, PolicyEffect.ALLOW),
        (
            "playwright__browser_navigate",
            {"url": f"{PREVIEW_URL}/status"},
            PolicyEffect.ALLOW,
        ),
        (
            "playwright__browser_navigate",
            {"url": f"{PREVIEW_URL}.attacker.invalid"},
            PolicyEffect.DENY,
        ),
        (
            "playwright__browser_navigate",
            {"url": "http://169.254.169.254/"},
            PolicyEffect.DENY,
        ),
        ("playwright__browser_snapshot", {}, PolicyEffect.ALLOW),
        ("playwright__browser_run_code_unsafe", {"code": "x"}, PolicyEffect.DENY),
        ("unknown_tool", {}, PolicyEffect.DENY),
    ],
)
def test_default_policy_decisions(
    tool: str, arguments: dict[str, Any], effect: PolicyEffect
) -> None:
    assert default_policy(PREVIEW_URL).evaluate(tool, arguments).effect is effect


def test_operators_can_supply_a_policy_file(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "version": "studio-autonomous",
                "rules": [
                    {
                        "name": "trusted releases",
                        "effect": "allow",
                        "tool": RELEASE_TOOL,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = load_policy(path, PREVIEW_URL)
    assert policy.version == "studio-autonomous"
    assert policy.evaluate(RELEASE_TOOL, {"summary": "x"}).effect is PolicyEffect.ALLOW
    assert load_policy(None, PREVIEW_URL).version == "studio-v1"


def test_release_key_identifies_the_run_files_and_request(tmp_path: Path) -> None:
    workspace = SafeWorkspace(tmp_path / "run")
    workspace.write("app.py", "print('v1')")
    run_id = uuid4()
    select = release_key(run_id, workspace)
    first = select({"summary": "v1"})

    assert first.startswith(f"release:{run_id}:")
    assert select({"summary": "v1"}) == first
    assert select({"summary": "another"}) != first
    workspace.write_release_marker({"fingerprint": "ignored"})
    assert select({"summary": "v1"}) == first
    workspace.write("app.py", "print('v2')")
    assert select({"summary": "v1"}) != first
    assert release_key(uuid4(), workspace)({"summary": "v1"}) != select(
        {"summary": "v1"}
    )


def test_approval_interrupts_become_approval_records() -> None:
    class Interrupt:
        def __init__(self, value: Any) -> None:
            self.value = value

    approvals = approvals_from_interrupts(
        [
            Interrupt(
                {
                    "type": "agloom.approval",
                    "action_id": "action-1",
                    "tool": RELEASE_TOOL,
                    "arguments": {"summary": "v1"},
                    "policy_rule": "release review",
                    "expires_at_ns": 1_700_000_000_000_000_000,
                }
            ),
            Interrupt({"type": "agloom.loop"}),
            Interrupt("not a mapping"),
        ]
    )
    assert [item.action_id for item in approvals] == ["action-1"]
    assert approvals[0].expires_at is not None
    assert approvals[0].expires_at.year == 2023


def test_reconciliation_and_unsupported_pauses_are_recognized() -> None:
    interrupts = [
        SimpleNamespace(
            value={
                "type": RECONCILIATION_INTERRUPT,
                "action_id": "action-2",
                "tool": RELEASE_TOOL,
                "arguments": {"summary": "v2"},
                "error": "OSError",
                "namespace": POLICY_NAMESPACE,
                "organization_id": TEST_PRINCIPAL.tenant_id,
            }
        ),
        SimpleNamespace(value={"type": "agloom.loop"}),
    ]

    [item] = reconciliations_from_interrupts(interrupts)

    assert (item.kind, item.status, item.error) == (
        "reconciliation",
        "unknown",
        "OSError",
    )
    assert item.arguments == {"summary": "v2"}
    assert approvals_from_interrupts(interrupts) == []
    assert unsupported_pauses(interrupts) == ["agloom.loop"]


def test_only_reviewers_of_the_tenant_may_reconcile() -> None:
    def action(**changes: Any) -> SimpleNamespace:
        values = {
            "organization_id": TEST_PRINCIPAL.tenant_id,
            "namespace": POLICY_NAMESPACE,
            "requested_by": "builder",
        }
        return SimpleNamespace(**{**values, **changes})

    reviewer_principal = Principal("reviewer", TEST_PRINCIPAL.tenant_id)
    builder = Principal("builder", TEST_PRINCIPAL.tenant_id)

    assert (
        reconciliation_refusal(action(), reviewer_principal, separate_approver=True)
        is None
    )
    assert reconciliation_refusal(action(), builder, separate_approver=False) is None
    assert "requester" in reconciliation_refusal(
        action(), builder, separate_approver=True
    )
    assert "another tenant" in reconciliation_refusal(
        action(organization_id="other"), reviewer_principal, separate_approver=False
    )
    assert "another tenant" in reconciliation_refusal(
        action(namespace="other-app"), reviewer_principal, separate_approver=False
    )


def _scope(store: MemoryStudioStore, tmp_path: Path) -> tuple[RunScope, RunWorkspace]:
    project = Project(
        id=uuid4(),
        name="Demo",
        objective="Demo objective",
        acceptance_criteria=[],
        created_at="2026-01-01T00:00:00Z",
    )
    task = Task(
        id=uuid4(),
        project_id=project.id,
        title="Task",
        objective="Task objective",
        status="running",
        created_at="2026-01-01T00:00:00Z",
    )
    run = LoopRun(
        id=uuid4(),
        task_id=task.id,
        status="running",
        proof=RunProof(),
        created_at="2026-01-01T00:00:00Z",
    )
    store.runs[run.id] = run
    store.owners[run.id] = TEST_PRINCIPAL.tenant_id
    workspace = RunWorkspace.open(tmp_path / "projects", project.id, run.id)
    return RunScope(run, task, project, TEST_PRINCIPAL), workspace


def test_checkpoint_threads_are_authorized_for_the_owning_tenant_only(
    tmp_path: Path,
) -> None:
    store = MemoryStudioStore()
    scope, _ = _scope(store, tmp_path)
    authorize = TenantRunAuthorizer(store, scope.run.id)

    assert authorize(str(scope.run.id)) is True
    assert authorize(str(uuid4())) is False
    with principal_scope(Principal("intruder", "other-tenant")):
        assert authorize(str(scope.run.id)) is False


def test_build_agent_wires_the_policy_loop_and_tools(tmp_path: Path) -> None:
    store = MemoryStudioStore()
    scope, workspace = _scope(store, tmp_path)
    policy_store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
    try:
        agent = build_agent(
            scope=scope,
            run_workspace=workspace,
            model=scripted(),
            store=store,
            publish_event=lambda *args, **kwargs: None,
            policy=default_policy(PREVIEW_URL),
            policy_store=policy_store,
            checkpointer=InMemorySaver(),
            mcp=None,
            preview_url=PREVIEW_URL,
        )
        tools = {item.name for item in agent.capabilities.descriptors()}
        policy = agent.policy
    finally:
        policy_store.close()

    assert tools == {
        "inspect_workspace",
        "read_workspace_file",
        "write_workspace_file",
        "run_allowlisted_check",
        "read_validation_feedback",
        "validate_preview",
        RELEASE_TOOL,
    }
    assert policy is not None
    assert set(policy.consequential) == {RELEASE_TOOL}
    assert policy.barrier.namespace == POLICY_NAMESPACE
    assert policy.barrier.organization_id == TEST_PRINCIPAL.tenant_id
    assert policy.barrier.requested_by == TEST_PRINCIPAL.subject
    assert agent.loop_policy == LOOP_POLICY
    assert agent.limits.max_depth == 0


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self, capability_id: str, arguments: dict[str, Any]
    ) -> ToolMessage:
        self.calls.append((capability_id, arguments))
        # The shape returned by Playwright MCP through langchain-mcp-adapters.
        return ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": '### Result\n{"url":"http://target-preview/",'
                    '"snapshot":"Product preview ready"}',
                    "id": "lc_result",
                },
                {"type": "text", "text": "### Page\n- Page Title: Studio preview"},
            ],
            tool_call_id="call",
        )


@pytest.mark.anyio
async def test_validate_preview_records_trusted_browser_evidence(
    tmp_path: Path,
) -> None:
    store = MemoryStudioStore()
    scope, workspace = _scope(store, tmp_path)
    runtime = FakeRuntime()
    policy_store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
    try:
        agent = build_agent(
            scope=scope,
            run_workspace=workspace,
            model=scripted(),
            store=store,
            publish_event=lambda *args, **kwargs: None,
            policy=default_policy(PREVIEW_URL),
            policy_store=policy_store,
            checkpointer=InMemorySaver(),
            mcp=runtime,  # type: ignore[arg-type]
            preview_url=PREVIEW_URL,
        )
        preview = agent.capabilities.descriptor("validate_preview").value
        denied = json.loads(
            await preview.ainvoke({"url": "http://localhost:8000/unsafe"})
        )
        marker = workspace.workspace.resolve(".agloom/playwright.json")
        assert not marker.exists()
        recorded = json.loads(await preview.ainvoke({"url": PREVIEW_URL + "/"}))
    finally:
        policy_store.close()

    assert denied == {
        "error": "preview validation is restricted to TARGET_PREVIEW_URL",
        "expected_url": PREVIEW_URL,
    }
    assert recorded["navigation_succeeded"] is True
    assert recorded["snapshot"].startswith("### Result")
    assert json.loads(marker.read_text())["url"] == PREVIEW_URL + "/"
    assert (
        ObjectiveValidator(workspace.workspace)
        .validate(
            [
                AcceptanceCriterion(
                    id="AC-9",
                    description="The preview renders",
                    kind="browser",
                    target=PREVIEW_URL,
                    expected="Product preview ready",
                )
            ]
        )[0]
        .passed
    )
    assert runtime.calls[0][0] == RUN_CODE_CAPABILITY
    assert "page.goto" in runtime.calls[0][1]["code"]


@pytest.mark.anyio
async def test_validate_preview_without_browser_validation(tmp_path: Path) -> None:
    store = MemoryStudioStore()
    scope, workspace = _scope(store, tmp_path)
    policy_store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
    try:
        agent = build_agent(
            scope=scope,
            run_workspace=workspace,
            model=scripted(),
            store=store,
            publish_event=lambda *args, **kwargs: None,
            policy=default_policy(PREVIEW_URL),
            policy_store=policy_store,
            checkpointer=InMemorySaver(),
            mcp=None,
            preview_url=PREVIEW_URL,
        )
        preview = agent.capabilities.descriptor("validate_preview").value
        result = json.loads(await preview.ainvoke({"url": PREVIEW_URL}))
    finally:
        policy_store.close()

    assert result == {"error": "browser validation is not configured"}


def test_release_gate_reports_failing_criteria(tmp_path: Path) -> None:
    store = MemoryStudioStore()
    scope, workspace = _scope(store, tmp_path)
    criterion = AcceptanceCriterion(
        id="AC-1", description="App exists", kind="file", target="app.py"
    )
    scope = RunScope(
        scope.run,
        scope.task,
        scope.project.model_copy(update={"acceptance_criteria": [criterion]}),
        scope.principal,
    )
    gate = ReleaseGate(ObjectiveValidator(workspace.workspace), scope)
    release = SimpleNamespace(
        tool_call={"name": RELEASE_TOOL, "args": {"summary": "x"}, "id": "call-1"}
    )

    blocked = gate.wrap_tool_call(
        release, lambda request: pytest.fail("handler must not run")
    )
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert json.loads(blocked.content)["failed_criteria"] == ["AC-1"]
    workspace.workspace.write("app.py", "print('ok')")
    passed = gate.wrap_tool_call(release, lambda request: "handled")
    assert passed == "handled"
    other = SimpleNamespace(
        tool_call={"name": "inspect_workspace", "args": {}, "id": "c"}
    )
    assert gate.wrap_tool_call(other, lambda request: "inspected") == "inspected"


def test_browser_observation_uses_the_url_returned_by_playwright() -> None:
    payload = browser_observation(
        'result: {"url":"http://target-preview","snapshot":"Product preview ready"}'
    )
    assert payload["navigation_succeeded"] is True
    assert payload["url"] == "http://target-preview"
    assert "Product preview ready" in payload["snapshot"]
    assert browser_observation("no navigation")["navigation_succeeded"] is False
