"""The runtime policy and the consequential compliance record."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from agentbarrier import Decision
from agentbarrier.errors import ActionOutcomeUnknown, ApprovalRejected, ApprovalRequired
from agentbarrier.runtime import (
    PolicyEffect,
    RuntimeReconciliation,
    SQLiteRuntimeStore,
)
from backend.governance import (
    POLICY_SEARCH,
    RECORD_AUDIT,
    REFERENCE_PROMPT,
    REFERENCE_RESOURCE,
    policy_manager,
    runtime_policy,
)
from backend.mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT

from agloom import PolicyManager

POLICY = runtime_policy(approval_ttl_seconds=600)


def effect(tool: str, arguments: dict[str, Any] | None = None) -> PolicyEffect:
    return POLICY.evaluate(tool, arguments or {}).effect


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (POLICY_SEARCH, {"query": "labels"}),
        (f"{PLAYWRIGHT}__browser_navigate", {"url": "https://example.com/"}),
        (f"{PLAYWRIGHT}__browser_navigate", {"url": "http://target-site/"}),
        (f"{PLAYWRIGHT}__browser_snapshot", {}),
        (f"{FILESYSTEM}__list_allowed_directories", {}),
        (f"{EVERYTHING}__echo", {"message": "ok"}),
        (f"mcp:{EVERYTHING}:resource:{REFERENCE_RESOURCE}", {}),
        (f"mcp:{EVERYTHING}:prompt:{REFERENCE_PROMPT}", {}),
        (RECORD_AUDIT, {"verdict": "fail"}),
        (RECORD_AUDIT, {"verdict": "needs_review"}),
    ],
)
def test_read_only_evidence_and_conservative_verdicts_are_allowed(
    tool: str, arguments: dict[str, Any]
) -> None:
    assert effect(tool, arguments) is PolicyEffect.ALLOW


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (f"{PLAYWRIGHT}__browser_navigate", {"url": "file:///etc/passwd"}),
        (f"{PLAYWRIGHT}__browser_navigate", {"url": "javascript:alert(1)"}),
        (
            f"{PLAYWRIGHT}__browser_navigate",
            {"url": "http://169.254.169.254/latest/meta-data/"},
        ),
        (f"{PLAYWRIGHT}__browser_snapshot", {"filename": "../../leak.yml"}),
        (f"{PLAYWRIGHT}__browser_run_code_unsafe", {"code": "() => document.cookie"}),
        (f"{PLAYWRIGHT}__browser_evaluate", {"function": "() => 1"}),
        (f"{PLAYWRIGHT}__browser_click", {"element": "Continue"}),
        (f"{PLAYWRIGHT}__browser_fill_form", {"fields": []}),
        (f"{FILESYSTEM}__write_file", {"path": "policies.json", "content": "[]"}),
        (f"{FILESYSTEM}__read_text_file", {"path": "README.txt"}),
        (f"{EVERYTHING}__get-env", {}),
        (RECORD_AUDIT, {"verdict": "compliant"}),
        ("unknown_tool", {}),
    ],
)
def test_page_actions_writes_secrets_and_unknown_tools_are_denied(
    tool: str, arguments: dict[str, Any]
) -> None:
    assert effect(tool, arguments) is PolicyEffect.DENY


def test_passing_verdicts_need_a_compliance_officer() -> None:
    decision = POLICY.evaluate(RECORD_AUDIT, {"verdict": "pass"})

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL
    assert decision.rule_name == "compliance officer certifies passing audits"
    assert decision.approval_ttl_seconds == 600


@pytest.fixture
def manager(tmp_path: Path) -> Iterator[PolicyManager]:
    store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
    yield policy_manager(store, POLICY)
    store.close()


def test_tools_denied_for_every_argument_are_hidden(manager: PolicyManager) -> None:
    assert manager.visible(POLICY_SEARCH)
    assert manager.visible(f"{PLAYWRIGHT}__browser_navigate")
    assert not manager.visible(f"{PLAYWRIGHT}__browser_run_code_unsafe")
    assert not manager.visible(f"{FILESYSTEM}__write_file")
    assert not manager.visible(f"{EVERYTHING}__get-env")
    assert manager.is_consequential(RECORD_AUDIT)


def record(verdict: str, audit_id: str = "audit-1") -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "url": "http://target-site/",
        "verdict": verdict,
        "summary": "Inputs lack labels.",
        "policy_ids": ["WEB-ACCESS-001"],
    }


async def test_failing_audit_is_recorded_exactly_once(manager: PolicyManager) -> None:
    published: list[str] = []

    async def publish() -> dict[str, str]:
        published.append("audit-1")
        return {"recorded_at": "2026-10-06T00:00:00+00:00"}

    first = await manager.aexecute(RECORD_AUDIT, record("fail"), publish)
    second = await manager.aexecute(RECORD_AUDIT, record("fail"), publish)

    assert first == second
    assert published == ["audit-1"]


async def test_pass_waits_for_approval_then_runs_once(manager: PolicyManager) -> None:
    published: list[str] = []

    async def publish() -> dict[str, str]:
        published.append("audit-2")
        return {"recorded_at": "2026-10-06T00:00:00+00:00"}

    with pytest.raises(ApprovalRequired) as pending:
        await manager.aexecute(RECORD_AUDIT, record("pass", "audit-2"), publish)
    manager.store.decide(
        pending.value.action.action_id, Decision.APPROVE, decided_by="officer"
    )
    await manager.aexecute(RECORD_AUDIT, record("pass", "audit-2"), publish)
    await manager.aexecute(RECORD_AUDIT, record("pass", "audit-2"), publish)

    assert published == ["audit-2"]


async def test_rejected_pass_never_runs(manager: PolicyManager) -> None:
    async def publish() -> dict[str, str]:
        raise AssertionError("a rejected certification must not be recorded")

    with pytest.raises(ApprovalRequired) as pending:
        await manager.aexecute(RECORD_AUDIT, record("pass", "audit-3"), publish)
    manager.store.decide(
        pending.value.action.action_id, Decision.REJECT, decided_by="officer"
    )

    with pytest.raises(ApprovalRejected):
        await manager.aexecute(RECORD_AUDIT, record("pass", "audit-3"), publish)


async def test_failed_record_fails_closed_until_reconciled(
    manager: PolicyManager,
) -> None:
    attempts: list[str] = []
    receipt = {"recorded_at": "2026-10-06T00:00:00+00:00"}

    async def publish() -> dict[str, str]:
        attempts.append("audit-4")
        raise ConnectionError("compliance register unavailable")

    with pytest.raises(ActionOutcomeUnknown) as failed:
        await manager.aexecute(RECORD_AUDIT, record("fail", "audit-4"), publish)
    assert isinstance(failed.value.__cause__, ConnectionError)
    with pytest.raises(ActionOutcomeUnknown) as unknown:
        await manager.aexecute(RECORD_AUDIT, record("fail", "audit-4"), publish)
    assert unknown.value.action.action_id == failed.value.action.action_id
    manager.store.reconcile(
        unknown.value.action.action_id,
        RuntimeReconciliation.COMMITTED,
        resolved_by="compliance-operator",
        reason="The register row exists",
        result=receipt,
    )
    replayed = await manager.aexecute(RECORD_AUDIT, record("fail", "audit-4"), publish)

    assert replayed == receipt
    assert attempts == ["audit-4"]
