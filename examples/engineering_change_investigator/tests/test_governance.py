"""The runtime policy and the consequential release-decision action."""

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
    EVIDENCE_SEARCH,
    REFERENCE_PROMPT,
    REFERENCE_RESOURCE,
    RELEASE_DECISION,
    policy_manager,
    runtime_policy,
)
from backend.mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT

from agloom import PolicyManager

TARGET = "http://target-site"
POLICY = runtime_policy(target_site_url=TARGET, approval_ttl_seconds=600)


def effect(tool: str, arguments: dict[str, Any] | None = None) -> PolicyEffect:
    return POLICY.evaluate(tool, arguments or {}).effect


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (EVIDENCE_SEARCH, {"query": "pool"}),
        (f"{FILESYSTEM}__read_text_file", {"path": "ADR-101.md"}),
        (f"{FILESYSTEM}__read_multiple_files", {"paths": ["ADR-101.md"]}),
        (f"{FILESYSTEM}__list_directory", {"path": "."}),
        (f"{FILESYSTEM}__list_allowed_directories", {}),
        (f"{FILESYSTEM}__search_files", {"path": ".", "pattern": "*.md"}),
        (f"{FILESYSTEM}__get_file_info", {"path": "ADR-101.md"}),
        (f"{FILESYSTEM}__directory_tree", {"path": "."}),
        (f"{EVERYTHING}__echo", {"message": "ok"}),
        (f"mcp:{EVERYTHING}:resource:{REFERENCE_RESOURCE}", {}),
        (f"mcp:{EVERYTHING}:prompt:{REFERENCE_PROMPT}", {}),
        (f"{PLAYWRIGHT}__browser_navigate", {"url": f"{TARGET}/"}),
        (f"{PLAYWRIGHT}__browser_snapshot", {}),
        (RELEASE_DECISION, {"decision": "no_go"}),
    ],
)
def test_read_only_evidence_and_no_go_decisions_are_allowed(
    tool: str, arguments: dict[str, Any]
) -> None:
    assert effect(tool, arguments) is PolicyEffect.ALLOW


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (f"{FILESYSTEM}__write_file", {"path": "ADR-101.md", "content": "x"}),
        (f"{FILESYSTEM}__edit_file", {"path": "ADR-101.md", "edits": []}),
        (f"{FILESYSTEM}__move_file", {"source": "a", "destination": "b"}),
        (f"{FILESYSTEM}__create_directory", {"path": "new"}),
        (f"{EVERYTHING}__get-env", {}),
        (f"{EVERYTHING}__trigger-long-running-operation", {"duration": 10}),
        (f"mcp:{EVERYTHING}:prompt:args-prompt", {"city": "x"}),
        (f"{PLAYWRIGHT}__browser_navigate", {"url": "https://attacker.example/"}),
        (f"{PLAYWRIGHT}__browser_navigate", {"url": f"{TARGET}.attacker.example/"}),
        (f"{PLAYWRIGHT}__browser_snapshot", {"filename": "../../leak.yml"}),
        (f"{PLAYWRIGHT}__browser_run_code_unsafe", {"code": "() => 1"}),
        (f"{PLAYWRIGHT}__browser_click", {"element": "Deploy"}),
        (f"{PLAYWRIGHT}__browser_type", {"text": "x"}),
        (RELEASE_DECISION, {"decision": "rollback"}),
        ("unknown_tool", {}),
    ],
)
def test_writes_secrets_and_unknown_tools_are_denied(
    tool: str, arguments: dict[str, Any]
) -> None:
    assert effect(tool, arguments) is PolicyEffect.DENY


def test_go_decisions_need_a_release_manager() -> None:
    decision = POLICY.evaluate(RELEASE_DECISION, {"decision": "go"})

    assert decision.effect is PolicyEffect.REQUIRE_APPROVAL
    assert decision.rule_name == "release manager approves go decisions"
    assert decision.approval_ttl_seconds == 600


@pytest.fixture
def manager(tmp_path: Path) -> Iterator[PolicyManager]:
    store = SQLiteRuntimeStore(tmp_path / "agentbarrier.db")
    yield policy_manager(store, POLICY)
    store.close()


def test_tools_denied_for_every_argument_are_hidden(manager: PolicyManager) -> None:
    assert manager.visible(f"{FILESYSTEM}__read_text_file")
    assert manager.visible(f"{PLAYWRIGHT}__browser_navigate")
    assert not manager.visible(f"{FILESYSTEM}__write_file")
    assert not manager.visible(f"{EVERYTHING}__get-env")
    assert not manager.visible(f"{PLAYWRIGHT}__browser_click")
    assert manager.is_consequential(RELEASE_DECISION)
    assert manager.on_denied == "inform_model"


def decision(verdict: str, investigation: str = "inv-1") -> dict[str, Any]:
    return {
        "investigation_id": investigation,
        "decision": verdict,
        "risk": "high",
        "summary": "Pool below the documented minimum.",
        "citations": ["ADR-101"],
    }


async def test_no_go_is_published_exactly_once_and_replayed(
    manager: PolicyManager,
) -> None:
    published: list[str] = []

    async def publish() -> dict[str, str]:
        published.append("inv-1")
        return {"recorded_at": "2026-10-06T00:00:00+00:00"}

    first = await manager.aexecute(RELEASE_DECISION, decision("no_go"), publish)
    second = await manager.aexecute(RELEASE_DECISION, decision("no_go"), publish)

    assert first == second
    assert published == ["inv-1"]


async def test_go_waits_for_approval_then_runs_once(manager: PolicyManager) -> None:
    published: list[str] = []

    async def publish() -> dict[str, str]:
        published.append("inv-2")
        return {"recorded_at": "2026-10-06T00:00:00+00:00"}

    with pytest.raises(ApprovalRequired) as pending:
        await manager.aexecute(RELEASE_DECISION, decision("go", "inv-2"), publish)
    assert published == []

    manager.store.decide(
        pending.value.action.action_id, Decision.APPROVE, decided_by="release-manager"
    )
    await manager.aexecute(RELEASE_DECISION, decision("go", "inv-2"), publish)
    await manager.aexecute(RELEASE_DECISION, decision("go", "inv-2"), publish)

    assert published == ["inv-2"]


async def test_rejected_go_never_runs(manager: PolicyManager) -> None:
    async def publish() -> dict[str, str]:
        raise AssertionError("a rejected decision must not be published")

    with pytest.raises(ApprovalRequired) as pending:
        await manager.aexecute(RELEASE_DECISION, decision("go", "inv-3"), publish)
    manager.store.decide(
        pending.value.action.action_id, Decision.REJECT, decided_by="release-manager"
    )

    with pytest.raises(ApprovalRejected):
        await manager.aexecute(RELEASE_DECISION, decision("go", "inv-3"), publish)


async def test_failed_publication_fails_closed_until_reconciled(
    manager: PolicyManager,
) -> None:
    attempts: list[str] = []
    receipt = {"recorded_at": "2026-10-06T00:00:00+00:00"}

    async def publish() -> dict[str, str]:
        attempts.append("inv-4")
        raise ConnectionError("release register unavailable")

    with pytest.raises(ActionOutcomeUnknown) as failed:
        await manager.aexecute(RELEASE_DECISION, decision("no_go", "inv-4"), publish)
    assert isinstance(failed.value.__cause__, ConnectionError)
    with pytest.raises(ActionOutcomeUnknown) as unknown:
        await manager.aexecute(RELEASE_DECISION, decision("no_go", "inv-4"), publish)
    assert unknown.value.action.action_id == failed.value.action.action_id
    manager.store.reconcile(
        unknown.value.action.action_id,
        RuntimeReconciliation.COMMITTED,
        resolved_by="release-operator",
        reason="The register row exists",
        result=receipt,
    )
    replayed = await manager.aexecute(
        RELEASE_DECISION, decision("no_go", "inv-4"), publish
    )

    assert replayed == receipt
    assert attempts == ["inv-4"]
