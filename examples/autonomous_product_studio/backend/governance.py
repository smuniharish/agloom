"""Governance of the studio's tool calls with agentbarrier.

The release (`publish_release`) is the studio's consequential action: it
replaces the project's active workspace and indexes the published files. It is
durably mediated: a human approves each release, and an approved release runs
exactly once even when the worker crashes and the run resumes. Every other
tool is checked statelessly against the same policy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from agentbarrier import Decision
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    DecisionAuthorization,
    PolicyEffect,
    PolicyRule,
    RuntimeAction,
    RuntimePolicy,
    RuntimeStore,
)
from agloom import APPROVAL_INTERRUPT, RECONCILIATION_INTERRUPT, PolicyManager

from .auth import Principal
from .models import Approval
from .safety import SafeWorkspace

POLICY_NAMESPACE = "autonomous-product-studio"
POLICY_VERSION = "studio-v1"
RELEASE_TOOL = "publish_release"
CHECK_NAMES = ("frontend-build", "frontend-test", "backend-test")


def default_policy(preview_url: str) -> RuntimePolicy:
    """The studio policy; anything not allowed here is denied."""
    return RuntimePolicy(
        version=POLICY_VERSION,
        rules=(
            PolicyRule(
                "release review", PolicyEffect.REQUIRE_APPROVAL, tool=RELEASE_TOOL
            ),
            PolicyRule(
                "internal evidence is read-only",
                PolicyEffect.DENY,
                tool="write_workspace_file",
                conditions=(
                    ArgumentCondition("path", ConditionOperator.CONTAINS, ".agloom"),
                ),
            ),
            PolicyRule(
                "workspace edits", PolicyEffect.ALLOW, tool="write_workspace_file"
            ),
            PolicyRule(
                "workspace inventory", PolicyEffect.ALLOW, tool="inspect_workspace"
            ),
            PolicyRule(
                "workspace reads", PolicyEffect.ALLOW, tool="read_workspace_file"
            ),
            PolicyRule(
                "validation feedback",
                PolicyEffect.ALLOW,
                tool="read_validation_feedback",
            ),
            PolicyRule(
                "allowlisted checks",
                PolicyEffect.ALLOW,
                tool="run_allowlisted_check",
                conditions=(
                    ArgumentCondition("name", ConditionOperator.IN, list(CHECK_NAMES)),
                ),
            ),
            PolicyRule("preview evidence", PolicyEffect.ALLOW, tool="validate_preview"),
            PolicyRule(
                "preview navigation",
                PolicyEffect.ALLOW,
                tool="playwright__browser_navigate",
                conditions=(
                    ArgumentCondition("url", ConditionOperator.EQ, preview_url),
                ),
            ),
            PolicyRule(
                "preview page navigation",
                PolicyEffect.ALLOW,
                tool="playwright__browser_navigate",
                conditions=(
                    ArgumentCondition(
                        "url", ConditionOperator.STARTS_WITH, f"{preview_url}/"
                    ),
                ),
            ),
            PolicyRule(
                "page snapshots",
                PolicyEffect.ALLOW,
                tool="playwright__browser_snapshot",
            ),
            PolicyRule(
                "no other browser control", PolicyEffect.DENY, tool="playwright__*"
            ),
        ),
    )


def load_policy(policy_file: Path | None, preview_url: str) -> RuntimePolicy:
    """The operator's strict JSON policy file, or the default studio policy."""
    if policy_file is not None:
        return RuntimePolicy.from_file(policy_file)
    return default_policy(preview_url)


def release_key(
    run_id: UUID, workspace: SafeWorkspace
) -> Callable[[Mapping[str, Any]], str]:
    """Business idempotency key of a release: the run, its files, and its summary.

    Re-running the same release (for example after a crash) replays the stored
    outcome; changed files or a different summary are a new release that needs
    its own approval.
    """

    def select(arguments: Mapping[str, Any]) -> str:
        request = hashlib.sha256(
            json.dumps(dict(arguments), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"release:{run_id}:{workspace.fingerprint()[:32]}:{request[:16]}"

    return select


def policy_manager(
    *,
    policy: RuntimePolicy,
    store: RuntimeStore,
    principal: Principal,
    run_id: UUID,
    workspace: SafeWorkspace,
) -> PolicyManager:
    """The policy boundary of one run, bound to the run's tenant and requester."""
    return PolicyManager(
        policy=policy,
        store=store,
        consequential={RELEASE_TOOL: release_key(run_id, workspace)},
        namespace=POLICY_NAMESPACE,
        organization_id=principal.tenant_id,
        requested_by=principal.subject,
        on_denied="inform_model",
    )


def approvals_from_interrupts(interrupts: Iterable[Any]) -> list[Approval]:
    approvals: list[Approval] = []
    for value in _payloads(interrupts, APPROVAL_INTERRUPT):
        expires = value.get("expires_at_ns")
        approvals.append(
            Approval(
                action_id=str(value["action_id"]),
                tool=str(value["tool"]),
                arguments=dict(value.get("arguments") or {}),
                policy_rule=str(value.get("policy_rule") or ""),
                expires_at=(
                    datetime.fromtimestamp(expires / 1_000_000_000, UTC)
                    if isinstance(expires, int)
                    else None
                ),
            )
        )
    return approvals


def reconciliations_from_interrupts(interrupts: Iterable[Any]) -> list[Approval]:
    """Calls whose outcome is unknown because the tool raised after starting."""
    return [
        Approval(
            action_id=str(value["action_id"]),
            tool=str(value["tool"]),
            kind="reconciliation",
            arguments=dict(value.get("arguments") or {}),
            error=str(value.get("error") or "") or None,
            status="unknown",
        )
        for value in _payloads(interrupts, RECONCILIATION_INTERRUPT)
    ]


def unsupported_pauses(interrupts: Iterable[Any]) -> list[str]:
    """Interrupt types the studio does not configure (loop or breakpoint pauses)."""
    known = {APPROVAL_INTERRUPT, RECONCILIATION_INTERRUPT}
    kinds: list[str] = []
    for interrupt in interrupts:
        value = getattr(interrupt, "value", None)
        kind = value.get("type") if isinstance(value, Mapping) else None
        if kind not in known:
            kinds.append(str(kind))
    return kinds


def _payloads(interrupts: Iterable[Any], kind: str) -> Iterator[Mapping[str, Any]]:
    for interrupt in interrupts:
        value = getattr(interrupt, "value", None)
        if isinstance(value, Mapping) and value.get("type") == kind:
            yield value


def with_decision(approval: Approval, action: RuntimeAction) -> Approval:
    """The recorded approval with its current agentbarrier status."""
    return approval.model_copy(
        update={
            "status": action.status.value,
            "decided_by": action.decided_by if action.decided_by != "policy" else None,
            "decision_reason": action.decision_reason,
        }
    )


def reviewer(principal: Principal, *, separate_approver: bool) -> DecisionAuthorization:
    """Reviewers decide only their own tenant's studio actions."""
    return DecisionAuthorization(
        actor=principal.subject,
        organization_id=principal.tenant_id,
        namespaces=frozenset({POLICY_NAMESPACE}),
        decisions=frozenset({Decision.APPROVE, Decision.REJECT}),
        require_separate_approver=separate_approver,
        reviewer_subject=principal.subject,
    )


def reconciliation_refusal(
    action: RuntimeAction, principal: Principal, *, separate_approver: bool
) -> str | None:
    """Why `principal` may not reconcile `action`, or `None` when they may.

    agentbarrier records reconciliations without authorizing them, so the studio
    applies the reviewer rules: same tenant and namespace, and, when separate
    approvers are required, not the person who requested the action.
    """
    if (
        action.organization_id != principal.tenant_id
        or action.namespace != POLICY_NAMESPACE
    ):
        return "the action belongs to another tenant or application"
    if separate_approver and action.requested_by == principal.subject:
        return "the requester cannot reconcile their own action"
    return None
