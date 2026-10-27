"""The runtime policy, the consequential publication action, and reviewer decisions.

agentbarrier enforces the policy: publishing evidence requires a reviewer's
approval bound to the exact synthesis and executes exactly once per run; every
other tool is allowed or denied statelessly by the same policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from agentbarrier import Decision
from agentbarrier.runtime import (
    DecisionAuthorization,
    RuntimeAction,
    RuntimePolicy,
    RuntimeStatus,
    RuntimeStore,
)

from agloom import APPROVAL_INTERRUPT, RECONCILIATION_INTERRUPT, PolicyManager

from .models import Approval

PUBLISH_TOOL = "publish_evidence"
FEEDBACK_TOOL = "get_validation_feedback"
REFRESH_ACTION = "refresh_evidence_source"
NAMESPACE = "continuous-evidence-lab"


def load_policy(path: Path) -> RuntimePolicy:
    """Load the strict JSON runtime policy."""
    return RuntimePolicy.from_file(path)


def publication_key(tenant_id: str, run_id: UUID) -> str:
    """The business idempotency key: one publication per run."""
    return f"{tenant_id}:{run_id}"


def run_policy(
    *,
    policy: RuntimePolicy,
    store: RuntimeStore,
    tenant_id: str,
    run_id: UUID,
    requested_by: str,
) -> PolicyManager:
    """Return the policy manager governing one run's tools and index refreshes."""
    key = publication_key(tenant_id, run_id)
    return PolicyManager(
        policy=policy,
        store=store,
        consequential={PUBLISH_TOOL: lambda arguments: key},
        namespace=NAMESPACE,
        organization_id=tenant_id,
        requested_by=requested_by,
        on_denied="raise",
    )


def approval_request(interrupts: Iterable[Any]) -> Mapping[str, Any] | None:
    """Return the pending publication approval among LangGraph interrupts."""
    return _interrupt(interrupts, APPROVAL_INTERRUPT)


def reconciliation_request(interrupts: Iterable[Any]) -> Mapping[str, Any] | None:
    """Return the publication whose outcome awaits reconciliation, if any."""
    return _interrupt(interrupts, RECONCILIATION_INTERRUPT)


def _interrupt(interrupts: Iterable[Any], kind: str) -> Mapping[str, Any] | None:
    for interrupt in interrupts:
        value = getattr(interrupt, "value", None)
        if isinstance(value, Mapping) and value.get("type") == kind:
            return value
    return None


def approval_from_request(request: Mapping[str, Any]) -> Approval:
    expires = request.get("expires_at_ns")
    return Approval(
        action_id=str(request["action_id"]),
        tool=str(request["tool"]),
        policy_rule=str(request["policy_rule"]),
        expires_at=datetime.fromtimestamp(int(expires) / 1_000_000_000, UTC)
        if expires
        else None,
    )


class ApprovalDesk:
    """Reviewer decisions on pending actions, authorized inside agentbarrier."""

    def __init__(
        self, store: RuntimeStore, *, require_separate_approver: bool = False
    ) -> None:
        self.store = store
        self.require_separate_approver = require_separate_approver

    async def decide(
        self,
        action_id: str,
        *,
        approve: bool,
        reviewer: str,
        tenant_id: str,
        note: str | None = None,
    ) -> RuntimeAction:
        """Approve or reject an action of the reviewer's tenant.

        Raises:
            agentbarrier.errors.ApprovalAuthorizationError: The reviewer belongs
                to another tenant or requested the action themselves while
                separate approvers are required.
            agentbarrier.errors.ApprovalExpired: The approval window closed.
            agentbarrier.errors.InvalidActionState: The action was already decided
                differently.
        """
        authorization = DecisionAuthorization(
            actor=reviewer,
            organization_id=tenant_id,
            namespaces=frozenset({NAMESPACE}),
            decisions=frozenset({Decision.APPROVE, Decision.REJECT}),
            require_separate_approver=self.require_separate_approver,
            reviewer_subject=reviewer,
        )
        return await asyncio.to_thread(
            self.store.decide_authorized,
            action_id,
            Decision.APPROVE if approve else Decision.REJECT,
            authorization=authorization,
            reason=note,
        )

    async def withdraw(self, action_id: str, *, actor: str) -> None:
        """Reject a still-pending action because its run was cancelled."""
        action = await asyncio.to_thread(self.store.get_action, action_id)
        if action.status is RuntimeStatus.PENDING:
            await asyncio.to_thread(
                self.store.decide,
                action_id,
                Decision.REJECT,
                decided_by=actor,
                reason="the run was cancelled",
            )

    async def status(self, action_id: str) -> RuntimeStatus:
        action = await asyncio.to_thread(self.store.get_action, action_id)
        return action.status
