"""Policy governance of every tool call and of the published release decision.

agentbarrier evaluates the rules in order, the first matching rule wins, and
anything no rule allows is denied. Evidence surfaces are read-only; publishing
a release decision is a consequential action recorded exactly once per
investigation, and a `go` decision waits for a release manager's approval.
"""

from __future__ import annotations

from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    RuntimeStore,
)

from agloom import PolicyManager

from .mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT
from .settings import APPLICATION_ID, TENANT_ID

POLICY_VERSION = "engineering-change-2026-10"
RELEASE_DECISION = "record_release_decision"
EVIDENCE_SEARCH = "semantic_change_search"
REFERENCE_RESOURCE = "demo://resource/static/document/architecture.md"
REFERENCE_PROMPT = "simple-prompt"

ALLOW = PolicyEffect.ALLOW
DENY = PolicyEffect.DENY
APPROVE = PolicyEffect.REQUIRE_APPROVAL
EQ = ConditionOperator.EQ


def runtime_policy(
    *, target_site_url: str, approval_ttl_seconds: float
) -> RuntimePolicy:
    """Return the investigator's runtime policy.

    Args:
        target_site_url: The only site the browser may open.
        approval_ttl_seconds: How long a pending go decision can be approved.
    """
    return RuntimePolicy(
        version=POLICY_VERSION,
        rules=(
            PolicyRule(
                "release manager approves go decisions",
                APPROVE,
                tool=RELEASE_DECISION,
                conditions=(ArgumentCondition("decision", EQ, "go"),),
                approval_ttl_seconds=approval_ttl_seconds,
            ),
            PolicyRule(
                "record no-go decisions",
                ALLOW,
                tool=RELEASE_DECISION,
                conditions=(ArgumentCondition("decision", EQ, "no_go"),),
            ),
            PolicyRule(
                "browser tools never write files",
                DENY,
                tool=f"{PLAYWRIGHT}__*",
                # Optional arguments reach the policy with their default, null.
                conditions=(ArgumentCondition("filename", ConditionOperator.NE, None),),
            ),
            PolicyRule(
                "open only the release status site",
                ALLOW,
                tool=f"{PLAYWRIGHT}__browser_navigate",
                conditions=(
                    ArgumentCondition(
                        "url", ConditionOperator.STARTS_WITH, f"{target_site_url}/"
                    ),
                ),
            ),
            PolicyRule(
                "read the open page", ALLOW, tool=f"{PLAYWRIGHT}__browser_snapshot"
            ),
            PolicyRule("search indexed evidence", ALLOW, tool=EVIDENCE_SEARCH),
            PolicyRule(
                "read workspace files", ALLOW, tool=f"{FILESYSTEM}__read_text_file"
            ),
            PolicyRule(
                "read several workspace files",
                ALLOW,
                tool=f"{FILESYSTEM}__read_multiple_files",
            ),
            PolicyRule("list workspace files", ALLOW, tool=f"{FILESYSTEM}__list_*"),
            PolicyRule(
                "search workspace files", ALLOW, tool=f"{FILESYSTEM}__search_files"
            ),
            PolicyRule(
                "describe workspace files", ALLOW, tool=f"{FILESYSTEM}__get_file_info"
            ),
            PolicyRule(
                "show the workspace tree", ALLOW, tool=f"{FILESYSTEM}__directory_tree"
            ),
            PolicyRule("check MCP connectivity", ALLOW, tool=f"{EVERYTHING}__echo"),
            PolicyRule(
                "read the MCP reference document",
                ALLOW,
                tool=f"mcp:{EVERYTHING}:resource:{REFERENCE_RESOURCE}",
            ),
            PolicyRule(
                "render the MCP reference prompt",
                ALLOW,
                tool=f"mcp:{EVERYTHING}:prompt:{REFERENCE_PROMPT}",
            ),
        ),
    )


def policy_manager(store: RuntimeStore, policy: RuntimePolicy) -> PolicyManager:
    """Bind the policy to the durable agentbarrier store.

    Denied model tool calls are returned to the model as errors so the
    investigation can continue with permitted evidence.
    """
    return PolicyManager(
        policy=policy,
        store=store,
        consequential={RELEASE_DECISION: "investigation_id"},
        namespace=APPLICATION_ID,
        organization_id=TENANT_ID,
        requested_by=APPLICATION_ID,
        on_denied="inform_model",
    )
