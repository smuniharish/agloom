"""Policy governance of every tool call and of the compliance register.

agentbarrier evaluates the rules in order, the first matching rule wins, and
anything no rule allows is denied. Evidence collection is read-only; the
browser opens only http(s) pages. Publishing an audit is a consequential action
recorded exactly once per audit, and certifying a site as compliant (`pass`)
waits for a compliance officer.
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

POLICY_VERSION = "web-compliance-2026-10"
RECORD_AUDIT = "record_audit"
POLICY_SEARCH = "search_policies"
REFERENCE_RESOURCE = "demo://resource/static/document/architecture.md"
REFERENCE_PROMPT = "simple-prompt"
METADATA_SERVICE = "http://169.254.169.254"

ALLOW = PolicyEffect.ALLOW
DENY = PolicyEffect.DENY
APPROVE = PolicyEffect.REQUIRE_APPROVAL
EQ = ConditionOperator.EQ
STARTS_WITH = ConditionOperator.STARTS_WITH


def runtime_policy(*, approval_ttl_seconds: float) -> RuntimePolicy:
    """Return the auditor's runtime policy.

    Args:
        approval_ttl_seconds: How long a pending passing audit can be approved.
    """
    navigate = f"{PLAYWRIGHT}__browser_navigate"
    return RuntimePolicy(
        version=POLICY_VERSION,
        rules=(
            PolicyRule(
                "compliance officer certifies passing audits",
                APPROVE,
                tool=RECORD_AUDIT,
                conditions=(ArgumentCondition("verdict", EQ, "pass"),),
                approval_ttl_seconds=approval_ttl_seconds,
            ),
            PolicyRule(
                "record failing and inconclusive audits",
                ALLOW,
                tool=RECORD_AUDIT,
                conditions=(
                    ArgumentCondition(
                        "verdict", ConditionOperator.IN, ["fail", "needs_review"]
                    ),
                ),
            ),
            PolicyRule(
                "browser tools never write files",
                DENY,
                tool=f"{PLAYWRIGHT}__*",
                # Optional arguments reach the policy with their default, null.
                conditions=(ArgumentCondition("filename", ConditionOperator.NE, None),),
            ),
            PolicyRule(
                "never open the cloud metadata service",
                DENY,
                tool=navigate,
                conditions=(ArgumentCondition("url", STARTS_WITH, METADATA_SERVICE),),
            ),
            PolicyRule(
                "open public http pages",
                ALLOW,
                tool=navigate,
                conditions=(ArgumentCondition("url", STARTS_WITH, "http://"),),
            ),
            PolicyRule(
                "open public https pages",
                ALLOW,
                tool=navigate,
                conditions=(ArgumentCondition("url", STARTS_WITH, "https://"),),
            ),
            PolicyRule(
                "read the open page", ALLOW, tool=f"{PLAYWRIGHT}__browser_snapshot"
            ),
            PolicyRule("search approved policies", ALLOW, tool=POLICY_SEARCH),
            PolicyRule(
                "list the policy workspace",
                ALLOW,
                tool=f"{FILESYSTEM}__list_allowed_directories",
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

    Denied model tool calls are returned to the model as errors so the audit
    can continue with permitted evidence.
    """
    return PolicyManager(
        policy=policy,
        store=store,
        consequential={RECORD_AUDIT: "audit_id"},
        namespace=APPLICATION_ID,
        organization_id=TENANT_ID,
        requested_by=APPLICATION_ID,
        on_denied="inform_model",
    )
