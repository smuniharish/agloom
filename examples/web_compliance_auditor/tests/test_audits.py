"""End-to-end API behavior with the real runtime composition, offline."""

from __future__ import annotations

import json
from typing import Any

from agentbarrier.runtime import RuntimeReconciliation, RuntimeStatus
from backend.auditor import UNGROUNDED
from backend.governance import POLICY_SEARCH, RECORD_AUDIT
from backend.mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT
from fastapi.testclient import TestClient

from .conftest import HarnessFactory
from .doubles import UNREACHABLE, Call, auditing, report, structured

TARGET = "http://target-site/"


def audit(client: TestClient, url: str = TARGET, **body: Any) -> dict[str, Any]:
    response = client.post("/api/audits", json={"url": url, **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_runtime_state(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["model_configured"] is True
    assert body["database"] == {"policies": 3}
    assert body["mcp_servers"] == [PLAYWRIGHT, FILESYSTEM, EVERYTHING]
    assert body["policy_version"] == "web-compliance-2026-10"


def test_policy_file_changes_reach_the_index(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        policies = json.loads(harness.policies_file.read_text(encoding="utf-8"))
        policies[0]["body"] = "Public pages must provide meaningful input labels."
        harness.policies_file.write_text(json.dumps(policies[:2]), encoding="utf-8")
        refreshed = client.post("/api/policies/refresh").json()

    assert sorted(harness.policies.upserts[:3]) == [
        "WEB-ACCESS-001",
        "WEB-PRIVACY-002",
        "WEB-SECURITY-003",
    ]
    assert refreshed["modified"] == 1
    assert refreshed["deleted"] == 1
    assert sorted(harness.policies.policies) == ["WEB-ACCESS-001", "WEB-PRIVACY-002"]


def test_failing_audit_is_recorded_once_with_verified_mcp_evidence(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = audit(client)
        stored = client.get(f"/api/audits/{body['audit_id']}").json()

    assert body["report"]["verdict"] == "fail"
    assert body["mcp_checks"] == {
        "filesystem": "verified",
        "everything": "verified",
        "resource": "verified",
        "prompt": "verified",
        "playwright": "verified",
    }
    assert body["record"]["status"] == "recorded"
    assert stored == body
    assert [item["audit_id"] for item in harness.register.published] == [
        body["audit_id"]
    ]
    assert harness.register.published[0]["policy_ids"] == [
        "WEB-ACCESS-001",
        "WEB-PRIVACY-002",
    ]
    assert harness.servers.opened == [TARGET]
    assert harness.audits_counted("fail") == 1
    assert {"ExecutionCompleted", "ActionExecuted"} <= set(harness.events.names)


def test_page_snapshot_is_untrusted_evidence_in_the_prompt(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        audit(client)

    assert harness.model is not None
    prompt = harness.model.calls[0].prompt
    assert "<evidence>" in prompt
    assert "Analytics cookies are enabled by default." in prompt
    assert "[WEB-PRIVACY-002] Data collection notice" in prompt
    assert "<xstructured>" in prompt


def test_model_is_offered_only_the_policy_search_tool(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        audit(client)

    assert harness.model is not None
    assert harness.model.calls[0].tools == (POLICY_SEARCH,)
    assert RECORD_AUDIT not in harness.model.calls[0].tools


def test_passing_audit_waits_for_a_compliance_officer(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(auditing(report("pass")))
    with TestClient(harness.app()) as client:
        body = audit(client)
        pending = body["record"]
        assert pending["status"] == "awaiting_approval"
        assert pending["policy_rule"] == "compliance officer certifies passing audits"
        assert harness.register.published == []

        approved = client.post(
            f"/api/audits/{body['audit_id']}/approval",
            json={"approve": True, "reviewer": "compliance-officer"},
        )
        repeated = client.post(
            f"/api/audits/{body['audit_id']}/approval",
            json={"approve": True, "reviewer": "compliance-officer"},
        )

    assert approved.status_code == 200, approved.text
    assert approved.json()["record"]["status"] == "recorded"
    assert approved.json()["record"]["decided_by"] == "compliance-officer"
    assert repeated.status_code == 409
    assert len(harness.register.published) == 1
    assert harness.events.names.count("ApprovalRequired") == 1


def test_rejected_passing_audit_is_never_recorded(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(auditing(report("pass")))
    with TestClient(harness.app()) as client:
        body = audit(client)
        rejected = client.post(
            f"/api/audits/{body['audit_id']}/approval",
            json={"approve": False, "reviewer": "compliance-officer"},
        )

    assert rejected.json()["record"]["status"] == "rejected"
    assert harness.register.published == []


def test_failed_record_is_outcome_unknown_and_never_retried(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    harness.register.failure = ConnectionError("compliance register unavailable")
    with TestClient(harness.app()) as client:
        body = audit(client)
        record = body["record"]
        review = client.post(
            f"/api/audits/{body['audit_id']}/approval",
            json={"approve": True, "reviewer": "compliance-officer"},
        )
        action = harness.policy_store.get_action(record["action_id"])

    assert record["status"] == "outcome_unknown"
    assert record["policy_rule"] == "record failing and inconclusive audits"
    assert action.status is RuntimeStatus.UNKNOWN
    assert review.status_code == 409
    assert harness.register.attempts == 1
    failed = [event for event in harness.events if event.name == "ActionFailed"]
    assert [event.payload["action_id"] for event in failed] == [record["action_id"]]


def test_certified_pass_whose_record_fails_is_outcome_unknown(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(auditing(report("pass")))
    with TestClient(harness.app()) as client:
        body = audit(client)
        harness.register.failure = ConnectionError("compliance register unavailable")
        approval = f"/api/audits/{body['audit_id']}/approval"
        officer = {"approve": True, "reviewer": "compliance-officer"}
        approved = client.post(approval, json=officer)
        repeated = client.post(approval, json=officer)
        stored = client.get(f"/api/audits/{body['audit_id']}").json()

    assert approved.status_code == 200, approved.text
    record = approved.json()["record"]
    assert record["status"] == "outcome_unknown"
    assert record["action_id"] == body["record"]["action_id"]
    assert record["decided_by"] == "compliance-officer"
    assert repeated.status_code == 409
    assert stored["record"] == record
    assert harness.register.attempts == 1
    assert harness.register.published == []


def test_record_reconciled_as_not_committed_is_written_once(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    harness.register.failure = ConnectionError("compliance register unavailable")
    with TestClient(harness.app()) as client:
        body = audit(client)
        retry = f"/api/audits/{body['audit_id']}/record"
        early = client.post(retry)
        harness.policy_store.reconcile(
            body["record"]["action_id"],
            RuntimeReconciliation.NOT_COMMITTED,
            resolved_by="compliance-operator",
            reason="No register row exists",
        )
        harness.register.failure = None
        recorded = client.post(retry)
        repeated = client.post(retry)

    assert early.status_code == 409
    assert "Reconcile agentbarrier action" in early.json()["detail"]
    assert recorded.status_code == 200, recorded.text
    assert recorded.json()["record"]["status"] == "recorded"
    assert repeated.status_code == 409
    assert len(harness.register.published) == 1
    assert harness.register.attempts == 2


def test_record_reconciled_as_committed_replays_the_receipt(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    harness.register.failure = ConnectionError("compliance register unavailable")
    with TestClient(harness.app()) as client:
        body = audit(client)
        harness.policy_store.reconcile(
            body["record"]["action_id"],
            RuntimeReconciliation.COMMITTED,
            resolved_by="compliance-operator",
            reason="The register row exists",
            result={"recorded_at": "2026-10-06T09:30:00+00:00"},
        )
        recorded = client.post(f"/api/audits/{body['audit_id']}/record")

    assert recorded.status_code == 200, recorded.text
    record = recorded.json()["record"]
    assert record["status"] == "recorded"
    assert record["recorded_at"].startswith("2026-10-06T09:30:00")
    assert harness.register.attempts == 1


def test_ungrounded_report_is_corrected_once_then_rejected(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(lambda call: structured(report(policy_ids=("UNKNOWN-1",))))
    with TestClient(harness.app()) as client:
        response = client.post("/api/audits", json={"url": TARGET})

    assert response.status_code == 502
    assert response.json()["detail"] == UNGROUNDED
    assert harness.model is not None
    assert len(harness.model.calls) == 2
    assert harness.register.published == []


def test_corrected_report_is_accepted(make_harness: HarnessFactory) -> None:
    def respond(call: Call) -> Any:
        if "Correct the report" in call.prompt:
            return structured(report())
        return structured(report(policy_ids=("WEB-ACCESS-999",)))

    harness = make_harness(respond)
    with TestClient(harness.app()) as client:
        body = audit(client)

    assert body["record"]["status"] == "recorded"


def test_invalid_structured_output_is_an_explicit_error(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(lambda call: "The site looks compliant.")
    with TestClient(harness.app()) as client:
        response = client.post("/api/audits", json={"url": TARGET})

    assert response.status_code == 502
    assert response.json()["detail"] == "The model did not return a valid audit report"


def test_unreachable_page_is_reported_for_review(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(auditing(report("needs_review")))
    with TestClient(harness.app()) as client:
        body = audit(client, url=f"https://{UNREACHABLE}/")

    assert body["mcp_checks"]["playwright"] == "unavailable"
    assert body["record"]["status"] == "recorded"
    assert harness.model is not None
    assert "Browser inspection was unavailable" in harness.model.calls[0].prompt


def test_cloud_metadata_url_is_refused_by_policy(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        response = client.post(
            "/api/audits", json={"url": "http://169.254.169.254/latest/meta-data/"}
        )

    assert response.status_code == 403
    assert harness.servers.opened == []
    assert "ToolCallDenied" in harness.events.names


def test_unknown_policy_filter_has_no_evidence(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        response = client.post(
            "/api/audits", json={"url": TARGET, "policy_ids": ["WEB-UNKNOWN-9"]}
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "No applicable policy evidence"


def test_feedback_is_recorded_once_per_rating_and_correction(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = audit(client)
        payload = {"rating": "down", "correction": "The labels are unclear."}
        first = client.post(f"/api/audits/{body['audit_id']}/feedback", json=payload)
        again = client.post(f"/api/audits/{body['audit_id']}/feedback", json=payload)

    assert first.json()["feedback_id"] == again.json()["feedback_id"]
    [event] = harness.feedback._events.values()
    assert str(event.category) == "correction"
    assert event.target.id == f"audit-{body['audit_id']}"
    assert event.payload == payload


def test_unknown_audit_is_not_found(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        response = client.get("/api/audits/00000000-0000-4000-8000-000000000000")

    assert response.status_code == 404


def test_without_a_model_key_audits_are_unavailable(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(responder=None)
    with TestClient(harness.app()) as client:
        health = client.get("/api/health").json()
        response = client.post("/api/audits", json={"url": TARGET})

    assert health["status"] == "unconfigured"
    assert response.status_code == 503


def test_capabilities_and_governance_describe_the_runtime(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        capabilities = client.get("/api/capabilities").json()
        governance = client.get("/api/governance").json()

    assert set(capabilities["servers"].values()) == {"healthy"}
    names = {item["name"] for item in capabilities["capabilities"]}
    assert {
        "browser_navigate",
        "echo",
        "list_allowed_directories",
        "simple-prompt",
    } <= names
    # Servers register with RefreshPolicy(on_register=True): every kind is
    # discovered at startup, before any request.
    kinds = {item["type"] for item in capabilities["capabilities"]}
    assert kinds == {"tool", "resource", "prompt"}
    assert governance["default_effect"] == "deny"
    assert governance["consequential"] == {RECORD_AUDIT: "audit_id"}
    assert governance["mcp_servers"]["browser"] == PLAYWRIGHT
