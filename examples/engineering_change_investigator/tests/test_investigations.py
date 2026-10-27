"""End-to-end API behavior with the real runtime composition, offline."""

from __future__ import annotations

from typing import Any

from agentbarrier.runtime import RuntimeReconciliation, RuntimeStatus
from backend.governance import EVIDENCE_SEARCH, RELEASE_DECISION
from backend.mcp_servers import EVERYTHING, FILESYSTEM, PLAYWRIGHT
from fastapi.testclient import TestClient

from .conftest import TARGET_SITE, HarnessFactory
from .doubles import Call, investigating, report, structured

QUESTION = "Is checkout release 4.9 ready to deploy?"


def investigate(client: TestClient) -> dict[str, Any]:
    response = client.post("/api/investigate", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_runtime_state(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["model_configured"] is True
    assert body["database"] == {"documents": 3}
    assert body["mcp_servers"] == [FILESYSTEM, EVERYTHING, PLAYWRIGHT]
    assert body["observability"] == {
        "logging": True,
        "prometheus": False,
        "metrics_port": None,
        "langfuse": False,
    }


def test_startup_indexes_the_workspace_with_the_refresh_engine(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        (harness.workspace / "RUNBOOK-201.md").write_text(
            "# RUNBOOK-201\n\nRoll back within five minutes.", encoding="utf-8"
        )
        (harness.workspace / "ADR-101.md").unlink()
        refreshed = client.post("/api/refresh").json()

    assert sorted(harness.evidence.upserts[:3]) == [
        "ADR-101",
        "RELEASE-301",
        "RUNBOOK-201",
    ]
    assert refreshed["modified"] == 1
    assert refreshed["deleted"] == 1
    assert harness.evidence.deletes == ["ADR-101"]
    assert sorted(harness.evidence.documents) == ["RELEASE-301", "RUNBOOK-201"]


def test_no_go_decision_is_published_once_with_verified_mcp_evidence(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = investigate(client)
        stored = client.get(f"/api/investigations/{body['investigation_id']}").json()

    assert body["report"]["decision"] == "no_go"
    assert body["mcp_checks"] == {
        "filesystem": "verified",
        "everything": "verified",
        "resource": "verified",
        "prompt": "verified",
        "playwright": "verified",
    }
    assert body["release_decision"]["status"] == "recorded"
    assert stored == body
    assert [item["investigation_id"] for item in harness.register.published] == [
        body["investigation_id"]
    ]
    assert harness.register.published[0]["citations"] == ["ADR-101", "RELEASE-301"]
    assert harness.servers.page_url == f"{TARGET_SITE}/"
    assert {"ExecutionCompleted", "ActionExecuted"} <= set(harness.events.names)


def test_prompt_carries_untrusted_mcp_evidence_and_allowed_citations(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        investigate(client)

    assert harness.model is not None
    first = harness.model.calls[0]
    assert "<evidence>" in first.prompt
    assert "Database load test: missing" in first.prompt
    assert "[FILE] RELEASE-301.md" in first.prompt
    assert (
        "Allowed citations: ['ADR-101', 'RELEASE-301', 'RUNBOOK-201']" in first.prompt
    )
    assert "<xstructured>" in first.prompt


def test_model_is_never_offered_unsafe_or_application_tools(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        investigate(client)

    assert harness.model is not None
    offered = set(harness.model.calls[0].tools)
    assert EVIDENCE_SEARCH in offered
    assert not offered & {
        f"{FILESYSTEM}__write_file",
        f"{EVERYTHING}__get-env",
        f"{EVERYTHING}__echo",
        f"{PLAYWRIGHT}__browser_navigate",
        f"{PLAYWRIGHT}__browser_click",
        RELEASE_DECISION,
    }


def test_model_is_offered_routed_mcp_evidence_tools(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        investigate(client)

    assert harness.model is not None
    offered = set(harness.model.calls[0].tools)
    assert {f"{FILESYSTEM}__list_directory", f"{FILESYSTEM}__read_text_file"} <= offered


def test_explainability_keeps_routed_tools(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = investigate(client)

    assert body["release_decision"]["status"] == "recorded"
    assert harness.provenance._records
    assert harness.model is not None
    offered = set(harness.model.calls[0].tools)
    assert {EVIDENCE_SEARCH, f"{FILESYSTEM}__read_text_file"} <= offered


def test_go_decision_waits_for_a_release_manager_and_publishes_once(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(investigating(report("go")))
    with TestClient(harness.app()) as client:
        body = investigate(client)
        pending = body["release_decision"]
        assert pending["status"] == "awaiting_approval"
        assert pending["policy_rule"] == "release manager approves go decisions"
        assert harness.register.published == []

        approved = client.post(
            f"/api/investigations/{body['investigation_id']}/approval",
            json={"approve": True, "reviewer": "release-manager"},
        )
        repeated = client.post(
            f"/api/investigations/{body['investigation_id']}/approval",
            json={"approve": True, "reviewer": "release-manager"},
        )

    assert approved.status_code == 200, approved.text
    decision = approved.json()["release_decision"]
    assert decision["status"] == "recorded"
    assert decision["decided_by"] == "release-manager"
    assert repeated.status_code == 409
    assert len(harness.register.published) == 1
    assert harness.events.names.count("ApprovalRequired") == 1


def test_rejected_go_decision_is_never_published(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(investigating(report("go")))
    with TestClient(harness.app()) as client:
        body = investigate(client)
        rejected = client.post(
            f"/api/investigations/{body['investigation_id']}/approval",
            json={
                "approve": False,
                "reviewer": "release-manager",
                "reason": "No load test",
            },
        )

    assert rejected.status_code == 200
    assert rejected.json()["release_decision"]["status"] == "rejected"
    assert harness.register.published == []


def test_failed_publication_is_outcome_unknown_and_never_retried(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    harness.register.failure = ConnectionError("release register unavailable")
    with TestClient(harness.app()) as client:
        body = investigate(client)
        decision = body["release_decision"]
        review = client.post(
            f"/api/investigations/{body['investigation_id']}/approval",
            json={"approve": True, "reviewer": "release-manager"},
        )
        action = harness.policy_store.get_action(decision["action_id"])

    assert decision["status"] == "outcome_unknown"
    assert decision["policy_rule"] == "record no-go decisions"
    assert action.status is RuntimeStatus.UNKNOWN
    assert review.status_code == 409
    assert harness.register.attempts == 1
    failed = [event for event in harness.events if event.name == "ActionFailed"]
    assert [event.payload["action_id"] for event in failed] == [decision["action_id"]]


def test_approved_go_whose_publication_fails_is_outcome_unknown(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(investigating(report("go")))
    with TestClient(harness.app()) as client:
        body = investigate(client)
        harness.register.failure = ConnectionError("release register unavailable")
        approval = f"/api/investigations/{body['investigation_id']}/approval"
        reviewer = {"approve": True, "reviewer": "release-manager"}
        approved = client.post(approval, json=reviewer)
        repeated = client.post(approval, json=reviewer)
        stored = client.get(f"/api/investigations/{body['investigation_id']}").json()

    assert approved.status_code == 200, approved.text
    decision = approved.json()["release_decision"]
    assert decision["status"] == "outcome_unknown"
    assert decision["action_id"] == body["release_decision"]["action_id"]
    assert decision["decided_by"] == "release-manager"
    assert repeated.status_code == 409
    assert stored["release_decision"] == decision
    assert harness.register.attempts == 1
    assert harness.register.published == []


def test_decision_reconciled_as_not_committed_is_published_once(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    harness.register.failure = ConnectionError("release register unavailable")
    with TestClient(harness.app()) as client:
        body = investigate(client)
        publication = f"/api/investigations/{body['investigation_id']}/publication"
        early = client.post(publication)
        harness.policy_store.reconcile(
            body["release_decision"]["action_id"],
            RuntimeReconciliation.NOT_COMMITTED,
            resolved_by="release-operator",
            reason="No register row exists",
        )
        harness.register.failure = None
        published = client.post(publication)
        repeated = client.post(publication)

    assert early.status_code == 409
    assert "Reconcile agentbarrier action" in early.json()["detail"]
    assert published.status_code == 200, published.text
    assert published.json()["release_decision"]["status"] == "recorded"
    assert repeated.status_code == 409
    assert len(harness.register.published) == 1
    assert harness.register.attempts == 2


def test_decision_reconciled_as_committed_replays_the_receipt(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    harness.register.failure = ConnectionError("release register unavailable")
    with TestClient(harness.app()) as client:
        body = investigate(client)
        harness.policy_store.reconcile(
            body["release_decision"]["action_id"],
            RuntimeReconciliation.COMMITTED,
            resolved_by="release-operator",
            reason="The register row exists",
            result={"recorded_at": "2026-10-06T09:30:00+00:00"},
        )
        published = client.post(
            f"/api/investigations/{body['investigation_id']}/publication"
        )

    assert published.status_code == 200, published.text
    decision = published.json()["release_decision"]
    assert decision["status"] == "recorded"
    assert decision["recorded_at"].startswith("2026-10-06T09:30:00")
    assert harness.register.attempts == 1


def test_ungrounded_report_is_corrected_once_then_rejected(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(lambda call: structured(report(citations=("[ADR-999]",))))
    with TestClient(harness.app()) as client:
        response = client.post("/api/investigate", json={"question": QUESTION})

    assert response.status_code == 502
    assert "outside the retrieved documents" in response.json()["detail"]
    assert harness.model is not None
    assert len(harness.model.calls) == 2
    assert "Correct the report" in harness.model.calls[1].prompt
    assert harness.register.published == []


def test_corrected_report_is_accepted(make_harness: HarnessFactory) -> None:
    def respond(call: Call) -> Any:
        if "Correct the report" in call.prompt:
            return structured(report())
        return structured(report(citations=("ADR-101.md",)))

    harness = make_harness(respond)
    with TestClient(harness.app()) as client:
        body = investigate(client)

    assert body["release_decision"]["status"] == "recorded"


def test_invalid_structured_output_is_an_explicit_error(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(lambda call: "The release looks fine.")
    with TestClient(harness.app()) as client:
        response = client.post("/api/investigate", json={"question": QUESTION})

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The model did not return a valid investigation report"
    )


def test_feedback_is_recorded_for_the_investigation_thread(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        body = investigate(client)
        first = client.post(
            f"/api/investigations/{body['investigation_id']}/feedback",
            json={"rating": "down", "comment": "Check the load-test ticket."},
        )
        again = client.post(
            f"/api/investigations/{body['investigation_id']}/feedback",
            json={"rating": "down", "comment": "Check the load-test ticket."},
        )

    assert first.status_code == 200
    assert first.json()["feedback_id"] == again.json()["feedback_id"]
    [event] = harness.feedback._events.values()
    assert event.target.id == f"investigation-{body['investigation_id']}"
    assert event.payload == {"rating": "down", "comment": "Check the load-test ticket."}


def test_unknown_investigation_is_not_found(make_harness: HarnessFactory) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        response = client.get(
            "/api/investigations/00000000-0000-4000-8000-000000000000"
        )

    assert response.status_code == 404


def test_without_a_model_key_investigations_are_unavailable(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness(responder=None)
    with TestClient(harness.app()) as client:
        health = client.get("/api/health").json()
        response = client.post("/api/investigate", json={"question": QUESTION})

    assert health["model_configured"] is False
    assert response.status_code == 503
    assert response.json()["detail"] == "EXPLABS_API_KEY is not configured"


def test_capabilities_and_governance_describe_the_runtime(
    make_harness: HarnessFactory,
) -> None:
    harness = make_harness()
    with TestClient(harness.app()) as client:
        capabilities = client.get("/api/capabilities").json()
        governance = client.get("/api/governance").json()

    assert set(capabilities["servers"].values()) == {"healthy"}
    names = {item["name"] for item in capabilities["capabilities"]}
    assert {"read_text_file", "echo", "browser_snapshot", "simple-prompt"} <= names
    # Servers register with RefreshPolicy(on_register=True): every kind is
    # discovered at startup, before any request.
    kinds = {item["type"] for item in capabilities["capabilities"]}
    assert kinds == {"tool", "resource", "prompt"}
    assert governance["default_effect"] == "deny"
    assert governance["consequential"] == {RELEASE_DECISION: "investigation_id"}
    assert governance["rules"][0]["effect"] == "require_approval"
