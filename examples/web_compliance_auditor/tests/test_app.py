from __future__ import annotations

import inspect
from uuid import uuid4

from examples.web_compliance_auditor.backend import app as application
from examples.web_compliance_auditor.backend.app import (
    _grounded,
    create_app,
    make_agent,
)
from examples.web_compliance_auditor.backend.models import (
    AuditReport,
    Evidence,
    Finding,
)
from fastapi.testclient import TestClient

from agloom import create_agent


class Store:
    database_url = "test"

    def initialize(self):
        pass

    def search(self, query, policy_ids):
        return [
            Evidence(
                policy_id="WEB-ACCESS-001",
                title="Access",
                excerpt="Controls need labels.",
                similarity=0.9,
            )
        ]

    def save_audit(self, *args):
        return uuid4()

    def save_feedback(self, *args):
        return uuid4()


class Structured:
    async def ainvoke(self, value, **kwargs):
        class Result:
            structured = AuditReport.model_validate(
                {
                    "verdict": "needs_review",
                    "summary": "Policy requires labels.",
                    "findings": [
                        {
                            "policy_id": "WEB-ACCESS-001",
                            "requirement": "Labels",
                            "status": "needs_review",
                            "rationale": "Browser evidence was unavailable.",
                            "evidence_ids": ["WEB-ACCESS-001"],
                        }
                    ],
                }
            )

        return Result()


class Agent:
    structured_output = Structured()


class CapturedAgent:
    pass


def test_health_and_governance():
    with TestClient(create_app(Store(), lambda: Agent())) as client:
        assert client.get("/api/health").status_code == 200
        body = client.get("/api/governance").json()
        assert body["mcp_servers"] == ["playwright", "filesystem", "everything"]


def test_audit_is_persisted_and_grounded():
    with TestClient(create_app(Store(), lambda: Agent())) as client:
        response = client.post("/api/audits", json={"url": "https://example.com"})
    assert response.status_code == 200
    assert response.json()["report"]["findings"][0]["evidence_ids"] == [
        "WEB-ACCESS-001"
    ]


def test_grounding_rejects_unknown_primary_policy_id():
    report = AuditReport(
        verdict="fail",
        summary="The finding cites an unknown primary policy.",
        findings=[
            Finding(
                policy_id="UNKNOWN-001",
                requirement="Labels",
                status="fail",
                rationale="The input has no label.",
                evidence_ids=["WEB-ACCESS-001"],
            )
        ],
    )
    assert not _grounded(report, {"WEB-ACCESS-001"})


def test_agent_production_call_captures_all_current_factory_parameters(monkeypatch):
    captured = {}
    monkeypatch.setenv("EXPLABS_API_KEY", "test-key")
    monkeypatch.setattr(application, "make_mcp_client", lambda: object())
    monkeypatch.setattr(
        application,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or CapturedAgent(),
    )
    make_agent()
    assert set(captured) == set(inspect.signature(create_agent).parameters)
