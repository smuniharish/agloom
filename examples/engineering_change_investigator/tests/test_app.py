from __future__ import annotations

import inspect

from examples.engineering_change_investigator.backend import app as application
from examples.engineering_change_investigator.backend.app import (
    EvidenceItem,
    InvestigationReport,
    Runtime,
    create_app,
    report_is_grounded,
)
from fastapi.testclient import TestClient

from agloom import create_agent


class FakeDatabase:
    def counts(self) -> dict[str, int]:
        return {
            "documents": 3,
            "feedback": 0,
            "behavior_states": 0,
            "mcp_capabilities": 0,
            "refresh_states": 0,
            "xai_records": 0,
        }


class FakeRuntime:
    database = FakeDatabase()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_health_reports_real_infrastructure_shape() -> None:
    with TestClient(create_app(FakeRuntime)) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["database"]["documents"] == 3


def test_report_rejects_malformed_or_unknown_citations() -> None:
    report = InvestigationReport(
        summary="Not ready.",
        risk="high",
        evidence=[EvidenceItem(claim="Missing approval.", citation="[release]")],
        recommendation="Do not deploy.",
        mcp_checks={},
    )
    assert not report_is_grounded(report, {"RELEASE-301"})

    report.evidence[0].citation = "[RELEASE-999]"
    assert not report_is_grounded(report, {"RELEASE-301"})

    report.evidence[0].citation = "[RELEASE-301]"
    assert report_is_grounded(report, {"RELEASE-301"})


def test_production_call_explicitly_supplies_every_create_agent_parameter(
    monkeypatch,
) -> None:
    class DummyClient:
        pass

    class DummyAgent:
        pass

    class DummyStore:
        async def write(self, item):
            pass

        async def get(self, entity_id):
            return None

        async def parents(self, entity_id, *, context):
            return ()

        async def children(self, entity_id, *, context):
            return ()

        async def lineage(self, entity_id, *, context, max_depth=100):
            return ()

        async def close(self):
            pass

        async def query(self, query):
            if False:
                yield None

    captured = {}
    instance = Runtime.__new__(Runtime)
    instance.database = FakeDatabase()
    instance.embeddings = object()
    instance.feedback_store = object()
    instance.behavior_store = object()
    instance.mcp_registry = object()
    instance.refresh_store = object()
    instance.xai_store = DummyStore()
    monkeypatch.setenv("EXPLABS_API_KEY", "test-key")
    monkeypatch.setattr(application, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        application,
        "MultiServerMCPClient",
        lambda configuration: DummyClient(),
    )
    monkeypatch.setattr(
        application,
        "application_observers",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        application,
        "PgVectorRetriever",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        application,
        "EvidenceReranker",
        lambda: object(),
    )
    monkeypatch.setattr(
        application,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or DummyAgent(),
    )

    instance.build_agent()

    assert set(captured) == set(inspect.signature(create_agent).parameters)
