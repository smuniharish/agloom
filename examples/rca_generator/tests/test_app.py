"""Semantic tests for evidence, API behavior and model wiring without network calls."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.app import (
    RcaReport,
    create_app,
    evidence_bundle,
    make_agent,
    make_tools,
    validate_report,
)
from backend.store import IncidentStore
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agloom import PipelineStage, create_agent


@pytest.fixture
def store() -> IncidentStore:
    return IncidentStore()


def sample_report(incident_id: str = "INC-2026-041") -> dict:
    return {
        "incident_id": incident_id,
        "summary": "Pool exhaustion followed the deployment.",
        "root_cause": {
            "description": "Pool capacity reduction exhausted connections.",
            "confidence": 0.91,
            "citations": ["INC-2026-041-I1", "INC-2026-041-M2"],
        },
        "confidence": 0.88,
        "impact": "Checkout errors increased.",
        "contributing_factors": [
            {
                "description": "Retry exhaustion",
                "confidence": 0.8,
                "citations": ["INC-2026-041-L2"],
            }
        ],
        "evidence": [
            {
                "claim": "Pool waits and errors rose together.",
                "confidence": 0.9,
                "citation_ids": ["INC-2026-041-M1", "INC-2026-041-M2"],
            }
        ],
        "citations": [],
        "timeline": [
            {
                "timestamp": "2026-09-18T18:34:00Z",
                "title": "Rollback",
                "citations": ["INC-2026-041-I2"],
            }
        ],
        "recommendations": [
            {
                "title": "Guard pool configuration",
                "action": "Add a pre-deploy capacity check.",
                "priority": "high",
            }
        ],
    }


class StubAgent:
    def __init__(self, report: dict) -> None:
        self.report = report
        self.prompts: list[str] = []

    def invoke(self, input: str) -> SimpleNamespace:
        self.prompts.append(input)
        return SimpleNamespace(structured=self.report)


class SequenceAgent:
    def __init__(self, reports: list[dict]) -> None:
        self.reports = iter(reports)
        self.prompts: list[str] = []

    def invoke(self, input: str) -> SimpleNamespace:
        self.prompts.append(input)
        return SimpleNamespace(structured=next(self.reports))


def test_seeded_store_and_scoped_tools(store: IncidentStore) -> None:
    assert len(store.list()) == 2
    incident, logs, metrics = make_tools(store)
    assert "INC-2026-041-I1" in incident.invoke({"incident_id": "INC-2026-041"})
    assert [
        r["id"] for r in json.loads(logs.invoke({"incident_id": "INC-2026-041"}))
    ] == [
        "INC-2026-041-L1",
        "INC-2026-041-L2",
    ]
    assert "INC-2026-052-M1" in metrics.invoke({"incident_id": "INC-2026-052"})
    assert "INC-2026-052" not in logs.invoke({"incident_id": "INC-2026-041"})
    assert "INC-2026-041-M2" in json.dumps(
        evidence_bundle(store, store.get("INC-2026-041"))
    )
    with pytest.raises(ValueError, match="Unknown incident"):
        logs.invoke({"incident_id": "UNKNOWN"})


def test_end_to_end_contract_and_grounding(store: IncidentStore) -> None:
    fake = StubAgent(sample_report())
    http = TestClient(create_app(store, agent_factory=lambda _: fake))
    health = http.get("/api/health")
    assert health.status_code == 200
    assert health.json()["agloom_features"] == [
        "pipeline",
        "pipeline_stages",
        "local_tools",
        "capability_routing",
        "structured_output",
        "evidence_validation",
    ]
    assert health.json()["observability"]["enabled"] is False
    governance = http.get("/api/governance").json()
    assert governance["structured_schema"] == "RcaReport"
    assert governance["human_review_required"] is True
    assert http.get("/api/incidents").json()[0]["id"] == "INC-2026-041"
    response = http.post("/api/analyze", json={"incident_id": "INC-2026-041"})
    assert response.status_code == 200
    report = response.json()
    assert report["root_cause"]["confidence"] == 0.91
    assert {c["id"] for c in report["citations"]} == {
        "INC-2026-041-I1",
        "INC-2026-041-M2",
        "INC-2026-041-L2",
        "INC-2026-041-M1",
        "INC-2026-041-I2",
    }
    assert all(c["excerpt"] for c in report["citations"])
    assert "checkout_http_5xx_rate" in fake.prompts[0]
    assert "pool exhausted" in fake.prompts[0]
    assert len(fake.prompts) == 1


def test_unknown_evidence_reference_is_repaired_once(store: IncidentStore) -> None:
    invalid = sample_report()
    invalid["root_cause"]["citations"].append("FAKE-ID")
    agent = SequenceAgent([invalid, sample_report()])
    http = TestClient(create_app(store, agent_factory=lambda _: agent))

    response = http.post("/api/analyze", json={"incident_id": "INC-2026-041"})

    assert response.status_code == 200
    assert len(agent.prompts) == 2
    assert "FAKE-ID" in agent.prompts[1]
    assert "INC-2026-041-M2" in agent.prompts[1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["root_cause"]["citations"].append("FAKE-ID"),
        lambda r: r.update(incident_id="INC-2026-052"),
        lambda r: r["evidence"][0].update(confidence=1.1),
        lambda r: r["citations"].append(
            {
                "id": "FAKE-ID",
                "source": "log",
                "title": "x",
                "timestamp": "x",
                "excerpt": "x",
            }
        ),
    ],
)
def test_unsupported_output_returns_gateway_error(
    store: IncidentStore, mutation
) -> None:
    report = sample_report()
    mutation(report)
    http = TestClient(create_app(store, agent_factory=lambda _: StubAgent(report)))
    response = http.post("/api/analyze", json={"incident_id": "INC-2026-041"})
    assert response.status_code == 502
    assert "unsupported evidence" in response.json()["detail"]


def test_unknown_and_invalid_request(store: IncidentStore) -> None:
    http = TestClient(create_app(store))
    assert http.post("/api/analyze", json={"incident_id": "missing"}).status_code == 404
    assert http.post("/api/analyze", json={"incident_id": ""}).status_code == 422


def test_credentials_are_required_but_never_returned(
    store: IncidentStore, monkeypatch
) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    http = TestClient(create_app(store))
    assert http.get("/api/health").json()["status"] == "unconfigured"
    response = http.post("/api/analyze", json={"incident_id": "INC-2026-041"})
    assert response.status_code == 503
    assert "EXPLABS_API_KEY" in response.json()["detail"]


def test_real_model_configuration_and_pipeline(
    store: IncidentStore, monkeypatch
) -> None:
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            structured_output=StubAgent(sample_report()),
            close_observability=lambda: None,
        )

    monkeypatch.setenv("EXPLABS_API_KEY", "test-key-not-saved")
    monkeypatch.setenv("EXPLABS_MODEL", "test-model")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr("backend.app.create_agent", capture)
    make_agent(store)
    assert captured["model"].model_name == "test-model"
    assert (
        str(captured["model"].openai_api_base).rstrip("/")
        == "https://example.invalid/v1"
    )
    assert captured["pattern"] == "pipeline"
    assert len(captured["stages"]) == 3
    assert captured["structured_output_schema"] is RcaReport
    assert len(captured["tools"]) == 3
    assert captured["xai_options"]["application_id"] == "rca-generator"
    assert captured["observers"] == ()
    assert set(captured) == set(inspect.signature(create_agent).parameters)


def test_agloom_pipeline_structured_output_executes(store: IncidentStore) -> None:
    incident = store.get("INC-2026-041")
    assert incident is not None
    payload = json.dumps(sample_report())
    model = FakeListChatModel(
        responses=[
            "The pool change preceded connection waits.",
            "The rollback and metrics support that hypothesis.",
            f"<xstructured>{payload}</xstructured>",
        ]
    )
    agent = create_agent(
        model=model,
        tools=make_tools(store),
        pattern="pipeline",
        stages=[
            PipelineStage(name="hypothesize", instruction="Hypothesize."),
            PipelineStage(name="verify", instruction="Verify."),
            PipelineStage(name="report", instruction="Return schema envelope."),
        ],
        structured_output_schema=RcaReport,
        xai_enabled=False,
    )
    result = agent.structured_output.invoke(
        f"Analyze {incident.id}: {json.dumps(evidence_bundle(store, incident))}"
    )
    parsed = RcaReport.model_validate(result.structured)
    assert validate_report(parsed, incident).root_cause.confidence == 0.91


def test_duplicate_data_rejected(tmp_path: Path) -> None:
    path = tmp_path / "incidents.json"
    rows = json.loads(
        (Path(__file__).parents[1] / "backend" / "data" / "incidents.json").read_text(
            encoding="utf-8"
        )
    )
    rows.append(rows[0])
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate incident IDs"):
        IncidentStore(path)
