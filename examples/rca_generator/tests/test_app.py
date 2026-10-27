"""Evidence, API, policy, and pipeline tests that never contact a model provider."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from agentbarrier.runtime import PolicyEffect
from backend import app as service
from backend.app import RcaReport, create_app, make_tools, validate_report
from backend.store import IncidentStore
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from agloom import (
    AgentEvent,
    EventName,
    ExecutionError,
    Observer,
    PolicyManager,
    PolicyViolationError,
)

INCIDENT = "INC-2026-041"


def sample_report(incident_id: str = INCIDENT) -> dict[str, Any]:
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


def envelope(report: dict[str, Any]) -> AIMessage:
    return AIMessage(content=f"<xstructured>{json.dumps(report)}</xstructured>")


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


class FakeAnalyst:
    def __init__(self, store: IncidentStore, *reports: dict[str, Any]) -> None:
        self.tools = {item.name: item for item in make_tools(store)}
        self.reports = list(reports) or [sample_report()]
        self.prompts: list[str] = []
        self.error: Exception | None = None
        self.closed = False

    async def evidence(self, incident_id: str) -> dict[str, object]:
        return {
            kind: json.loads(self.tools[name].invoke({"incident_id": incident_id}))
            for kind, name in service.EVIDENCE_TOOLS.items()
        }

    async def report(self, prompt: str) -> RcaReport:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        report = self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]
        return RcaReport.model_validate(report)

    def close(self) -> None:
        self.closed = True


class RecordingObserver(Observer):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


def client(store: IncidentStore, analyst: FakeAnalyst) -> TestClient:
    return TestClient(create_app(store, analyst_factory=lambda *_: analyst))


@pytest.fixture
def real_analyst(
    store: IncidentStore, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., service.IncidentAnalyst]]:
    policies: list[PolicyManager] = []

    def build(model: Any, observers: Sequence[Observer] = ()) -> Any:
        monkeypatch.setattr("backend.app.chat_model", lambda: model)
        policy = service.tool_policy(store)
        policies.append(policy)
        return service.make_analyst(store, policy, observers)

    yield build
    for policy in policies:
        policy.store.close()


def test_seeded_store_and_scoped_tools(store: IncidentStore) -> None:
    assert len(store.list()) == 2
    incident, logs, metrics = make_tools(store)
    assert "INC-2026-041-I1" in incident.invoke({"incident_id": INCIDENT})
    assert [r["id"] for r in json.loads(logs.invoke({"incident_id": INCIDENT}))] == [
        "INC-2026-041-L1",
        "INC-2026-041-L2",
    ]
    assert "INC-2026-052-M1" in metrics.invoke({"incident_id": "INC-2026-052"})
    assert "INC-2026-052" not in logs.invoke({"incident_id": INCIDENT})
    with pytest.raises(ValueError, match="Unknown incident"):
        logs.invoke({"incident_id": "UNKNOWN"})


def test_evidence_policy_allows_seeded_incidents_only(store: IncidentStore) -> None:
    manager = service.tool_policy(store)
    try:
        runtime = manager.policy
        for name in service.EVIDENCE_TOOLS.values():
            assert runtime.evaluate(name, {"incident_id": INCIDENT}).effect is (
                PolicyEffect.ALLOW
            )
            assert runtime.evaluate(name, {"incident_id": "INC-1999-001"}).effect is (
                PolicyEffect.DENY
            )
            assert manager.visible(name)
        assert runtime.evaluate("publish_report", {}).effect is PolicyEffect.DENY
        assert not manager.visible("publish_report")
        assert manager.consequential == {}
        assert manager.on_denied == "inform_model"
    finally:
        manager.store.close()


def test_end_to_end_contract_and_grounding(store: IncidentStore) -> None:
    analyst = FakeAnalyst(store)
    with client(store, analyst) as http:
        health = http.get("/api/health")
        assert health.status_code == 200
        assert health.json()["agloom_features"] == [
            "pipeline",
            "pipeline_stages",
            "local_tools",
            "policy_governed_tools",
            "structured_output",
            "evidence_validation",
        ]
        assert health.json()["observability"]["enabled"] is False
        governance = http.get("/api/governance").json()
        assert governance["structured_schema"] == "RcaReport"
        assert governance["human_review_required"] is True
        policy = governance["tool_policy"]
        assert policy["version"] == "rca-generator-v1"
        assert policy["default_effect"] == "deny"
        assert [rule["tool"] for rule in policy["rules"]] == [
            "get_incident",
            "get_logs",
            "get_metrics",
        ]
        assert policy["rules"][0]["conditions"] == [
            {
                "path": "incident_id",
                "operator": "in",
                "value": ["INC-2026-041", "INC-2026-052"],
            }
        ]
        listed = http.get("/api/incidents").json()
        assert listed[0]["id"] == INCIDENT
        assert "records" not in listed[0]
        assert "impact" not in listed[0]
        response = http.post("/api/analyze", json={"incident_id": INCIDENT})
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
    assert "checkout_http_5xx_rate" in analyst.prompts[0]
    assert "pool exhausted" in analyst.prompts[0]
    assert len(analyst.prompts) == 1
    assert analyst.closed


def test_unknown_evidence_reference_is_repaired_once(store: IncidentStore) -> None:
    invalid = sample_report()
    invalid["root_cause"]["citations"].append("FAKE-ID")
    analyst = FakeAnalyst(store, invalid, sample_report())
    with client(store, analyst) as http:
        response = http.post("/api/analyze", json={"incident_id": INCIDENT})

    assert response.status_code == 200
    assert len(analyst.prompts) == 2
    assert "FAKE-ID" in analyst.prompts[1]
    assert "INC-2026-041-M2" in analyst.prompts[1]


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
    store: IncidentStore, mutation: Callable[[dict[str, Any]], None]
) -> None:
    report = sample_report()
    mutation(report)
    with client(store, FakeAnalyst(store, report)) as http:
        response = http.post("/api/analyze", json={"incident_id": INCIDENT})
    assert response.status_code == 502
    assert "unsupported evidence" in response.json()["detail"]


def test_agent_failures_return_gateway_error(store: IncidentStore) -> None:
    analyst = FakeAnalyst(store)
    analyst.error = ExecutionError("provider unavailable")
    with client(store, analyst) as http:
        response = http.post("/api/analyze", json={"incident_id": INCIDENT})
    assert response.status_code == 502


def test_unknown_and_invalid_request(store: IncidentStore) -> None:
    with TestClient(create_app(store)) as http:
        assert (
            http.post("/api/analyze", json={"incident_id": "missing"}).status_code
            == 404
        )
        assert http.post("/api/analyze", json={"incident_id": ""}).status_code == 422


def test_credentials_are_required_but_never_returned(
    store: IncidentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    with TestClient(create_app(store)) as http:
        assert http.get("/api/health").json()["status"] == "unconfigured"
        response = http.post("/api/analyze", json={"incident_id": INCIDENT})
    assert response.status_code == 503
    assert "EXPLABS_API_KEY" in response.json()["detail"]


def test_analyst_wiring(store: IncidentStore, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class StubAgent:
        def with_structured_output(self, schema: Any, **options: Any) -> Any:
            captured["structured"] = (schema, options)
            return object()

    def capture(**kwargs: Any) -> StubAgent:
        captured.update(kwargs)
        return StubAgent()

    monkeypatch.setenv("EXPLABS_API_KEY", "test-key-not-saved")
    monkeypatch.setenv("EXPLABS_MODEL", "test-model")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr("backend.app.create_agent", capture)
    policy = service.tool_policy(store)
    observer = RecordingObserver()
    try:
        service.make_analyst(store, policy, [observer])
    finally:
        policy.store.close()
    assert captured["model"].model_name == "test-model"
    assert (
        str(captured["model"].openai_api_base).rstrip("/")
        == "https://example.invalid/v1"
    )
    assert captured["pattern"] == "pipeline"
    assert [stage.name for stage in captured["stages"]] == [
        "hypothesize",
        "verify",
        "report",
    ]
    assert [item.name for item in captured["tools"]] == [
        "get_incident",
        "get_logs",
        "get_metrics",
    ]
    assert captured["system_prompt"] == service.SYSTEM_PROMPT
    assert captured["limits"].max_depth == 0
    assert captured["policy"] is policy
    assert list(captured["observers"]) == [observer]
    assert captured["structured"] == (RcaReport, {})


def test_real_pipeline_produces_a_validated_report(
    store: IncidentStore,
    monkeypatch: pytest.MonkeyPatch,
    scripted_model: Callable[..., Any],
) -> None:
    model = scripted_model(
        AIMessage(content="The pool change preceded connection waits."),
        AIMessage(content="The rollback and metrics support that hypothesis."),
        envelope(sample_report()),
    )
    monkeypatch.setattr("backend.app.chat_model", lambda: model)
    with TestClient(create_app(store)) as http:
        response = http.post("/api/analyze", json={"incident_id": INCIDENT})
    assert response.status_code == 200, response.text
    assert response.json()["root_cause"]["confidence"] == 0.91
    assert len(model.received) == 3
    first_prompt = str(model.received[0][-1].content)
    assert "INC-2026-041-M2" in first_prompt
    assert "<xstructured>" in first_prompt


def test_real_pipeline_denies_lookups_for_unknown_incidents(
    real_analyst: Callable[..., service.IncidentAnalyst],
    scripted_model: Callable[..., Any],
) -> None:
    model = scripted_model(
        tool_call("get_logs", {"incident_id": "INC-1999-001"}, "call-1"),
        AIMessage(content="Only INC-2026-041 evidence is available."),
        AIMessage(content="Verified against the supplied records."),
        envelope(sample_report()),
    )
    observer = RecordingObserver()
    analyst = real_analyst(model, [observer])
    try:
        report = asyncio.run(analyst.report(f"Analyze incident {INCIDENT}."))
    finally:
        analyst.close()
    assert report.incident_id == INCIDENT
    denied = [
        event for event in observer.events if event.name == EventName.TOOL_CALL_DENIED
    ]
    assert [(event.payload["tool"], event.payload["reason"]) for event in denied] == [
        ("get_logs", "PolicyViolationError")
    ]
    selected = [
        event for event in observer.events if event.name == EventName.TOPOLOGY_SELECTED
    ]
    assert [event.topology for event in selected] == ["pipeline"]
    refusal = next(
        message for message in model.received[1] if isinstance(message, ToolMessage)
    )
    assert refusal.status == "error"
    assert "was not executed" in str(refusal.content)


def test_direct_evidence_requests_are_policy_checked(
    real_analyst: Callable[..., service.IncidentAnalyst],
    scripted_model: Callable[..., Any],
) -> None:
    observer = RecordingObserver()
    analyst = real_analyst(scripted_model(), [observer])
    try:
        bundle = asyncio.run(analyst.evidence(INCIDENT))
        assert set(bundle) == {"incident", "logs", "metrics"}
        assert bundle["incident"]["incident"]["id"] == INCIDENT  # type: ignore[index]
        with pytest.raises(PolicyViolationError):
            asyncio.run(
                analyst.agent.aexecute_capability(
                    "get_logs", {"incident_id": "INC-1999-001"}
                )
            )
    finally:
        analyst.close()
    denied = [
        event for event in observer.events if event.name == EventName.TOOL_CALL_DENIED
    ]
    assert [event.payload["tool"] for event in denied] == ["get_logs"]


def test_validate_report_replaces_citation_details(store: IncidentStore) -> None:
    incident = store.get(INCIDENT)
    assert incident is not None
    report = validate_report(RcaReport.model_validate(sample_report()), incident)
    assert [citation.id for citation in report.citations] == sorted(
        citation.id for citation in report.citations
    )
    assert report.citations[0].title == incident.title


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
