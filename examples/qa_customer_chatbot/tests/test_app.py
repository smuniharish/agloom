from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from backend import app as service
from backend.store import Store
from behaviorweave import BehaviorEngine
from fastapi.testclient import TestClient
from feedback_manager import FeedbackManager
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agloom import create_agent


class Capabilities:
    def __init__(self) -> None:
        self.feedback = FeedbackManager()
        self.behavior = BehaviorEngine()

    def resolve(self, name: str) -> object:
        return self.feedback if name == "feedback_manager" else self.behavior


class FakeAgent:
    def __init__(self) -> None:
        self.capabilities = Capabilities()
        self.calls: list[list[HumanMessage | AIMessage]] = []
        self.draft = service.Draft(
            answer="Unused items can be returned within 30 days. [POL-RETURNS]",
            policy_ids=["POL-RETURNS"],
        )

    def invoke(self, messages: list[HumanMessage | AIMessage]) -> service.Draft:
        self.calls.append(messages)
        return self.draft


@pytest.fixture
def setup() -> tuple[TestClient, FakeAgent]:
    agent = FakeAgent()
    return TestClient(service.create_app(agent_factory=lambda _: agent)), agent


def test_seeded_lookup_and_duplicate_ids(tmp_path: Path) -> None:
    store = Store()
    assert store.order("CUST-1024", "ORD-5001").status == "in transit"
    assert store.order("CUST-2048", "ORD-5001") is None
    assert store.search("return policy")[0].id == "POL-RETURNS"
    assert store.search("xylophone quasar nebula") == []
    (tmp_path / "customers.json").write_text(
        json.dumps(
            [
                {"id": "C", "name": "one", "orders": []},
                {"id": "C", "name": "two", "orders": []},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "policies.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        Store(tmp_path)


def test_evidence_prompt_defines_exact_citation_contract() -> None:
    policy = Store().policies["POL-RETURNS"]
    prompt = service.evidence_prompt("What is the return policy?", [policy], None)
    assert "Allowed policy IDs: POL-RETURNS." in prompt
    assert "same IDs in policy_ids" in prompt
    assert "Do not put an ID in only one field" in prompt


def test_chat_order_grounding_and_conversation(
    setup: tuple[TestClient, FakeAgent],
) -> None:
    http, agent = setup
    first = http.post(
        "/api/chat",
        json={"message": "What is the return policy?", "customer_id": "CUST-1024"},
    )
    assert first.status_code == 200
    data = first.json()
    assert data["sources"][0]["id"] == "POL-RETURNS"
    assert data["order"] is None
    assert data["guardrail"]["policy"] == service.GUARDRAIL
    assert "POL-RETURNS" in str(agent.calls[0][-1].content)
    agent.draft = service.Draft(answer="Your order is in transit.", policy_ids=[])
    second = http.post(
        "/api/chat",
        json={
            "message": "Where is ORD-5001?",
            "conversation_id": data["conversation_id"],
            "customer_id": "CUST-1024",
        },
    )
    assert second.status_code == 200
    assert second.json()["order"]["status"] == "in transit"
    assert second.json()["order"]["id"] == "ORD-5001"
    assert len(agent.calls[1]) == 3
    assert agent.calls[1][1].content == data["answer"]
    assert "in transit" in str(agent.calls[1][-1].content)


def test_no_order_leakage_and_no_evidence(setup: tuple[TestClient, FakeAgent]) -> None:
    http, agent = setup
    missing = http.post("/api/chat", json={"message": "Where is my order?"})
    assert missing.status_code == 200
    assert missing.json()["order"] is None
    assert "customer ID" in missing.json()["answer"]
    wrong = http.post(
        "/api/chat", json={"message": "Where is ORD-5001?", "customer_id": "CUST-2048"}
    )
    assert wrong.status_code == 200
    assert wrong.json()["order"] is None
    assert "couldn't find" in wrong.json()["answer"]
    empty = http.post("/api/chat", json={"message": "xylophone quasar nebula"})
    assert empty.status_code == 200
    assert empty.json()["sources"] == []
    assert agent.calls == []


def test_invalid_input_and_conversation_ownership(
    setup: tuple[TestClient, FakeAgent],
) -> None:
    http, _ = setup
    assert http.post("/api/chat", json={"message": " "}).status_code == 422
    assert (
        http.post(
            "/api/chat", json={"message": "return", "customer_id": "missing"}
        ).status_code
        == 404
    )
    assert (
        http.post(
            "/api/chat", json={"message": "return", "conversation_id": str(uuid4())}
        ).status_code
        == 404
    )
    initial = http.post(
        "/api/chat", json={"message": "return policy", "customer_id": "CUST-1024"}
    ).json()
    assert (
        http.post(
            "/api/chat",
            json={
                "message": "return policy",
                "conversation_id": initial["conversation_id"],
                "customer_id": "CUST-2048",
            },
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "draft",
    [
        service.Draft(answer="See [FAKE]", policy_ids=["FAKE"]),
        service.Draft(answer="  ", policy_ids=[]),
        service.Draft(answer="Unused items can be returned.", policy_ids=[]),
    ],
)
def test_invalid_model_evidence_is_rejected(
    setup: tuple[TestClient, FakeAgent], draft: service.Draft
) -> None:
    http, agent = setup
    agent.draft = draft
    result = http.post("/api/chat", json={"message": "return policy"})
    assert result.status_code == 502
    assert "invalid evidence" in result.json()["detail"]


def test_invalid_model_evidence_is_repaired_once(
    setup: tuple[TestClient, FakeAgent],
) -> None:
    http, agent = setup
    valid = agent.draft
    invalid = service.Draft(
        answer="Unused items can be returned within 30 days.",
        policy_ids=["POL-RETURNS"],
    )
    drafts = iter((invalid, valid))
    agent.invoke = lambda messages: next(drafts)  # type: ignore[method-assign]

    result = http.post("/api/chat", json={"message": "return policy"})

    assert result.status_code == 200
    assert result.json()["answer"] == valid.answer
    assert result.json()["sources"][0]["id"] == "POL-RETURNS"


def test_order_answer_without_verified_status_is_repaired(
    setup: tuple[TestClient, FakeAgent],
) -> None:
    http, agent = setup
    valid = service.Draft(
        answer="Your order is in transit.",
        policy_ids=[],
    )
    invalid = service.Draft(
        answer="I don't have verified order information. [POL-SHIPPING]",
        policy_ids=["POL-SHIPPING"],
    )
    drafts = iter((invalid, valid))
    agent.invoke = lambda messages: next(drafts)  # type: ignore[method-assign]

    result = http.post(
        "/api/chat",
        json={"message": "Where is ORD-5001?", "customer_id": "CUST-1024"},
    )

    assert result.status_code == 200
    assert result.json()["answer"] == valid.answer
    assert result.json()["order"]["status"] == "in transit"


def test_failed_turn_does_not_enter_history(
    setup: tuple[TestClient, FakeAgent], monkeypatch: pytest.MonkeyPatch
) -> None:
    http, agent = setup
    first = http.post("/api/chat", json={"message": "return policy"}).json()
    original = agent.invoke

    def fail(messages: list[HumanMessage | AIMessage]) -> service.Draft:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agent, "invoke", fail)
    failed = http.post(
        "/api/chat",
        json={
            "message": "return policy",
            "conversation_id": first["conversation_id"],
        },
    )
    assert failed.status_code == 502
    assert failed.json()["detail"] == "Assistant execution failed"
    monkeypatch.setattr(agent, "invoke", original)
    retry = http.post(
        "/api/chat",
        json={
            "message": "return policy",
            "conversation_id": first["conversation_id"],
        },
    )
    assert retry.status_code == 200
    assert len(agent.calls[-1]) == 3


def test_feedback_rating_correction_and_review_lifecycle(
    setup: tuple[TestClient, FakeAgent], monkeypatch: pytest.MonkeyPatch
) -> None:
    http, _ = setup
    message_id = http.post("/api/chat", json={"message": "return policy"}).json()[
        "message_id"
    ]
    assert (
        http.post(
            "/api/feedback", json={"message_id": str(uuid4()), "rating": "up"}
        ).status_code
        == 404
    )
    assert (
        http.post("/api/feedback", json={"message_id": message_id}).status_code == 422
    )
    assert (
        http.post(
            "/api/feedback", json={"message_id": message_id, "correction": "  "}
        ).status_code
        == 422
    )
    response = http.post(
        "/api/feedback",
        json={
            "message_id": message_id,
            "rating": "down",
            "correction": "The approved return period should be checked by staff.",
        },
    )
    assert response.status_code == 200
    ids = response.json()["feedback_ids"]
    assert len(ids) == 2 and response.json()["status"] == "received"
    rows = http.get(f"/api/feedback/{message_id}").json()
    assert {row["category"] for row in rows} == {"rating", "correction"}
    assert all(row["status"] == "received" for row in rows)
    monkeypatch.delenv("REVIEW_TOKEN", raising=False)
    path = f"/api/feedback/{ids[1]}/resolve"
    body = {"approved": False, "reason": "Seeded policy remains authoritative"}
    assert http.post(path, json=body).status_code == 503
    monkeypatch.setenv("REVIEW_TOKEN", "test-review-token")
    assert http.post(path, json=body).status_code == 403
    approved = http.post(
        path, json=body, headers={"X-Review-Token": "test-review-token"}
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "resolved"
    assert (
        http.post(
            path, json=body, headers={"X-Review-Token": "test-review-token"}
        ).status_code
        == 409
    )
    assert http.get(f"/api/feedback/{message_id}").json()[1]["status"] == "resolved"


def test_real_model_wiring_and_guardrail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    structured_options: dict[str, object] = {}
    capabilities = Capabilities()

    def fake_create_agent(**kwargs: object) -> object:
        captured.update(kwargs)
        capabilities.behavior = BehaviorEngine(
            policies=kwargs["behavior_options"]["policies"]
        )
        return SimpleNamespace(
            capabilities=capabilities,
            close_observability=lambda: None,
        )

    monkeypatch.setenv("EXPLABS_API_KEY", "test-only")
    monkeypatch.setenv("EXPLABS_MODEL", "test-model")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(service, "create_agent", fake_create_agent)

    def fake_structured_output(
        agent: object, schema: object, **kwargs: object
    ) -> object:
        structured_options.update(kwargs)
        return SimpleNamespace(
            instructions="Reply in the structured envelope.",
            invoke=lambda messages: SimpleNamespace(
                structured={
                    "answer": "Grounded. [POL-RETURNS]",
                    "policy_ids": ["POL-RETURNS"],
                }
            ),
        )

    monkeypatch.setattr(service, "with_structured_output", fake_structured_output)
    runner = service.make_agent(Store())
    assert (
        runner.invoke([HumanMessage(content="return")]).answer
        == "Grounded. [POL-RETURNS]"
    )
    assert captured["model"].model_name == "test-model"
    assert (
        str(captured["model"].openai_api_base).rstrip("/")
        == "https://example.invalid/v1"
    )
    assert captured["pattern"] == "react"
    assert captured["xai_options"]["application_id"] == "qa-customer-chatbot"
    assert captured["observers"] == ()
    assert captured["feedback_options"] == {}
    assert set(captured) == set(inspect.signature(create_agent).parameters)
    assert structured_options["inject_instructions"] is False
    assert structured_options["repair"] is not None
    assert structured_options["repair_config"].max_attempts == 1
    tools = {item.name: item for item in captured["tools"]}
    token_customer = service.active_customer.set("CUST-1024")
    token_scope = service.active_conversation.set("guardrail-test")
    token_order = service.active_order.set("ORD-5001")
    try:
        assert (
            json.loads(tools["order_status"].invoke({"order_id": "ORD-5001"}))["status"]
            == "in transit"
        )
        assert "not selected" in tools["order_status"].invoke({"order_id": "ORD-5002"})
        assert "POL-RETURNS" in tools["policy_lookup"].invoke(
            {"policy_id": "POL-RETURNS"}
        )
        assert tools["order_status"].invoke({"order_id": "ORD-5001"})
        assert tools["order_status"].invoke({"order_id": "ORD-5001"})
        with pytest.raises(RuntimeError, match="BehaviorWeave"):
            tools["order_status"].invoke({"order_id": "ORD-5001"})
    finally:
        service.active_customer.reset(token_customer)
        service.active_conversation.reset(token_scope)
        service.active_order.reset(token_order)


def test_missing_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    http = TestClient(service.create_app())
    health = http.get("/api/health").json()
    assert health["status"] == "unconfigured"
    assert "feedback" in health["agloom_features"]
    assert "behavior_guardrails" in health["agloom_features"]
    assert "structured_output" in health["agloom_features"]
    assert health["observability"]["enabled"] is False
    governance = http.get("/api/governance").json()
    assert governance["guardrail_policy"] == service.GUARDRAIL
    assert governance["feedback_review_required"] is True
    assert http.post("/api/chat", json={"message": "return policy"}).status_code == 503


def test_real_agloom_structured_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolCapableModel(FakeListChatModel):
        def bind_tools(self, tools: object, **kwargs: object) -> ToolCapableModel:
            return self

    monkeypatch.setenv("EXPLABS_API_KEY", "test-only")
    monkeypatch.setattr(
        service,
        "ChatOpenAI",
        lambda **kwargs: ToolCapableModel(
            responses=[
                '<xstructured>{"answer":"Unused items can be returned within 30 days. '
                '[POL-RETURNS]","policy_ids":["POL-RETURNS"]}</xstructured>',
            ]
        ),
    )
    http = TestClient(service.create_app())
    result = http.post("/api/chat", json={"message": "What is the return policy?"})
    assert result.status_code == 200
    assert result.json()["sources"][0]["id"] == "POL-RETURNS"
    assert result.json()["answer"].startswith("Unused items")
