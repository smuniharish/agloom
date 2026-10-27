from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from agentbarrier.runtime import PolicyEffect
from backend import app as service
from backend.store import Store
from behaviorweave.integrations.langchain import BehaviorWeaveMiddleware
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from xstructured import XStructuredError

from agloom import AgentEvent, EventName, ExecutionError, Observer, PolicyManager

RETURNS = service.Draft(
    answer="Unused items can be returned within 30 days. [POL-RETURNS]",
    policy_ids=["POL-RETURNS"],
)


def envelope(answer: str, policy_ids: list[str]) -> AIMessage:
    payload = json.dumps({"answer": answer, "policy_ids": policy_ids})
    return AIMessage(content=f"<xstructured>{payload}</xstructured>")


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


class FakeDrafter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[BaseMessage], str]] = []
        self.outcomes: list[service.Draft | Exception] = []
        self.default: service.Draft = RETURNS
        self.closed = False

    async def draft(
        self, messages: Sequence[BaseMessage], *, conversation_id: str
    ) -> service.Draft:
        self.calls.append((list(messages), conversation_id))
        outcome = self.outcomes.pop(0) if self.outcomes else self.default
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


class RecordingObserver(Observer):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)


@pytest.fixture
def drafter() -> FakeDrafter:
    return FakeDrafter()


@pytest.fixture
def http(store: Store, drafter: FakeDrafter) -> Iterator[TestClient]:
    app = service.create_app(store, drafter_factory=lambda *_: drafter)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def real_drafter_app(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., TestClient]:
    def build(model: Any) -> TestClient:
        monkeypatch.setattr("backend.app.chat_model", lambda: model)
        return TestClient(service.create_app(store))

    return build


def test_seeded_lookup_and_duplicate_ids(tmp_path: Path, store: Store) -> None:
    order = store.order("CUST-1024", "ORD-5001")
    assert order is not None
    assert order.status == "in transit"
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


def test_evidence_prompt_defines_exact_citation_contract(store: Store) -> None:
    policy = store.policies["POL-RETURNS"]
    prompt = service.evidence_prompt("What is the return policy?", [policy], None)
    assert "Allowed policy IDs: POL-RETURNS." in prompt
    assert "same IDs in policy_ids" in prompt
    assert "Do not put an ID in only one field" in prompt


def test_chat_order_grounding_and_conversation(
    http: TestClient, drafter: FakeDrafter
) -> None:
    first = http.post(
        "/api/chat",
        json={"message": "What is the return policy?", "customer_id": "CUST-1024"},
    )
    assert first.status_code == 200
    data = first.json()
    assert data["sources"][0]["id"] == "POL-RETURNS"
    assert data["order"] is None
    assert data["guardrail"] == {"status": "active", "policy": service.GUARDRAIL}
    first_messages, conversation_id = drafter.calls[0]
    assert conversation_id == data["conversation_id"]
    assert "POL-RETURNS" in str(first_messages[-1].content)
    drafter.default = service.Draft(answer="Your order is in transit.", policy_ids=[])
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
    second_messages, _ = drafter.calls[1]
    assert len(second_messages) == 3
    assert isinstance(second_messages[1], AIMessage)
    assert second_messages[1].content == data["answer"]
    assert "in transit" in str(second_messages[-1].content)


def test_no_order_leakage_and_no_evidence(
    http: TestClient, drafter: FakeDrafter
) -> None:
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
    assert drafter.calls == []


def test_invalid_input_and_conversation_ownership(http: TestClient) -> None:
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
    http: TestClient, drafter: FakeDrafter, draft: service.Draft
) -> None:
    drafter.default = draft
    result = http.post("/api/chat", json={"message": "return policy"})
    assert result.status_code == 502
    assert "invalid evidence" in result.json()["detail"]
    assert len(drafter.calls) == 2


def test_invalid_model_evidence_is_repaired_once(
    http: TestClient, drafter: FakeDrafter
) -> None:
    invalid = service.Draft(
        answer="Unused items can be returned within 30 days.",
        policy_ids=["POL-RETURNS"],
    )
    drafter.outcomes = [invalid, RETURNS]

    result = http.post("/api/chat", json={"message": "return policy"})

    assert result.status_code == 200
    assert result.json()["answer"] == RETURNS.answer
    assert result.json()["sources"][0]["id"] == "POL-RETURNS"
    repair, _ = drafter.calls[1]
    assert isinstance(repair[-2], AIMessage)
    assert repair[-2].content == invalid.model_dump_json()
    assert repair[-1].content == service.REPAIR_INSTRUCTION


def test_malformed_structured_answer_is_repaired(
    http: TestClient, drafter: FakeDrafter
) -> None:
    drafter.outcomes = [XStructuredError("no envelope"), RETURNS]
    result = http.post("/api/chat", json={"message": "return policy"})
    assert result.status_code == 200
    repair, _ = drafter.calls[1]
    assert isinstance(repair[-2], HumanMessage)
    assert repair[-1].content == service.REPAIR_INSTRUCTION


def test_order_answer_without_verified_status_is_repaired(
    http: TestClient, drafter: FakeDrafter
) -> None:
    valid = service.Draft(answer="Your order is in transit.", policy_ids=[])
    invalid = service.Draft(
        answer="I don't have verified order information. [POL-SHIPPING]",
        policy_ids=["POL-SHIPPING"],
    )
    drafter.outcomes = [invalid, valid]

    result = http.post(
        "/api/chat",
        json={"message": "Where is ORD-5001?", "customer_id": "CUST-1024"},
    )

    assert result.status_code == 200
    assert result.json()["answer"] == valid.answer
    assert result.json()["order"]["status"] == "in transit"


@pytest.mark.parametrize(
    "outcomes",
    [
        [ExecutionError("provider unavailable")],
        [XStructuredError("malformed"), XStructuredError("still malformed")],
    ],
)
def test_failed_turn_does_not_enter_history(
    http: TestClient, drafter: FakeDrafter, outcomes: list[Exception]
) -> None:
    first = http.post("/api/chat", json={"message": "return policy"}).json()
    drafter.outcomes = list(outcomes)
    failed = http.post(
        "/api/chat",
        json={"message": "return policy", "conversation_id": first["conversation_id"]},
    )
    assert failed.status_code == 502
    assert failed.json()["detail"] == "Assistant execution failed"
    retry = http.post(
        "/api/chat",
        json={"message": "return policy", "conversation_id": first["conversation_id"]},
    )
    assert retry.status_code == 200
    assert len(drafter.calls[-1][0]) == 3


def test_feedback_rating_correction_and_review_lifecycle(
    http: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    body = {
        "message_id": message_id,
        "rating": "down",
        "correction": "The approved return period should be checked by staff.",
    }
    response = http.post("/api/feedback", json=body)
    assert response.status_code == 200
    ids = response.json()["feedback_ids"]
    assert len(ids) == 2
    assert response.json()["status"] == "received"
    assert http.post("/api/feedback", json=body).json()["feedback_ids"] == ids
    rows = http.get(f"/api/feedback/{message_id}").json()
    assert {row["category"] for row in rows} == {"rating", "correction"}
    assert len(rows) == 2
    assert all(row["status"] == "received" for row in rows)
    monkeypatch.delenv("REVIEW_TOKEN", raising=False)
    path = f"/api/feedback/{ids[1]}/resolve"
    decision = {"approved": False, "reason": "Seeded policy remains authoritative"}
    assert http.post(path, json=decision).status_code == 503
    monkeypatch.setenv("REVIEW_TOKEN", "test-review-token")
    assert http.post(path, json=decision).status_code == 403
    headers = {"X-Review-Token": "test-review-token"}
    approved = http.post(path, json=decision, headers=headers)
    assert approved.status_code == 200
    assert approved.json()["status"] == "resolved"
    assert http.post(path, json=decision, headers=headers).status_code == 409
    missing = f"/api/feedback/{uuid4()}/resolve"
    assert http.post(missing, json=decision, headers=headers).status_code == 404
    statuses = {
        row["category"]: row["status"]
        for row in http.get(f"/api/feedback/{message_id}").json()
    }
    assert statuses == {"rating": "received", "correction": "resolved"}


def test_feedback_does_not_need_a_model(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    with TestClient(service.create_app(store)) as http:
        canned = http.post("/api/chat", json={"message": "Where is my order?"})
        assert canned.status_code == 200
        rated = http.post(
            "/api/feedback",
            json={"message_id": canned.json()["message_id"], "rating": "up"},
        )
        assert rated.status_code == 200
        assert len(rated.json()["feedback_ids"]) == 1


def test_tool_policy_allows_only_the_support_tools() -> None:
    manager = service.tool_policy()
    try:
        runtime = manager.policy
        assert runtime.evaluate("order_status", {"order_id": "ORD-5001"}).effect is (
            PolicyEffect.ALLOW
        )
        assert runtime.evaluate("policy_lookup", {"policy_id": "x"}).effect is (
            PolicyEffect.ALLOW
        )
        assert runtime.evaluate("cancel_order", {}).effect is PolicyEffect.DENY
        assert not manager.visible("issue_refund")
        assert manager.consequential == {}
        assert manager.on_denied == "inform_model"
    finally:
        manager.store.close()


def test_drafter_wiring_and_scoped_tools(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class StubAgent:
        def with_structured_output(self, schema: Any, **options: Any) -> Any:
            captured["structured"] = (schema, options)
            return object()

    def fake_create_agent(**kwargs: Any) -> StubAgent:
        captured.update(kwargs)
        return StubAgent()

    monkeypatch.setenv("EXPLABS_API_KEY", "test-only")
    monkeypatch.setenv("EXPLABS_MODEL", "test-model")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(service, "create_agent", fake_create_agent)
    policy = service.tool_policy()
    observer = RecordingObserver()
    try:
        service.make_drafter(store, policy, [observer])
    finally:
        policy.store.close()
    assert captured["model"].model_name == "test-model"
    assert (
        str(captured["model"].openai_api_base).rstrip("/")
        == "https://example.invalid/v1"
    )
    assert captured["pattern"] == "react"
    assert captured["system_prompt"] == service.SYSTEM_PROMPT
    assert captured["limits"].max_depth == 0
    assert captured["policy"] is policy
    assert list(captured["observers"]) == [observer]
    (guardrail,) = captured["middleware"]
    assert isinstance(guardrail, BehaviorWeaveMiddleware)
    assert guardrail.on_decision is service.record_behavior
    assert captured["structured"] == (service.Draft, {})
    tools = {item.name: item for item in captured["tools"]}
    token_customer = service.active_customer.set("CUST-1024")
    token_order = service.active_order.set("ORD-5001")
    try:
        order = json.loads(tools["order_status"].invoke({"order_id": "ORD-5001"}))
        assert order["status"] == "in transit"
        assert "not selected" in tools["order_status"].invoke({"order_id": "ORD-5002"})
        assert "POL-RETURNS" in tools["policy_lookup"].invoke(
            {"policy_id": "POL-RETURNS"}
        )
        assert "Unknown" in tools["policy_lookup"].invoke({"policy_id": "POL-X"})
    finally:
        service.active_customer.reset(token_customer)
        service.active_order.reset(token_order)
    assert "No order approved" in tools["order_status"].invoke({})


def test_real_agent_answers_with_the_verified_order(
    real_drafter_app: Callable[..., TestClient],
    scripted_model: Callable[..., Any],
) -> None:
    model = scripted_model(
        tool_call("order_status", {"order_id": "ORD-5001"}, "call-1"),
        envelope("Your travel backpack is in transit.", []),
    )
    with real_drafter_app(model) as http:
        result = http.post(
            "/api/chat",
            json={"message": "Where is ORD-5001?", "customer_id": "CUST-1024"},
        )
    assert result.status_code == 200, result.text
    assert result.json()["answer"] == "Your travel backpack is in transit."
    assert result.json()["order"]["id"] == "ORD-5001"
    first_prompt = model.received[0][-1]
    assert "<xstructured>" in str(first_prompt.content)
    tool_result = next(
        message for message in model.received[1] if isinstance(message, ToolMessage)
    )
    assert json.loads(str(tool_result.content))["status"] == "in transit"


def test_real_agent_guardrail_stops_the_third_identical_lookup(
    real_drafter_app: Callable[..., TestClient],
    scripted_model: Callable[..., Any],
) -> None:
    replies: list[AIMessage] = []
    for turn in range(3):
        replies.append(
            tool_call("order_status", {"order_id": "ORD-5001"}, f"call-{turn}")
        )
        replies.append(envelope("Your order is in transit.", []))
    model = scripted_model(*replies)
    request = {"message": "Where is ORD-5001?", "customer_id": "CUST-1024"}
    with real_drafter_app(model) as http:
        first = http.post("/api/chat", json=request).json()
        follow_up = {**request, "conversation_id": first["conversation_id"]}
        second = http.post("/api/chat", json=follow_up).json()
        third = http.post("/api/chat", json=follow_up).json()
    assert first["guardrail"]["status"] == "active"
    assert second["guardrail"]["status"] == "active"
    assert third["guardrail"] == {"status": "intervened", "policy": service.GUARDRAIL}
    assert third["order"]["status"] == "in transit"
    blocked = next(
        message for message in model.received[-1] if isinstance(message, ToolMessage)
    )
    assert blocked.status == "error"
    assert "Repeated identical lookups have been stopped." in str(blocked.content)


def test_real_agent_policy_denial_is_reported_to_the_model(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    scripted_model: Callable[..., Any],
) -> None:
    model = scripted_model(
        tool_call("cancel_order", {"order_id": "ORD-5001"}, "call-1"),
        envelope("I cannot cancel orders. [POL-RETURNS]", ["POL-RETURNS"]),
    )
    monkeypatch.setattr("backend.app.chat_model", lambda: model)
    observer = RecordingObserver()
    policy = service.tool_policy()
    try:
        drafter = service.make_drafter(store, policy, [observer])
        draft = asyncio.run(
            drafter.draft(
                [HumanMessage(content="Cancel ORD-5001 please.")],
                conversation_id="conversation-1",
            )
        )
        drafter.close()
    finally:
        policy.store.close()
    assert draft.policy_ids == ["POL-RETURNS"]
    denied = [
        event for event in observer.events if event.name == EventName.TOOL_CALL_DENIED
    ]
    assert [(event.payload["tool"], event.payload["reason"]) for event in denied] == [
        ("cancel_order", "PolicyViolationError")
    ]
    selected = [
        event for event in observer.events if event.name == EventName.TOPOLOGY_SELECTED
    ]
    assert [event.topology for event in selected] == ["react"]
    refusal = next(
        message for message in model.received[-1] if isinstance(message, ToolMessage)
    )
    assert refusal.status == "error"
    assert "was not executed" in str(refusal.content)


def test_missing_model_configuration(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    monkeypatch.delenv("AGLOOM_OBSERVABILITY_ENABLED", raising=False)
    with TestClient(service.create_app(store)) as http:
        health = http.get("/api/health").json()
        assert health["status"] == "unconfigured"
        assert health["agloom_features"] == [
            "react",
            "local_tools",
            "policy_governed_tools",
            "structured_output",
            "behavior_guardrails",
            "feedback",
            "conversation_context",
        ]
        assert health["observability"]["enabled"] is False
        governance = http.get("/api/governance").json()
        assert governance["guardrail_policy"] == service.GUARDRAIL
        assert governance["feedback_review_required"] is True
        assert governance["tool_policy"]["version"] == "qa-customer-chatbot-v1"
        assert governance["tool_policy"]["default_effect"] == "deny"
        assert [rule["tool"] for rule in governance["tool_policy"]["rules"]] == [
            "order_status",
            "policy_lookup",
        ]
        result = http.post("/api/chat", json={"message": "return policy"})
        assert result.status_code == 503
        assert "EXPLABS_API_KEY" in result.json()["detail"]


def test_shutdown_closes_the_drafter(store: Store, drafter: FakeDrafter) -> None:
    received: list[PolicyManager] = []

    def factory(
        _: Store, policy: PolicyManager, observers: Sequence[Observer]
    ) -> FakeDrafter:
        received.append(policy)
        return drafter

    with TestClient(service.create_app(store, drafter_factory=factory)) as http:
        assert http.post("/api/chat", json={"message": "return policy"}).is_success
    assert drafter.closed
    assert received[0].policy.version == "qa-customer-chatbot-v1"
