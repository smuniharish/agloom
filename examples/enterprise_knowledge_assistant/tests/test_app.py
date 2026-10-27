from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from agentbarrier.runtime import PolicyEffect
from backend.app import (
    SYSTEM_PROMPT,
    create_app,
    knowledge_tools,
    make_agent,
    tool_policy,
)
from backend.knowledge import KnowledgeBase
from backend.observability import Telemetry
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agloom import AgentEvent, EventName, ExecutionError, Observer, PolicyManager


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


class RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []
        self.answer = "Report the lost device immediately. [IT-101-2]"
        self.error: Exception | None = None
        self.closed = False

    async def ainvoke(self, input: Sequence[BaseMessage]) -> AIMessage:
        self.calls.append(list(input))
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.answer)

    def close(self) -> None:
        self.closed = True


class RecordingObserver(Observer):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.shut_down = False

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)

    def shutdown(self) -> None:
        self.shut_down = True


@pytest.fixture
def recording() -> RecordingAgent:
    return RecordingAgent()


@pytest.fixture
def http(kb: KnowledgeBase, recording: RecordingAgent) -> Iterator[TestClient]:
    with TestClient(create_app(kb, agent_factory=lambda *_: recording)) as client:
        yield client


def test_retrieval_is_ranked_and_does_not_invent_matches(kb: KnowledgeBase) -> None:
    hits = kb.search("lost company device")
    assert hits[0].id == "IT-101-2"
    assert "remotely locks" in hits[0].text
    assert kb.search("xylophone quasar nebula") == []
    assert kb.get("NOT-REAL-1") is None
    assert len(kb.documents()) == 6


def test_duplicate_passage_ids_fail(tmp_path: Path) -> None:
    path = tmp_path / "docs.json"
    path.write_text(
        '[{"id":"X","title":"One","owner":"A","updated":"2026",'
        '"sections":[{"heading":"A","text":"first"}]},'
        '{"id":"X","title":"Two","owner":"B","updated":"2026",'
        '"sections":[{"heading":"B","text":"second"}]}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate passage ID"):
        KnowledgeBase(path)


def test_chat_citations_and_conversation(
    http: TestClient, recording: RecordingAgent
) -> None:
    first = http.post("/api/chat", json={"question": "How do I report a lost device?"})
    assert first.status_code == 200
    data = first.json()
    assert data["citations"][0]["id"] == "IT-101-2"
    assert data["citations"][0]["owner"] == "IT Operations"
    assert "Report a lost company device" in str(recording.calls[0][-1].content)
    assert len(http.get("/api/documents").json()) == 6

    second = http.post(
        "/api/chat",
        json={
            "question": "What about the lost device?",
            "conversation_id": data["conversation_id"],
        },
    )
    assert second.status_code == 200
    history = recording.calls[1]
    assert len(history) == 3
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "How do I report a lost device?"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == data["answer"]
    stored = http.get(f"/api/conversations/{data['conversation_id']}").json()
    assert [item["role"] for item in stored["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_unknown_and_invalid_conversations(http: TestClient) -> None:
    unknown = "00000000-0000-0000-0000-000000000000"
    assert http.get(f"/api/conversations/{unknown}").status_code == 404
    assert (
        http.post(
            "/api/chat", json={"question": "lost device", "conversation_id": unknown}
        ).status_code
        == 404
    )
    assert http.post("/api/chat", json={"question": "   "}).status_code == 422


def test_no_evidence_does_not_call_model(
    http: TestClient, recording: RecordingAgent
) -> None:
    result = http.post("/api/chat", json={"question": "xylophone quasar nebula"})
    assert result.status_code == 200
    assert result.json()["citations"] == []
    assert "couldn't find" in result.json()["answer"]
    assert recording.calls == []


@pytest.mark.parametrize(
    "answer",
    [
        "Report immediately.",
        "See [FAKE-999-1].",
        "Report it [IT-101-2] and see [FAKE-999-1].",
    ],
)
def test_unsupported_citations_are_rejected(
    http: TestClient, recording: RecordingAgent, answer: str
) -> None:
    recording.answer = answer
    result = http.post("/api/chat", json={"question": "lost device"})
    assert result.status_code == 502
    assert "valid citations" in result.json()["detail"]


@pytest.mark.parametrize(
    ("answer", "error"),
    [("Report it. [IT-101-2]", ExecutionError("provider failed")), ("   ", None)],
)
def test_agent_failures_return_bad_gateway(
    http: TestClient, recording: RecordingAgent, answer: str, error: Exception | None
) -> None:
    recording.answer, recording.error = answer, error
    result = http.post("/api/chat", json={"question": "lost device"})
    assert result.status_code == 502
    assert result.json()["detail"] == "The assistant could not answer this question"


def test_shutdown_closes_the_agent(
    kb: KnowledgeBase, recording: RecordingAgent
) -> None:
    received: list[tuple[PolicyManager, tuple[Observer, ...]]] = []

    def factory(
        _: KnowledgeBase, policy: PolicyManager, observers: Sequence[Observer]
    ) -> RecordingAgent:
        received.append((policy, tuple(observers)))
        return recording

    with TestClient(create_app(kb, agent_factory=factory)) as client:
        assert client.post("/api/chat", json={"question": "lost device"}).is_success
    assert recording.closed
    policy, observers = received[0]
    assert policy.policy.version == "enterprise-knowledge-assistant-v1"
    assert observers == ()


def test_shutdown_releases_telemetry_without_an_agent(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer = RecordingObserver()
    monkeypatch.setattr(
        "backend.app.start_telemetry", lambda *_, **__: Telemetry((observer,))
    )
    with TestClient(create_app(kb)) as client:
        assert client.get("/api/health").is_success
    assert observer.shut_down


def test_tool_policy_allows_only_the_knowledge_tools() -> None:
    manager = tool_policy()
    try:
        runtime = manager.policy
        assert runtime.evaluate("search_knowledge", {"query": "x"}).effect is (
            PolicyEffect.ALLOW
        )
        assert runtime.evaluate("read_passage", {"passage_id": "IT-101-2"}).effect is (
            PolicyEffect.ALLOW
        )
        denied = runtime.evaluate("send_email", {"to": "all@example.com"})
        assert denied.effect is PolicyEffect.DENY
        assert denied.rule_name == "<default>"
        assert manager.visible("read_passage")
        assert not manager.visible("send_email")
        assert manager.consequential == {}
        assert manager.on_denied == "inform_model"
    finally:
        manager.store.close()


def test_provider_configuration_and_tool_wiring(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_agent(**kwargs: Any) -> RecordingAgent:
        captured.update(kwargs)
        return RecordingAgent()

    monkeypatch.setenv("EXPLABS_API_KEY", "test-only-key")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EXPLABS_MODEL", "test-model")
    monkeypatch.setattr("backend.app.create_agent", fake_create_agent)
    policy = tool_policy()
    observer = RecordingObserver()
    try:
        make_agent(kb, policy, [observer])
    finally:
        policy.store.close()
    model = captured["model"]
    assert model.model_name == "test-model"
    assert str(model.openai_api_base).rstrip("/") == "https://example.invalid/v1"
    assert captured["pattern"] == "react"
    assert captured["system_prompt"] == SYSTEM_PROMPT
    assert captured["limits"].max_depth == 0
    assert captured["policy"] is policy
    assert list(captured["observers"]) == [observer]
    tools = captured["tools"]
    assert [tool.name for tool in tools] == ["search_knowledge", "read_passage"]
    assert "IT-101-2" in tools[0].invoke({"query": "lost device"})
    assert "remotely locks" in tools[1].invoke({"passage_id": "IT-101-2"})
    assert "Passage not found" in tools[1].invoke({"passage_id": "invalid"})


def test_real_agent_cites_a_tool_discovered_passage(
    kb: KnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    scripted_model: Callable[..., Any],
) -> None:
    model = scripted_model(
        tool_call("read_passage", {"passage_id": "HR-310-1"}, "call-1"),
        AIMessage(content="Submit vacation in the HR portal. [HR-310-1]"),
    )
    monkeypatch.setattr("backend.app.chat_model", lambda: model)
    with TestClient(create_app(kb)) as client:
        result = client.post("/api/chat", json={"question": "lost device"})
    assert result.status_code == 200
    assert [item["id"] for item in result.json()["citations"]] == ["HR-310-1"]
    tool_results = [
        message for message in model.received[-1] if isinstance(message, ToolMessage)
    ]
    assert json.loads(str(tool_results[0].content))["id"] == "HR-310-1"


def test_policy_denies_unlisted_tools_and_informs_the_model(
    kb: KnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    scripted_model: Callable[..., Any],
) -> None:
    model = scripted_model(
        tool_call("delete_document", {"document_id": "IT-101"}, "call-1"),
        AIMessage(content="I can only read documents. [IT-101-2]"),
    )
    monkeypatch.setattr("backend.app.chat_model", lambda: model)
    observer = RecordingObserver()
    policy = tool_policy()
    try:
        agent = make_agent(kb, policy, [observer])
        answer = asyncio.run(
            agent.ainvoke([HumanMessage(content="Delete the device policy.")])
        )
    finally:
        policy.store.close()
    assert answer.content == "I can only read documents. [IT-101-2]"
    denied = [
        event for event in observer.events if event.name == EventName.TOOL_CALL_DENIED
    ]
    assert [(event.payload["tool"], event.payload["reason"]) for event in denied] == [
        ("delete_document", "PolicyViolationError")
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


def test_knowledge_tools_record_seen_passages(kb: KnowledgeBase) -> None:
    from backend.app import seen_passages

    search, read = knowledge_tools(kb)
    seen: set[str] = set()
    token = seen_passages.set(seen)
    try:
        search.invoke({"query": "expense claim"})
        read.invoke({"passage_id": "HR-310-1"})
    finally:
        seen_passages.reset(token)
    assert "HR-310-1" in seen
    assert any(passage.startswith("FIN-") for passage in seen)


def test_missing_credentials_are_explicit(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    monkeypatch.delenv("AGLOOM_OBSERVABILITY_ENABLED", raising=False)
    with TestClient(create_app(kb)) as http:
        health = http.get("/api/health").json()
        assert health["status"] == "unconfigured"
        assert health["agloom_features"] == [
            "react",
            "local_tools",
            "policy_governed_tools",
            "grounded_citations",
            "conversation_context",
        ]
        assert health["observability"]["enabled"] is False
        governance = http.get("/api/governance").json()
        assert governance["evidence_policy"] == "exact_passage_citations"
        assert governance["capability_scope"] == ["search_knowledge", "read_passage"]
        assert governance["tool_policy"] == {
            "version": "enterprise-knowledge-assistant-v1",
            "default_effect": "deny",
            "rules": [
                {
                    "name": "search approved knowledge",
                    "effect": "allow",
                    "tool": "search_knowledge",
                },
                {
                    "name": "read approved passages",
                    "effect": "allow",
                    "tool": "read_passage",
                },
            ],
            "consequential_tools": [],
        }
        result = http.post("/api/chat", json={"question": "lost device"})
        assert result.status_code == 503
        assert (
            result.json()["detail"] == "EXPLABS_API_KEY is required to answer questions"
        )
