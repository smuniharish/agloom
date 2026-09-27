from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from backend.app import create_app, make_agent
from backend.knowledge import KnowledgeBase
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from agloom import create_agent


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase()


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


class RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[list[HumanMessage | AIMessage]] = []
        self.answer = "Report the lost device immediately. [IT-101-2]"

    def invoke(self, input: list[HumanMessage | AIMessage]) -> AIMessage:
        self.calls.append(input)
        return AIMessage(content=self.answer)


@pytest.fixture
def client(kb: KnowledgeBase) -> tuple[TestClient, RecordingAgent]:
    agent = RecordingAgent()
    return TestClient(create_app(kb, agent_factory=lambda _: agent)), agent


def test_chat_citations_and_conversation(
    client: tuple[TestClient, RecordingAgent],
) -> None:
    http, agent = client
    first = http.post("/api/chat", json={"question": "How do I report a lost device?"})
    assert first.status_code == 200
    data = first.json()
    assert data["citations"][0]["id"] == "IT-101-2"
    assert data["citations"][0]["owner"] == "IT Operations"
    assert "Report a lost company device" in str(agent.calls[0][-1].content)
    assert len(http.get("/api/documents").json()) == 6

    second = http.post(
        "/api/chat",
        json={
            "question": "What about the lost device?",
            "conversation_id": data["conversation_id"],
        },
    )
    assert second.status_code == 200
    assert len(agent.calls[1]) == 3
    assert agent.calls[1][0].content == "How do I report a lost device?"
    assert agent.calls[1][1].content == data["answer"]
    stored = http.get(f"/api/conversations/{data['conversation_id']}").json()
    assert [item["role"] for item in stored["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_unknown_and_invalid_conversations(
    client: tuple[TestClient, RecordingAgent],
) -> None:
    http, _ = client
    assert (
        http.get("/api/conversations/00000000-0000-0000-0000-000000000000").status_code
        == 404
    )
    assert (
        http.post(
            "/api/chat",
            json={
                "question": "lost device",
                "conversation_id": "00000000-0000-0000-0000-000000000000",
            },
        ).status_code
        == 404
    )
    assert http.post("/api/chat", json={"question": "   "}).status_code == 422


def test_no_evidence_does_not_call_model(
    client: tuple[TestClient, RecordingAgent],
) -> None:
    http, agent = client
    result = http.post("/api/chat", json={"question": "xylophone quasar nebula"})
    assert result.status_code == 200
    assert result.json()["citations"] == []
    assert "couldn't find" in result.json()["answer"]
    assert agent.calls == []


@pytest.mark.parametrize(
    "answer",
    [
        "Report immediately.",
        "See [FAKE-999-1].",
        "Report it [IT-101-2] and see [FAKE-999-1].",
    ],
)
def test_unsupported_citations_are_rejected(
    client: tuple[TestClient, RecordingAgent], answer: str
) -> None:
    http, agent = client
    agent.answer = answer
    result = http.post("/api/chat", json={"question": "lost device"})
    assert result.status_code == 502
    assert "valid citations" in result.json()["detail"]


def test_provider_configuration_and_tool_wiring(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_create_agent(**kwargs: object) -> RecordingAgent:
        captured.update(kwargs)
        return RecordingAgent()

    monkeypatch.setenv("EXPLABS_API_KEY", "test-only-key")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EXPLABS_MODEL", "test-model")
    monkeypatch.setattr("backend.app.create_agent", fake_create_agent)
    make_agent(kb)
    model = captured["model"]
    assert model.model_name == "test-model"
    assert str(model.openai_api_base).rstrip("/") == "https://example.invalid/v1"
    assert captured["pattern"] == "react"
    assert captured["xai_options"]["application_id"] == (
        "enterprise-knowledge-assistant"
    )
    assert captured["observers"] == ()
    assert set(captured) == set(inspect.signature(create_agent).parameters)
    tools = captured["tools"]
    assert "IT-101-2" in tools[0].invoke({"query": "lost device"})
    assert "remotely locks" in tools[1].invoke({"passage_id": "IT-101-2"})
    assert "Passage not found" in tools[1].invoke({"passage_id": "invalid"})


def test_real_agloom_agent_constructs_without_network(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXPLABS_API_KEY", "test-only-key")
    monkeypatch.setenv("EXPLABS_BASE_URL", "https://example.invalid/v1")
    agent = make_agent(kb)
    assert agent.config.pattern_candidates[0].value == "react"
    assert {tool.name for tool in agent.config.tools} == {
        "search_knowledge",
        "read_passage",
    }


def test_tool_discovered_passage_is_citable(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ToolUsingAgent:
        def __init__(self, read_tool: object) -> None:
            self.read_tool = read_tool

        def invoke(self, input: list[HumanMessage | AIMessage]) -> AIMessage:
            self.read_tool.invoke({"passage_id": "HR-310-1"})
            return AIMessage(content="Submit vacation in the HR portal. [HR-310-1]")

    def factory(knowledge: KnowledgeBase) -> ToolUsingAgent:
        def fake_create_agent(**kwargs: object) -> ToolUsingAgent:
            return ToolUsingAgent(kwargs["tools"][1])

        monkeypatch.setattr("backend.app.create_agent", fake_create_agent)
        monkeypatch.setenv("EXPLABS_API_KEY", "test-only-key")
        return make_agent(knowledge)

    http = TestClient(create_app(kb, agent_factory=factory))
    result = http.post("/api/chat", json={"question": "lost device"})
    assert result.status_code == 200
    assert result.json()["citations"][0]["id"] == "HR-310-1"


def test_missing_credentials_are_explicit(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    http = TestClient(create_app(kb))
    health = http.get("/api/health").json()
    assert health["status"] == "unconfigured"
    assert health["agloom_features"] == [
        "react",
        "local_tools",
        "capability_routing",
        "grounded_citations",
        "conversation_context",
    ]
    assert health["observability"]["enabled"] is False
    governance = http.get("/api/governance").json()
    assert governance["evidence_policy"] == "exact_passage_citations"
    assert governance["capability_scope"] == ["search_knowledge", "read_passage"]
    result = http.post("/api/chat", json={"question": "lost device"})
    assert result.status_code == 503
    assert result.json()["detail"] == "EXPLABS_API_KEY is required to answer questions"
