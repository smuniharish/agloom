"""Live checks against the model configured by EXPLABS_* (run with ``-m live``).

The repository-root ``.env`` is loaded when present; variables already set in
the environment take precedence.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from backend.app import create_app
from backend.knowledge import KnowledgeBase
from dotenv import load_dotenv
from fastapi.testclient import TestClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_client() -> Iterator[TestClient]:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    if not os.getenv("EXPLABS_API_KEY"):
        pytest.skip("EXPLABS_API_KEY is not configured")
    with TestClient(create_app()) as client:
        yield client


def test_live_grounded_conversation(live_client: TestClient) -> None:
    passages = KnowledgeBase().passages
    first = live_client.post(
        "/api/chat", json={"question": "How do I report a lost company device?"}
    )
    assert first.status_code == 200, first.text
    answer = first.json()
    assert answer["citations"]
    assert {citation["id"] for citation in answer["citations"]} <= passages.keys()

    follow_up = live_client.post(
        "/api/chat",
        json={
            "question": "What happens after I report the lost device?",
            "conversation_id": answer["conversation_id"],
        },
    )
    assert follow_up.status_code == 200, follow_up.text
    assert follow_up.json()["citations"]
    history = live_client.get(f"/api/conversations/{answer['conversation_id']}")
    assert len(history.json()["messages"]) == 4
