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


def test_live_grounded_support_conversation(live_client: TestClient) -> None:
    policy = live_client.post(
        "/api/chat",
        json={"message": "What is the return policy?", "customer_id": "CUST-1024"},
    )
    assert policy.status_code == 200, policy.text
    first = policy.json()
    assert [source["id"] for source in first["sources"]] == ["POL-RETURNS"]

    order = live_client.post(
        "/api/chat",
        json={
            "message": "Where is ORD-5001?",
            "customer_id": "CUST-1024",
            "conversation_id": first["conversation_id"],
        },
    )
    assert order.status_code == 200, order.text
    reply = order.json()
    assert reply["order"]["status"] == "in transit"
    assert "in transit" in reply["answer"].casefold()

    feedback = live_client.post(
        "/api/feedback", json={"message_id": reply["message_id"], "rating": "up"}
    )
    assert feedback.status_code == 200, feedback.text
