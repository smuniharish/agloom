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
from backend.store import IncidentStore
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


def test_live_incident_analysis(live_client: TestClient) -> None:
    incident = IncidentStore().get("INC-2026-041")
    assert incident is not None
    response = live_client.post("/api/analyze", json={"incident_id": incident.id})
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["incident_id"] == incident.id
    assert 0 <= report["confidence"] <= 1
    assert report["citations"]
    known = {record.id for record in incident.records}
    assert {citation["id"] for citation in report["citations"]} <= known
