"""HTTP API: health, readiness, authentication, and tenant isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.auth import OIDCAuthenticator, Principal, StaticAuthenticator
from backend.store import MemoryStudioStore
from tests.conftest import AUTHORIZATION, TEST_PRINCIPAL
from tests.support import memory_services, scripted

MISSING = "00000000-0000-0000-0000-000000000000"
PROJECT = {
    "name": "Private",
    "objective": "Tenant private project",
    "acceptance_criteria": [
        {
            "id": "AC-1",
            "description": "Private criterion",
            "kind": "file",
            "target": "x.txt",
        }
    ],
}


class TokenMapAuthenticator:
    def verify_configuration(self) -> None:
        return None

    def authenticate(self, token: str) -> Principal:
        if token not in {"tenant-a", "tenant-b"}:
            raise HTTPException(401, "invalid bearer token")
        return Principal("user-1", token)


def test_health_ready_and_missing_resources(tmp_path: Path) -> None:
    app = create_app(
        memory_services(tmp_path, scripted()), StaticAuthenticator(TEST_PRINCIPAL)
    )
    with TestClient(app, headers=AUTHORIZATION) as client:
        health = client.get("/api/health", headers={"Authorization": ""}).json()
        ready = client.get("/api/ready", headers={"Authorization": ""})
        assert client.get(f"/api/projects/{MISSING}").status_code == 404
        assert client.get(f"/api/runs/{MISSING}").status_code == 404
        assert client.get(f"/api/runs/{MISSING}/events").status_code == 404
        assert client.get(f"/api/runs/{MISSING}/approvals").status_code == 404
        assert client.post(f"/api/runs/{MISSING}/cancel").status_code == 404
        assert (
            client.post(
                f"/api/projects/{MISSING}/tasks",
                json={"title": "Task", "objective": "Missing project"},
            ).status_code
            == 404
        )

    assert ready.json() == {"status": "ready"}
    assert health["policy"] == {
        "namespace": "autonomous-product-studio",
        "version": "studio-v1",
    }
    assert health["mcp_servers"] == []
    assert health["observability"]["enabled"] is False


def test_readiness_reports_an_unavailable_database_without_details(
    tmp_path: Path,
) -> None:
    class UnavailableStore(MemoryStudioStore):
        def ping(self) -> None:
            raise OSError("connection refused by 10.0.0.5:5432")

    app = create_app(
        memory_services(tmp_path, scripted(), store=UnavailableStore()),
        StaticAuthenticator(TEST_PRINCIPAL),
    )
    with TestClient(app) as client:
        response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}


def test_projects_are_seeded_once_per_tenant(tmp_path: Path) -> None:
    app = create_app(
        memory_services(tmp_path, scripted()), StaticAuthenticator(TEST_PRINCIPAL)
    )
    with TestClient(app, headers=AUTHORIZATION) as client:
        first = client.get("/api/projects").json()
        second = client.get("/api/projects").json()

    assert [project["name"] for project in first] == ["Pulseboard"]
    assert second == first
    assert len(first[0]["acceptance_criteria"]) == 4


def test_missing_invalid_and_cross_tenant_ids_are_hidden(tmp_path: Path) -> None:
    app = create_app(
        memory_services(tmp_path, scripted(), embedded_worker=False),
        TokenMapAuthenticator(),
    )
    tenant_a = {"Authorization": "Bearer tenant-a"}
    tenant_b = {"Authorization": "Bearer tenant-b"}
    with TestClient(app) as client:
        assert client.get("/api/projects").status_code == 401
        assert (
            client.get(
                "/api/projects", headers={"Authorization": "Bearer nope"}
            ).status_code
            == 401
        )
        project = client.post("/api/projects", headers=tenant_a, json=PROJECT).json()
        task = client.post(
            f"/api/projects/{project['id']}/tasks",
            headers=tenant_a,
            json={"title": "Private", "objective": "Private run objective"},
        ).json()
        run = client.post(f"/api/tasks/{task['id']}/run", headers=tenant_a).json()
        assert (
            client.get(f"/api/projects/{project['id']}", headers=tenant_b).status_code
            == 404
        )
        assert all(
            item["id"] != project["id"]
            for item in client.get("/api/projects", headers=tenant_b).json()
        )
        assert (
            client.post(f"/api/tasks/{task['id']}/run", headers=tenant_b).status_code
            == 404
        )
        for path in ("", "/events", "/evidence", "/approvals"):
            assert (
                client.get(f"/api/runs/{run['id']}{path}", headers=tenant_b).status_code
                == 404
            )
        assert (
            client.post(f"/api/runs/{run['id']}/cancel", headers=tenant_b).status_code
            == 404
        )
        assert (
            client.post(
                f"/api/runs/{run['id']}/approvals/any",
                headers=tenant_b,
                json={"decision": "approve"},
            ).status_code
            == 404
        )
        assert (
            client.get(f"/api/runs/{run['id']}", headers=tenant_a).json()["status"]
            == "queued"
        )


def _token(private_key: Any, now: datetime, **changes: Any) -> str:
    claims = {
        "sub": "user",
        "tenant_id": "tenant",
        "iss": "https://issuer.example",
        "aud": "studio",
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(minutes=5),
        **changes,
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


def test_oidc_rejects_wrong_issuer_audience_and_algorithm() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(UTC)
    auth = OIDCAuthenticator(
        issuer="https://issuer.example",
        audience="studio",
        jwks_uri="https://issuer.example/jwks",
    )
    auth._jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda _: SimpleNamespace(key=private_key.public_key())
    )

    principal = auth.authenticate(_token(private_key, now))
    assert (principal.subject, principal.tenant_id) == ("user", "tenant")
    for encoded in (
        _token(private_key, now, iss="https://wrong"),
        _token(private_key, now, aud="wrong"),
        _token(private_key, now, exp=now - timedelta(minutes=5)),
        _token(private_key, now, tenant_id=""),
    ):
        with pytest.raises(HTTPException) as rejected:
            auth.authenticate(encoded)
        assert rejected.value.status_code == 401
    symmetric = jwt.encode(
        {
            "sub": "user",
            "tenant_id": "tenant",
            "iss": "https://issuer.example",
            "aud": "studio",
            "exp": now + timedelta(minutes=5),
        },
        "not-a-production-secret-used-only-for-test",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as rejected:
        auth.authenticate(symmetric)
    assert rejected.value.status_code == 401


@pytest.mark.parametrize("algorithm", [(), ("HS256",), ("rs256",), ("none",)])
def test_oidc_rejects_unsafe_or_unknown_algorithm_configuration(
    algorithm: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="asymmetric"):
        OIDCAuthenticator(
            issuer="https://issuer.example",
            audience="studio",
            jwks_uri="https://issuer.example/jwks",
            algorithms=algorithm,
        )


def test_oidc_rejects_negative_clock_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OIDC_CLOCK_SKEW_SECONDS", "-1")
    with pytest.raises(ValueError, match="must not be negative"):
        OIDCAuthenticator(
            issuer="https://issuer.example",
            audience="studio",
            jwks_uri="https://issuer.example/jwks",
        )


def test_oidc_configuration_is_required_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URI"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(memory_services(tmp_path, scripted()), OIDCAuthenticator())
    with pytest.raises(RuntimeError, match="OIDC is not configured"), TestClient(app):
        pass
