from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from backend.auth import (
    OIDCAuthenticator,
    Principal,
    current_principal,
    principal_scope,
)
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

ISSUER = "https://issuer.test/realms/agloom"
AUDIENCE = "evidence-lab"


@pytest.fixture
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def authenticator(signing_key, monkeypatch) -> OIDCAuthenticator:
    oidc = OIDCAuthenticator(
        issuer=ISSUER, audience=AUDIENCE, jwks_uri="https://issuer.test/certs"
    )
    monkeypatch.setattr(
        oidc._jwks,
        "get_signing_key_from_jwt",
        lambda token: SimpleNamespace(key=signing_key.public_key()),
    )
    return oidc


def token(signing_key, **claims) -> str:
    now = datetime.now(UTC)
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "alice",
        "tenant_id": "acme",
        "exp": now + timedelta(minutes=5),
        **claims,
    }
    return jwt.encode(
        {key: value for key, value in payload.items() if value is not None},
        signing_key,
        algorithm="RS256",
    )


def test_valid_tokens_yield_the_tenant_principal(signing_key, monkeypatch):
    principal = authenticator(signing_key, monkeypatch).authenticate(token(signing_key))
    assert (principal.subject, principal.tenant_id) == ("alice", "acme")


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "another-client"},
        {"iss": "https://attacker.test"},
        {"tenant_id": None},
        {"exp": datetime.now(UTC) - timedelta(minutes=5)},
    ],
)
def test_invalid_tokens_are_rejected(signing_key, monkeypatch, claims):
    with pytest.raises(HTTPException) as error:
        authenticator(signing_key, monkeypatch).authenticate(
            token(signing_key, **claims)
        )
    assert error.value.status_code == 401


def test_only_asymmetric_algorithms_are_accepted(monkeypatch):
    monkeypatch.setenv("OIDC_ALGORITHMS", "HS256")
    with pytest.raises(ValueError, match="asymmetric"):
        OIDCAuthenticator()


def test_missing_configuration_is_reported(monkeypatch):
    for name in ("OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URI"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="OIDC_ISSUER"):
        OIDCAuthenticator().verify_configuration()


def test_principal_scope_is_required_and_restored():
    outer = current_principal()
    with principal_scope(Principal("bob", "globex")):
        assert current_principal().tenant_id == "globex"
    assert current_principal() == outer
