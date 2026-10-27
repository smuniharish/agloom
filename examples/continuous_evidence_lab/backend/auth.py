"""OIDC bearer-token authentication and the per-request tenant principal."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

import jwt
from fastapi import HTTPException
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError

_ASYMMETRIC_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
        "EdDSA",
    }
)


@dataclass(frozen=True)
class Principal:
    """An authenticated user acting inside one tenant."""

    subject: str
    tenant_id: str
    claims: dict[str, Any] = field(default_factory=dict)


class Authenticator(Protocol):
    """Validates bearer tokens."""

    def authenticate(self, token: str) -> Principal: ...

    def verify_configuration(self) -> None: ...


class OIDCAuthenticator:
    """Validates OIDC access tokens against the issuer's JWKS."""

    def __init__(
        self,
        issuer: str | None = None,
        audience: str | None = None,
        jwks_uri: str | None = None,
        algorithms: tuple[str, ...] | None = None,
        tenant_claim: str | None = None,
    ) -> None:
        self.issuer = issuer or os.getenv("OIDC_ISSUER")
        self.audience = audience or os.getenv("OIDC_AUDIENCE")
        self.jwks_uri = jwks_uri or os.getenv("OIDC_JWKS_URI")
        configured = (
            algorithms
            if algorithms is not None
            else tuple(
                item.strip()
                for item in os.getenv("OIDC_ALGORITHMS", "RS256").split(",")
                if item.strip()
            )
        )
        if not configured or any(
            item not in _ASYMMETRIC_ALGORITHMS for item in configured
        ):
            raise ValueError(
                "OIDC_ALGORITHMS must contain supported asymmetric algorithms only"
            )
        self.algorithms = configured
        self.tenant_claim = tenant_claim or os.getenv("OIDC_TENANT_CLAIM", "tenant_id")
        try:
            self.clock_skew_seconds = int(os.getenv("OIDC_CLOCK_SKEW_SECONDS", "30"))
        except ValueError as exc:
            raise ValueError("OIDC_CLOCK_SKEW_SECONDS must be an integer") from exc
        if self.clock_skew_seconds < 0:
            raise ValueError("OIDC_CLOCK_SKEW_SECONDS must not be negative")
        self._jwks = (
            PyJWKClient(self.jwks_uri, cache_jwk_set=True, lifespan=300)
            if self.jwks_uri
            else None
        )

    def verify_configuration(self) -> None:
        missing = [
            name
            for name, value in (
                ("OIDC_ISSUER", self.issuer),
                ("OIDC_AUDIENCE", self.audience),
                ("OIDC_JWKS_URI", self.jwks_uri),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("OIDC is not configured; missing " + ", ".join(missing))

    def authenticate(self, token: str) -> Principal:
        self.verify_configuration()
        if self._jwks is None:
            raise RuntimeError("OIDC JWKS client is not configured")
        try:
            key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=list(self.algorithms),
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "sub", self.tenant_claim]},
                leeway=self.clock_skew_seconds,
            )
        except PyJWTError as exc:
            raise HTTPException(
                status_code=401,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        subject = claims.get("sub")
        tenant_id = claims.get(self.tenant_claim)
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(tenant_id, str)
            or not tenant_id
        ):
            raise HTTPException(401, "token is missing required identity claims")
        return Principal(subject=subject, tenant_id=tenant_id, claims=dict(claims))


class StaticAuthenticator:
    """Accepts one fixed token for one principal; for tests and local tooling."""

    def __init__(self, principal: Principal, token: str = "test-token") -> None:
        self.principal = principal
        self.token = token

    def verify_configuration(self) -> None:
        return None

    def authenticate(self, token: str) -> Principal:
        if token != self.token:
            raise HTTPException(401, "invalid bearer token")
        return self.principal


_principal: ContextVar[Principal | None] = ContextVar(
    "evidence_lab_principal", default=None
)


def current_principal() -> Principal:
    """Return the principal of the active request or job."""
    value = _principal.get()
    if value is None:
        raise RuntimeError("authenticated principal context is required")
    return value


@contextmanager
def principal_scope(principal: Principal) -> Iterator[Principal]:
    """Bind a principal to the current request or job."""
    token = _principal.set(principal)
    try:
        yield principal
    finally:
        _principal.reset(token)
