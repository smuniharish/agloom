"""Configuration read from environment variables."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPLICATION_ID = "engineering-change-investigator"
TENANT_ID = "demo-engineering"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class SettingsError(ValueError):
    """Raised when the environment does not describe a valid configuration."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-secret runtime configuration.

    Credentials (the model API key and Langfuse keys) are read from the
    environment where they are used and are never stored here.
    """

    database_url: str
    model: str
    model_base_url: str
    ollama_url: str
    embedding_model: str
    target_site_url: str
    workspace: Path
    mcp_launcher: str
    browser_executable: str | None
    browser_channel: str | None
    browser_sandbox: bool
    browser_output_dir: Path
    metrics_enabled: bool
    metrics_host: str
    metrics_port: int
    investigation_timeout: float
    approval_ttl_seconds: float

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read the configuration from `environ` (the process environment by default).

        Raises:
            SettingsError: If a required variable is missing or a value is invalid.
        """
        env = os.environ if environ is None else environ
        database_url = env.get("DATABASE_URL", "").strip()
        if not database_url:
            raise SettingsError("DATABASE_URL is required")
        launcher = env.get("MCP_SERVER_LAUNCHER", "npx").strip() or "npx"
        if launcher not in {"npx", "installed"}:
            raise SettingsError("MCP_SERVER_LAUNCHER must be 'npx' or 'installed'")
        target = env.get("TARGET_SITE_URL", "http://target-site").strip().rstrip("/")
        if not target.startswith(("http://", "https://")):
            raise SettingsError("TARGET_SITE_URL must be an http(s) URL")
        return cls(
            database_url=database_url,
            model=env.get("EXPLABS_MODEL", "gpt-5.6-luna"),
            model_base_url=env.get(
                "EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1"
            ),
            ollama_url=env.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            embedding_model=env.get("OLLAMA_EMBED_MODEL", "all-minilm"),
            target_site_url=target,
            workspace=Path(env.get("WORKSPACE_DIR", str(ROOT / "workspace"))).resolve(),
            mcp_launcher=launcher,
            browser_executable=env.get("BROWSER_EXECUTABLE") or None,
            browser_channel=env.get("BROWSER_CHANNEL") or None,
            browser_sandbox=_flag(env, "BROWSER_SANDBOX", default=True),
            browser_output_dir=Path(
                env.get(
                    "BROWSER_OUTPUT_DIR",
                    str(Path(tempfile.gettempdir()) / APPLICATION_ID / "browser"),
                )
            ),
            metrics_enabled=_flag(env, "METRICS_ENABLED", default=False),
            metrics_host=env.get("METRICS_HOST", "127.0.0.1"),
            metrics_port=_number(env, "METRICS_PORT", 9468, int),
            investigation_timeout=_number(env, "INVESTIGATION_TIMEOUT_SECONDS", 240.0),
            approval_ttl_seconds=_number(env, "APPROVAL_TTL_SECONDS", 86_400.0),
        )


def model_api_key(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the model API key from the environment, if it is configured."""
    env = os.environ if environ is None else environ
    return env.get("EXPLABS_API_KEY") or None


def _flag(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = env.get(name, "").strip().casefold()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise SettingsError(f"{name} must be a boolean")


def _number[T: (int, float)](
    env: Mapping[str, str], name: str, default: T, kind: type[T] = float
) -> T:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError as error:
        raise SettingsError(f"{name} must be a number") from error
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value
