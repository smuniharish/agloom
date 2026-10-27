"""Environment-driven settings shared by the API, the worker, and the setup command."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
POLICY_FILE = Path(__file__).resolve().with_name("policy.json")
_TRUE = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment variable."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().casefold() in _TRUE


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw and raw.strip() else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Model credentials are read only by the model factory."""

    database_url: str | None
    queue: str = "evidence"
    worker_concurrency: int = 2
    worker_in_api: bool = False
    max_attempts: int = 3
    retry_backoff_seconds: int = 5
    stalled_after_seconds: float = 60.0
    policy_file: Path = POLICY_FILE
    barrier_schema: str = "agentbarrier"
    require_separate_approver: bool = False
    frontend_origin: str = "http://localhost:5180"

    @classmethod
    def from_env(cls) -> Settings:
        """Read the settings from environment variables."""
        return cls(
            database_url=os.getenv("DATABASE_URL") or None,
            queue=os.getenv("EVIDENCE_QUEUE", "evidence"),
            worker_concurrency=_positive_int("WORKER_CONCURRENCY", 2),
            worker_in_api=env_flag("WORKER_IN_API"),
            max_attempts=_positive_int("JOB_MAX_ATTEMPTS", 3),
            retry_backoff_seconds=_positive_int("JOB_RETRY_BACKOFF_SECONDS", 5),
            stalled_after_seconds=float(_positive_int("STALLED_JOB_SECONDS", 60)),
            policy_file=Path(os.getenv("EVIDENCE_POLICY_FILE") or POLICY_FILE),
            barrier_schema=os.getenv("AGENTBARRIER_SCHEMA", "agentbarrier"),
            require_separate_approver=env_flag("REQUIRE_SEPARATE_APPROVER"),
            frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5180"),
        )

    def require_database(self) -> str:
        """Return the database URL or fail with a clear error."""
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required")
        return self.database_url
