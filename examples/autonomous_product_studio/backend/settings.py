"""Process configuration read from the environment.

Credentials (database password, model key, Langfuse keys) are read where they
are used and never stored on this object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAYWRIGHT_EXECUTABLE = Path("/usr/local/bin/agloom-chromium")
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.casefold() in _TRUE_VALUES


@dataclass(frozen=True)
class StudioSettings:
    database_url: str
    projects_root: Path = APP_ROOT / "workspace" / "projects"
    preview_url: str = "http://target-preview"
    validation_executor_url: str | None = None
    browser_validation: bool = True
    playwright_executable: str | None = None
    playwright_browser: str | None = None
    policy_file: Path | None = None
    policy_schema: str = "agentbarrier"
    require_separate_approver: bool = False
    worker_concurrency: int = 2
    worker_poll_interval: float = 5.0
    shutdown_grace_seconds: float = 30.0
    embedded_worker: bool = False
    model: str = "gpt-5.6-luna"
    model_base_url: str = "https://api.experientiallabs.ai/v1"

    @classmethod
    def from_env(cls) -> StudioSettings:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is required")
        executable = os.getenv("PLAYWRIGHT_EXECUTABLE_PATH")
        if executable is None and DEFAULT_PLAYWRIGHT_EXECUTABLE.exists():
            executable = str(DEFAULT_PLAYWRIGHT_EXECUTABLE)
        policy_file = os.getenv("STUDIO_POLICY_FILE")
        return cls(
            database_url=database_url,
            projects_root=Path(
                os.getenv(
                    "STUDIO_PROJECTS_ROOT", str(APP_ROOT / "workspace" / "projects")
                )
            ),
            preview_url=os.getenv("TARGET_PREVIEW_URL", "http://target-preview").rstrip(
                "/"
            ),
            validation_executor_url=os.getenv("VALIDATION_EXECUTOR_URL") or None,
            browser_validation=_flag("STUDIO_BROWSER_VALIDATION", True),
            playwright_executable=executable or None,
            playwright_browser=os.getenv("PLAYWRIGHT_BROWSER") or None,
            policy_file=Path(policy_file) if policy_file else None,
            policy_schema=os.getenv("STUDIO_POLICY_SCHEMA", "agentbarrier"),
            require_separate_approver=_flag("STUDIO_REQUIRE_SEPARATE_APPROVER", False),
            worker_concurrency=int(os.getenv("STUDIO_WORKER_CONCURRENCY", "2")),
            worker_poll_interval=float(os.getenv("STUDIO_WORKER_POLL_SECONDS", "5")),
            shutdown_grace_seconds=float(
                os.getenv("STUDIO_SHUTDOWN_GRACE_SECONDS", "30")
            ),
            embedded_worker=_flag("STUDIO_EMBEDDED_WORKER", False),
            model=os.getenv("EXPLABS_MODEL", "gpt-5.6-luna"),
            model_base_url=os.getenv(
                "EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1"
            ),
        )
