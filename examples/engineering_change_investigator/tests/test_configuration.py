"""Configuration, MCP server launch targets, and report grounding."""

from __future__ import annotations

from pathlib import Path

import pytest
from backend.mcp_servers import (
    EVERYTHING,
    FILESYSTEM,
    PACKAGES,
    PLAYWRIGHT,
    browser_arguments,
    server_targets,
)
from backend.models import EvidenceItem, InvestigationReport
from backend.settings import Settings, SettingsError

BASE = {"DATABASE_URL": "postgresql://db.invalid/engineering"}


def settings(**overrides: str) -> Settings:
    return Settings.from_environment({**BASE, **overrides})


def test_defaults_are_safe_for_local_development() -> None:
    configured = settings()

    assert configured.target_site_url == "http://target-site"
    assert configured.mcp_launcher == "npx"
    assert configured.browser_sandbox is True
    assert configured.metrics_enabled is False
    assert configured.metrics_port == 9468
    assert configured.workspace.name == "workspace"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"DATABASE_URL": ""}, "DATABASE_URL is required"),
        ({"MCP_SERVER_LAUNCHER": "docker"}, "MCP_SERVER_LAUNCHER"),
        ({"TARGET_SITE_URL": "file:///etc/passwd"}, "TARGET_SITE_URL"),
        ({"METRICS_ENABLED": "maybe"}, "METRICS_ENABLED must be a boolean"),
        ({"METRICS_PORT": "http"}, "METRICS_PORT must be a number"),
        ({"INVESTIGATION_TIMEOUT_SECONDS": "-1"}, "must be positive"),
    ],
)
def test_invalid_configuration_is_rejected(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(SettingsError, match=message):
        Settings.from_environment({**BASE, **environment})


def test_npx_launches_pinned_package_versions(tmp_path: Path) -> None:
    configured = settings(WORKSPACE_DIR=str(tmp_path))
    targets = server_targets(configured)

    for server_id, (package, version, _) in PACKAGES.items():
        server = targets[server_id]["mcpServers"][server_id]
        assert Path(server["command"]).stem.casefold() == "npx"
        assert server["args"][:2] == ["-y", f"{package}@{version}"]
        assert server["keep_alive"] is False
    assert targets[FILESYSTEM]["mcpServers"][FILESYSTEM]["args"][-1] == str(
        tmp_path.resolve()
    )
    assert targets[EVERYTHING]["mcpServers"][EVERYTHING]["args"][-1] == "stdio"


def test_installed_servers_run_without_downloads() -> None:
    targets = server_targets(settings(MCP_SERVER_LAUNCHER="installed"))

    command = targets[PLAYWRIGHT]["mcpServers"][PLAYWRIGHT]["command"]
    assert Path(command).stem == "playwright-mcp"
    assert "-y" not in targets[PLAYWRIGHT]["mcpServers"][PLAYWRIGHT]["args"]
    assert targets[PLAYWRIGHT]["mcpServers"][PLAYWRIGHT]["keep_alive"] is False


def test_browser_is_headless_isolated_and_writes_only_to_its_output_dir(
    tmp_path: Path,
) -> None:
    arguments = browser_arguments(
        settings(
            BROWSER_EXECUTABLE="/usr/local/bin/chromium",
            BROWSER_SANDBOX="false",
            BROWSER_OUTPUT_DIR=str(tmp_path),
        )
    )

    assert arguments[:5] == [
        "--headless",
        "--isolated",
        "--no-webmcp",
        "--image-responses",
        "omit",
    ]
    assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path)
    assert arguments[arguments.index("--executable-path") + 1] == (
        "/usr/local/bin/chromium"
    )
    assert arguments[-1] == "--no-sandbox"
    assert "--browser" in browser_arguments(settings(BROWSER_CHANNEL="msedge"))


def build_report(*citations: str) -> InvestigationReport:
    return InvestigationReport(
        decision="no_go",
        risk="high",
        summary="Not ready.",
        evidence=[
            EvidenceItem(claim="Missing approval.", citation=citation)
            for citation in citations
        ],
        recommendation="Do not deploy.",
    )


def test_grounding_requires_one_allowed_bracketed_id_per_claim() -> None:
    allowed = {"RELEASE-301", "ADR-101"}

    assert build_report("[RELEASE-301]", "[ADR-101]").grounded(allowed)
    assert build_report("[RELEASE-301]").citations() == ["RELEASE-301"]
    assert not build_report("[release]").grounded(allowed)
    assert not build_report("[RELEASE-999]").grounded(allowed)
    assert not build_report("RELEASE-301").grounded(allowed)
    assert not build_report("[RELEASE-301] [ADR-101]").grounded(allowed)
