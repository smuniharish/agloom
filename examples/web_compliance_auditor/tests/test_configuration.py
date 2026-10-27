"""Configuration, MCP server launch targets, policy sources, and grounding."""

from __future__ import annotations

import json
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
from backend.models import AuditReport, Finding
from backend.policies import IndexOperation, PolicyFileSource
from backend.settings import Settings, SettingsError
from refresh_engine import PlanAction, RefreshRequest

from .doubles import MemoryPolicies

BASE = {"DATABASE_URL": "postgresql://db.invalid/compliance"}


def settings(**overrides: str) -> Settings:
    return Settings.from_environment({**BASE, **overrides})


def test_defaults_are_safe_for_local_development() -> None:
    configured = settings()

    assert configured.mcp_launcher == "npx"
    assert configured.browser_sandbox is True
    assert configured.metrics_enabled is False
    assert configured.metrics_port == 9467
    assert configured.policies_file.name == "policies.json"
    assert configured.workspace.name == "workspace"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"DATABASE_URL": " "}, "DATABASE_URL is required"),
        ({"MCP_SERVER_LAUNCHER": "docker"}, "MCP_SERVER_LAUNCHER"),
        ({"BROWSER_SANDBOX": "sometimes"}, "BROWSER_SANDBOX must be a boolean"),
        ({"AUDIT_TIMEOUT_SECONDS": "soon"}, "must be a number"),
        ({"APPROVAL_TTL_SECONDS": "0"}, "must be positive"),
    ],
)
def test_invalid_configuration_is_rejected(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(SettingsError, match=message):
        Settings.from_environment({**BASE, **environment})


def test_npx_launches_pinned_package_versions() -> None:
    targets = server_targets(settings())

    assert list(targets) == [PLAYWRIGHT, FILESYSTEM, EVERYTHING]
    for server_id, (package, version, _) in PACKAGES.items():
        server = targets[server_id]["mcpServers"][server_id]
        assert Path(server["command"]).stem.casefold() == "npx"
        assert server["args"][:2] == ["-y", f"{package}@{version}"]
        assert server["keep_alive"] is False


def test_installed_servers_run_without_downloads() -> None:
    targets = server_targets(settings(MCP_SERVER_LAUNCHER="installed"))

    server = targets[FILESYSTEM]["mcpServers"][FILESYSTEM]
    assert Path(server["command"]).stem == "mcp-server-filesystem"
    assert server["args"] == [str(settings().workspace)]
    assert server["keep_alive"] is False


def test_browser_is_headless_isolated_and_ignores_page_tools(tmp_path: Path) -> None:
    arguments = browser_arguments(
        settings(BROWSER_CHANNEL="msedge", BROWSER_OUTPUT_DIR=str(tmp_path))
    )

    assert "--no-webmcp" in arguments
    assert arguments[arguments.index("--image-responses") + 1] == "omit"
    assert arguments[arguments.index("--browser") + 1] == "msedge"
    assert arguments[arguments.index("--output-dir") + 1] == str(tmp_path)
    assert "--no-sandbox" not in arguments


async def test_policy_file_source_versions_policies_by_content(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(
        json.dumps([{"id": "P-1", "title": "Labels", "body": "Inputs need labels."}]),
        encoding="utf-8",
    )
    source = PolicyFileSource(path)

    discovery = await source.discover()
    [resource] = [item async for item in discovery.resources]
    snapshot = await source.snapshot(resource)
    path.write_text(
        json.dumps([{"id": "P-1", "title": "Labels", "body": "Changed."}]),
        encoding="utf-8",
    )
    [changed] = [item async for item in (await source.discover()).resources]

    assert discovery.complete is True
    assert snapshot.content == "Inputs need labels."
    assert snapshot.metadata["title"] == "Labels"
    assert changed.version != resource.version


@pytest.mark.parametrize(
    "entries",
    [
        [
            {"id": "P-1", "title": "A", "body": "x"},
            {"id": "P-1", "title": "B", "body": "y"},
        ],
        [{"id": "P-1", "title": "A", "body": " "}],
    ],
)
async def test_invalid_policy_files_fail_discovery(
    tmp_path: Path, entries: list[dict[str, str]]
) -> None:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(entries), encoding="utf-8")

    with pytest.raises(ValueError, match="policy"):
        await PolicyFileSource(path).discover()


async def test_index_operation_applies_refresh_and_delete_plans(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(
        json.dumps([{"id": "P-1", "title": "Labels", "body": "Inputs need labels."}]),
        encoding="utf-8",
    )
    source = PolicyFileSource(path)
    [resource] = [item async for item in (await source.discover()).resources]
    index = MemoryPolicies()
    operation = IndexOperation(index)
    request = RefreshRequest()

    await operation(
        resource, await source.snapshot(resource), PlanAction.REFRESH, request
    )
    assert index.policies == {"P-1": ("Labels", "Inputs need labels.")}
    await operation(resource, None, PlanAction.DELETE, request)
    assert index.policies == {}


def test_grounding_requires_retrieved_policy_ids() -> None:
    def build(policy_id: str, evidence: list[str]) -> AuditReport:
        return AuditReport(
            verdict="fail",
            summary="Labels are missing.",
            findings=[
                Finding(
                    policy_id=policy_id,
                    requirement="Labels",
                    status="fail",
                    rationale="No label.",
                    evidence_ids=evidence,
                )
            ],
        )

    allowed = {"WEB-ACCESS-001"}
    assert build("WEB-ACCESS-001", ["WEB-ACCESS-001"]).grounded(allowed)
    assert not build("UNKNOWN-001", ["WEB-ACCESS-001"]).grounded(allowed)
    assert not build("WEB-ACCESS-001", ["UNKNOWN-001"]).grounded(allowed)
