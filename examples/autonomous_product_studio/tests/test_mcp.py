from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp_capability_router import RefreshPolicy

from backend import mcp


def test_installed_playwright_server_runs_without_npm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mcp, "which", lambda binary: f"/usr/local/bin/{binary}")
    server = mcp.playwright_server(
        output_dir=tmp_path, executable_path="/usr/local/bin/agloom-chromium"
    )["mcpServers"]["playwright"]

    assert server["command"] == "/usr/local/bin/playwright-mcp"
    assert server["args"] == [
        "--headless",
        "--isolated",
        "--output-dir",
        str(tmp_path),
        "--no-sandbox",
        "--executable-path",
        "/usr/local/bin/agloom-chromium",
    ]
    assert server["transport"] == "stdio"
    assert server["keep_alive"] is False


def test_missing_playwright_server_uses_the_pinned_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mcp, "which", lambda binary: None)
    server = mcp.playwright_server(output_dir=tmp_path, browser="msedge")["mcpServers"][
        "playwright"
    ]

    assert server["command"] == "npx"
    assert server["args"] == [
        "-y",
        "@playwright/mcp@0.0.82",
        "--headless",
        "--isolated",
        "--output-dir",
        str(tmp_path),
        "--browser",
        "msedge",
    ]
    assert server["keep_alive"] is False


def test_capability_ids_follow_the_router_convention() -> None:
    assert mcp.RUN_CODE_CAPABILITY == "playwright:tool:browser_run_code_unsafe"


@pytest.mark.anyio
async def test_browser_validation_can_be_disabled() -> None:
    async with mcp.open_mcp_runtime(None) as runtime:
        assert runtime is None


@pytest.mark.anyio
async def test_the_server_is_registered_with_discovery_on_register(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registered: list[tuple[str, dict[str, Any], Any]] = []

    class RecordingRuntime:
        def __init__(self, **options: Any) -> None:
            self.options = options

        async def __aenter__(self) -> RecordingRuntime:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def register_mcp(
            self, server_id: str, target: dict[str, Any], *, refresh: Any = None
        ) -> None:
            registered.append((server_id, target, refresh))

    monkeypatch.setattr(mcp, "MCPRuntime", RecordingRuntime)
    server = mcp.playwright_server(output_dir=tmp_path, browser="msedge")
    async with mcp.open_mcp_runtime(server) as runtime:
        assert isinstance(runtime, RecordingRuntime)

    # Without a refresh policy the router discovers no capabilities at all.
    [(server_id, target, refresh)] = registered
    assert server_id == mcp.PLAYWRIGHT_SERVER_ID
    assert target == server
    assert isinstance(refresh, RefreshPolicy)
    assert refresh.on_register is True
