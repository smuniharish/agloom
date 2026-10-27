"""The MCP servers the auditor registers on its `MCPRuntime`.

Every server runs over stdio. Locally they are started with `npx` at the pinned
versions; container images install the same versions and set
`MCP_SERVER_LAUNCHER=installed` to run them without downloads.
"""

from __future__ import annotations

import shutil
from typing import Any

from .settings import Settings

FILESYSTEM = "web-compliance-filesystem"
EVERYTHING = "web-compliance-everything"
PLAYWRIGHT = "web-compliance-playwright"

# server ID -> (npm package, pinned version, installed executable)
PACKAGES: dict[str, tuple[str, str, str]] = {
    FILESYSTEM: (
        "@modelcontextprotocol/server-filesystem",
        "2026.8.31",
        "mcp-server-filesystem",
    ),
    EVERYTHING: (
        "@modelcontextprotocol/server-everything",
        "2026.8.31",
        "mcp-server-everything",
    ),
    PLAYWRIGHT: ("@playwright/mcp", "0.0.83", "playwright-mcp"),
}


def server_targets(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return the FastMCP configuration of every server, keyed by server ID."""
    return {
        PLAYWRIGHT: _stdio(settings, PLAYWRIGHT, browser_arguments(settings)),
        FILESYSTEM: _stdio(settings, FILESYSTEM, [str(settings.workspace)]),
        EVERYTHING: _stdio(settings, EVERYTHING, ["stdio"]),
    }


def browser_arguments(settings: Settings) -> list[str]:
    """Return the Playwright MCP arguments: headless, isolated, and read-mostly.

    Audited pages are untrusted, so tools that a page registers through WebMCP
    and image responses are disabled, and output files go to a dedicated
    directory instead of the working directory.
    """
    arguments = [
        "--headless",
        "--isolated",
        "--no-webmcp",
        "--image-responses",
        "omit",
        "--output-dir",
        str(settings.browser_output_dir),
    ]
    if settings.browser_executable:
        arguments += ["--executable-path", settings.browser_executable]
    elif settings.browser_channel:
        arguments += ["--browser", settings.browser_channel]
    if not settings.browser_sandbox:
        arguments.append("--no-sandbox")
    return arguments


def _stdio(settings: Settings, server_id: str, arguments: list[str]) -> dict[str, Any]:
    package, version, executable = PACKAGES[server_id]
    if settings.mcp_launcher == "installed":
        command, args = shutil.which(executable) or executable, arguments
    else:
        command = shutil.which("npx") or "npx"
        args = ["-y", f"{package}@{version}", *arguments]
    # The session stays open while the runtime runs; keep_alive=False stops the
    # server process when MCPRuntime.close() ends that session.
    server = {"command": command, "args": args, "keep_alive": False}
    return {"mcpServers": {server_id: server}}
