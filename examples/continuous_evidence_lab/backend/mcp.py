"""Real MCP servers: the bounded workspace, the reference server, and the
isolated browser, registered on the application's `MCPRuntime`."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from shutil import which
from typing import Any

from mcp_capability_router import MCPRuntime, RefreshPolicy

from .config import APP_ROOT

WORKSPACE = APP_ROOT / "workspace"
FILESYSTEM_SERVER = "evidence-filesystem"
EVERYTHING_SERVER = "evidence-everything"
BROWSER_SERVER = "evidence-browser"
FILESYSTEM_PACKAGE = "@modelcontextprotocol/server-filesystem@2026.8.31"
EVERYTHING_PACKAGE = "@modelcontextprotocol/server-everything@2026.8.31"
BRIDGE_CLIENT = Path(__file__).resolve().with_name("browser_bridge_client.py")


def capability(server_id: str, tool: str) -> str:
    """Return the capability and policy name of one MCP tool."""
    return f"{server_id}__{tool}"


def _stdio(
    command: str, args: list[str], env: dict[str, str] | None = None
) -> dict[str, Any]:
    # The runtime holds one session per server for its lifetime. Without
    # keep-alive, closing that session stops the subprocess; otherwise FastMCP
    # stops it only once the transport is garbage collected.
    server: dict[str, Any] = {
        "command": command,
        "args": args,
        "transport": "stdio",
        "keep_alive": False,
    }
    if env is not None:
        server["env"] = env
    return server


def _npm_server(binary: str, package: str, *args: str) -> dict[str, Any]:
    installed = which(binary)
    if installed:
        return _stdio(installed, list(args))
    return _stdio(which("npx") or "npx", ["-y", package, *args])


def mcp_servers() -> dict[str, dict[str, Any]]:
    """Return the stdio MCP servers of this deployment.

    The browser bridge is included only when `BROWSER_BRIDGE_TOKEN` is set; it
    receives nothing but its own connection settings.
    """
    servers = {
        FILESYSTEM_SERVER: _npm_server(
            "mcp-server-filesystem", FILESYSTEM_PACKAGE, str(WORKSPACE)
        ),
        EVERYTHING_SERVER: _npm_server(
            "mcp-server-everything", EVERYTHING_PACKAGE, "stdio"
        ),
    }
    token = os.getenv("BROWSER_BRIDGE_TOKEN")
    if token:
        servers[BROWSER_SERVER] = _stdio(
            sys.executable,
            [str(BRIDGE_CLIENT)],
            {
                "BROWSER_BRIDGE_HOST": os.getenv("BROWSER_BRIDGE_HOST", "browser"),
                "BROWSER_BRIDGE_PORT": os.getenv("BROWSER_BRIDGE_PORT", "8931"),
                "BROWSER_BRIDGE_TOKEN": token,
            },
        )
    return servers


async def register_servers(runtime: MCPRuntime, servers: Mapping[str, Any]) -> None:
    """Register each server and discover its capabilities before serving runs.

    `MCPRuntime.register_mcp` discovers nothing by itself; without
    `RefreshPolicy(on_register=True)` the agent would see no MCP tools.

    A mapping value is a stdio server configuration; any other value is passed
    to `MCPRuntime.register_mcp` as is (a URL, a FastMCP server, a client).
    """
    for server_id, target in servers.items():
        await runtime.register_mcp(
            server_id,
            {"mcpServers": {server_id: dict(target)}}
            if isinstance(target, Mapping)
            else target,
            refresh=RefreshPolicy(on_register=True),
        )
