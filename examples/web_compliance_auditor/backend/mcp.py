"""Real stdio MCP configuration; servers are installed by npm at runtime."""

from __future__ import annotations

from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


def make_mcp_client() -> MultiServerMCPClient:
    workspace = str((Path(__file__).resolve().parent.parent / "workspace").resolve())
    return MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": [
                    "-y",
                    "@playwright/mcp@0.0.82",
                    "--headless",
                    "--isolated",
                    "--no-sandbox",
                    "--executable-path",
                    "/usr/local/bin/agloom-chromium",
                ],
                "transport": "stdio",
            },
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem@2026.8.31",
                    workspace,
                ],
                "transport": "stdio",
            },
            "everything": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-everything@2026.8.31",
                    "stdio",
                ],
                "transport": "stdio",
            },
        }
    )
