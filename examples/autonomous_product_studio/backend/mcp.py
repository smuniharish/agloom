"""The Playwright MCP server, registered once per process on an `MCPRuntime`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from shutil import which
from typing import Any

from mcp_capability_router import MCPRuntime, RefreshPolicy, make_capability_id

PLAYWRIGHT_SERVER_ID = "playwright"
PLAYWRIGHT_PACKAGE = "@playwright/mcp@0.0.82"
RUN_CODE_CAPABILITY = make_capability_id(
    PLAYWRIGHT_SERVER_ID, "tool", "browser_run_code_unsafe"
)


def playwright_server(
    *,
    output_dir: Path,
    executable_path: str | None = None,
    browser: str | None = None,
) -> dict[str, Any]:
    """MCP configuration for a headless, isolated Playwright MCP server.

    The image-installed `playwright-mcp` binary is preferred; otherwise the pinned
    package runs through `npx`. Console logs and other output files go to
    `output_dir` instead of the process's working directory. `keep_alive` is
    off so that closing the `MCPRuntime` also stops the stdio server process
    (mcp-capability-router 0.2.0 otherwise leaves it running); the runtime
    still keeps one session open for its whole lifetime.
    """
    arguments = ["--headless", "--isolated", "--output-dir", str(output_dir)]
    if executable_path:
        arguments += ["--no-sandbox", "--executable-path", executable_path]
    elif browser:
        arguments += ["--browser", browser]
    installed = which("playwright-mcp")
    command, prefix = (
        (installed, []) if installed else ("npx", ["-y", PLAYWRIGHT_PACKAGE])
    )
    return {
        "mcpServers": {
            PLAYWRIGHT_SERVER_ID: {
                "command": command,
                "args": [*prefix, *arguments],
                "transport": "stdio",
                "keep_alive": False,
            }
        }
    }


@asynccontextmanager
async def open_mcp_runtime(
    server: Mapping[str, Any] | None,
) -> AsyncIterator[MCPRuntime | None]:
    """Own the process-wide MCP runtime; `None` when browser validation is off."""
    if server is None:
        yield None
        return
    async with MCPRuntime(max_concurrency=4, operation_timeout=60.0) as runtime:
        await runtime.register_mcp(
            PLAYWRIGHT_SERVER_ID, dict(server), refresh=RefreshPolicy(on_register=True)
        )
        yield runtime
