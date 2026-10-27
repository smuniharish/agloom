"""MCP: tools, resources, and prompts from MCP servers through an app-owned runtime.

The application owns an `mcp_capability_router.MCPRuntime`, registers its MCP
servers, and passes the runtime to `create_agent(mcp=...)`. Register with
`refresh=RefreshPolicy(on_register=True)` (or call `await runtime.refresh()`):
registration alone discovers nothing, and the agent would see no MCP
capabilities. Here the example server runs as a separate process over stdio,
from a standard MCP server configuration; `"keep_alive": False` makes the
server process end when the runtime closes. The model calls MCP tools as
`<server>__<tool>`, and the application reads resources and renders prompts
with `aexecute_capability`. Agents with MCP use the asynchronous APIs.

Run:
    uv run python examples/13_mcp.py
"""

import asyncio
import sys
from pathlib import Path

from _support import real_model, verified_text
from mcp_capability_router import MCPRuntime, RefreshPolicy

from agloom import create_agent

OPS_SERVER = {
    "mcpServers": {
        "ops": {
            "command": sys.executable,
            "args": [str(Path(__file__).with_name("_mcp_server.py"))],
            "keep_alive": False,
        }
    }
}


async def main() -> None:
    async with MCPRuntime() as runtime:
        await runtime.register_mcp(
            "ops", OPS_SERVER, refresh=RefreshPolicy(on_register=True)
        )
        agent = create_agent(model=real_model(), mcp=runtime, pattern="react")

        selection = await agent.aroute_capabilities("Check the billing service.")
        assert "ops__service_status" in selection.tool_names, selection.names
        print("MCP capabilities:", ", ".join(selection.names))

        answer = await agent.ainvoke(
            "Use the ops tools to report the current health of the billing service."
        )
        print(verified_text(answer, "2.4"))

        runbook = await agent.aexecute_capability(
            "mcp:ops:resource:ops://runbooks/billing"
        )
        prompt = await agent.aexecute_capability(
            "mcp:ops:prompt:incident_summary", {"service": "billing"}
        )
        print("Runbook:", verified_text(runbook[0].text, "payment gateway"))
        print("Prompt:", verified_text(prompt.messages[0].content.text, "billing"))


if __name__ == "__main__":
    asyncio.run(main())
