from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from _support import real_model
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agloom import create_agent


@tool
def local_uppercase(text: str) -> str:
    """Convert local text to uppercase."""

    return text.upper()


async def main() -> None:
    server = Path(__file__).with_name("_mcp_server.py")
    client = MultiServerMCPClient(
        {
            "demo": {
                "command": sys.executable,
                "args": [str(server)],
                "transport": "stdio",
            }
        }
    )
    agent = create_agent(
        model=real_model(),
        tools=[local_uppercase],
        pattern="react",
        capabilities={"application_status": lambda _: "ready"},
        mcp_client=client,
        mcp_server_name="demo",
        mcp_discover_resources=True,
    )

    readiness = await agent.ainvoke(
        "Reply in one sentence containing: MCP capability catalog is ready."
    )
    assert "MCP capability catalog is ready" in str(readiness.content)
    selection = await agent.aroute_capabilities(
        "Use local uppercase, application status, MCP addition, and the guide."
    )
    tool_result = await agent.aexecute_capability(
        "mcp:demo:tool:add",
        {"left": 17, "right": 23},
    )
    resource_result = await agent.aexecute_capability(
        "mcp:demo:resource:agloom://guide"
    )
    prompt = await client.get_prompt(
        "demo",
        "summarize",
        arguments={"topic": "capability routing"},
    )
    print("Selected:", selection.names)
    print("MCP tool:", tool_result)
    print("MCP resource:", resource_result)
    print("MCP prompt:", prompt)
    assert "40" in str(tool_result)
    assert "Agloom routes local" in str(resource_result)
    assert "capability routing" in str(prompt)


asyncio.run(main())
