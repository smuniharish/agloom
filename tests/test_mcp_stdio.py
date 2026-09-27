from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_capability_router import CapabilityType, MCPRuntime, RefreshPolicy

from agloom.capabilities.providers import MCPCapabilityProvider


@pytest.mark.asyncio
async def test_real_stdio_mcp_tool_resource_and_prompt() -> None:
    server = Path(__file__).parents[1] / "examples" / "_mcp_server.py"
    client = MultiServerMCPClient(
        {
            "demo": {
                "command": sys.executable,
                "args": [str(server)],
                "transport": "stdio",
            }
        }
    )
    runtime = MCPRuntime()
    await runtime.register_mcp_client(
        "demo",
        client,
        discover_resources=True,
        refresh=RefreshPolicy(on_register=True),
    )

    try:
        catalog = await runtime.registry.list(server_id="demo")
        provider = MCPCapabilityProvider(
            {"mcp_runtime": runtime},
            server_name="demo",
        )
        discovered = await provider.adiscover()
        by_type = {item.metadata["mcp_type"]: item for item in discovered}

        tool_result = await by_type[CapabilityType.TOOL].value.ainvoke(
            {"left": 17, "right": 23}
        )
        resource_result = await by_type[CapabilityType.RESOURCE].value.ainvoke()
        prompt_result = await client.get_prompt(
            "demo",
            "summarize",
            arguments={"topic": "capability routing"},
        )

        assert {item.type for item in catalog} == {
            CapabilityType.TOOL,
            CapabilityType.RESOURCE,
        }
        assert tool_result[0]["text"] == "40"
        assert "Agloom routes local" in resource_result
        assert (
            prompt_result[0].content == "Summarize capability routing in one sentence."
        )
    finally:
        await runtime.close()
