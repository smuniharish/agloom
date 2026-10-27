"""MCP capabilities through mcp-capability-router and an in-process FastMCP server."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from agentbarrier.runtime import (
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from fastmcp import FastMCP
from mcp_capability_router import MCPRuntime, RefreshPolicy

from agloom import (
    CapabilityKind,
    CapabilityResolutionError,
    ConfigurationError,
    PolicyManager,
    create_agent,
)
from agloom.capabilities.providers import MCPCapability, mcp_descriptors
from helpers import Call, ScriptedChatModel, tool_call

restarts: list[str] = []


def build_server() -> FastMCP:
    server = FastMCP("ops")

    @server.tool
    def service_status(service: str) -> str:
        """Return the status of a service."""
        return f"{service} is healthy"

    @server.tool
    def restart(service: str) -> str:
        """Restart a service."""
        restarts.append(service)
        return f"{service} restarted"

    @server.tool
    def wipe_disk() -> str:
        """Erase every disk."""
        return "never"

    @server.resource("docs://runbook")
    def runbook() -> str:
        """The incident runbook."""
        return "Restart, then verify health."

    @server.prompt
    def triage(service: str) -> str:
        """Triage a service."""
        return f"Triage {service} now."

    return server


@pytest.fixture
async def runtime() -> AsyncIterator[MCPRuntime]:
    restarts.clear()
    async with MCPRuntime() as mcp:
        await mcp.register_mcp(
            "ops", build_server(), refresh=RefreshPolicy(on_register=True)
        )
        yield mcp


def calling(name: str, args: dict[str, Any]) -> Callable[[Call], Any]:
    def respond(call: Call) -> Any:
        if call.last.type == "tool":
            return f"Done: {call.last.text}"
        return tool_call(name, args)

    return respond


async def test_descriptors(runtime: MCPRuntime) -> None:
    descriptors = {item.name: item for item in await mcp_descriptors(runtime)}
    assert {
        "ops__service_status",
        "ops__restart",
        "mcp:ops:resource:docs://runbook",
    } <= set(descriptors)
    tool = descriptors["ops__service_status"]
    assert tool.kind is CapabilityKind.MCP
    assert tool.model_callable
    assert tool.metadata["mcp_server_id"] == "ops"
    assert not descriptors["mcp:ops:prompt:triage"].model_callable


async def test_agent_routes_and_calls_mcp_tools(
    runtime: MCPRuntime, model: Callable[..., ScriptedChatModel]
) -> None:
    scripted = model(calling("ops__service_status", {"service": "billing"}))
    agent = create_agent(model=scripted, pattern="react", mcp=runtime)
    selection = await agent.aroute_capabilities("check billing")
    assert "ops__service_status" in selection.tool_names
    answer = await agent.ainvoke("Check billing status.")
    assert answer.text == "Done: billing is healthy"
    assert "ops__service_status" in scripted.calls[0].tools
    with pytest.raises(ConfigurationError, match="asynchronous integrations"):
        agent.invoke("sync")


async def test_execute_mcp_capabilities(
    runtime: MCPRuntime, model: Callable[..., ScriptedChatModel]
) -> None:
    agent = create_agent(model=model(), mcp=runtime)
    status = await agent.aexecute_capability("ops__service_status", {"service": "api"})
    assert "api is healthy" in str(status)
    resource = await agent.aexecute_capability("mcp:ops:resource:docs://runbook")
    assert "Restart, then verify health." in str(resource)
    prompt = await agent.aexecute_capability("mcp:ops:prompt:triage", {"service": "db"})
    assert "Triage db now." in str(prompt)
    with pytest.raises(CapabilityResolutionError, match="do not accept arguments"):
        await agent.aexecute_capability("mcp:ops:resource:docs://runbook", {"x": 1})
    handle = MCPCapability(
        runtime, "ops:tool:service_status", "tool", "ops__service_status"
    )
    with pytest.raises(CapabilityResolutionError, match="asynchronous"):
        handle.invoke({"service": "api"})


async def test_mcp_tools_under_policy(
    runtime: MCPRuntime, model: Callable[..., ScriptedChatModel], tmp_path: Path
) -> None:
    store = SQLiteRuntimeStore(str(tmp_path / "mcp.db"))
    try:
        policy = PolicyManager(
            policy=RuntimePolicy(
                version="mcp-v1",
                rules=(
                    PolicyRule(
                        "allow status", PolicyEffect.ALLOW, tool="ops__service_status"
                    ),
                    PolicyRule(
                        "allow restarts", PolicyEffect.ALLOW, tool="ops__restart"
                    ),
                    PolicyRule("no wipes", PolicyEffect.DENY, tool="ops__wipe_disk"),
                ),
            ),
            store=store,
            consequential={"ops__restart": "service"},
            namespace="ops",
        )
        scripted = model(calling("ops__restart", {"service": "api"}))
        agent = create_agent(
            model=scripted, pattern="react", mcp=runtime, policy=policy
        )
        names = (await agent.aroute_capabilities("anything")).names
        assert "ops__wipe_disk" not in names
        assert "ops__restart" in names
        first = await agent.ainvoke("Restart api")
        second = await agent.ainvoke("Restart api again")
        assert "api restarted" in first.text
        assert "api restarted" in second.text
        assert restarts == ["api"]
        governed = await agent.aexecute_capability("ops__restart", {"service": "api"})
        assert "api restarted" in str(governed)
        assert restarts == ["api"]
    finally:
        store.close()


async def test_explainability_keeps_routed_tools(
    runtime: MCPRuntime, model: Callable[..., ScriptedChatModel]
) -> None:
    from langchain_core.tools import tool
    from langgraph_xai import XAIRuntime

    @tool
    def local_lookup(query: str) -> str:
        """Look something up locally."""
        return query

    xai = XAIRuntime(application_id="agloom-mcp-tests")
    try:
        scripted = model(lambda call: "done")
        agent = create_agent(
            model=scripted,
            tools=[local_lookup],
            pattern="react",
            mcp=runtime,
            explainability=xai,
        )
        await agent.ainvoke("Check the billing service status.")
        assert {"local_lookup", "ops__service_status"} <= set(scripted.calls[0].tools)
    finally:
        await xai.close()
