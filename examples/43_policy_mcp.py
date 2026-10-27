"""MCP under policy: govern MCP tools, resources, and prompts like local tools.

MCP capabilities are policy subjects too. Tools are named
`<server_id>__<tool>` (here `ops__service_status`, `ops__restart_service`),
resources and prompts `mcp:<server>:<type>:<name>`. Reads are allowed, a
restart is consequential (idempotency key `change_id`) and needs approval,
and everything else is denied by default. The server runs in-process here.

Run:
    uv run python examples/43_policy_mcp.py
"""

import asyncio
import tempfile
from pathlib import Path

import _mcp_server
from _support import real_model, verified_text
from agentbarrier import Decision
from agentbarrier.runtime import (
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from langgraph.checkpoint.memory import InMemorySaver
from mcp_capability_router import MCPRuntime, RefreshPolicy

from agloom import (
    APPROVAL_INTERRUPT,
    ExecutionPaused,
    PolicyManager,
    PolicyViolationError,
    create_agent,
)

OPS_POLICY = RuntimePolicy(
    version="ops-2026-10",
    rules=(
        PolicyRule(
            "read service status", PolicyEffect.ALLOW, tool="ops__service_status"
        ),
        PolicyRule("read runbooks", PolicyEffect.ALLOW, tool="mcp:ops:resource:*"),
        PolicyRule(
            "approve restarts",
            PolicyEffect.REQUIRE_APPROVAL,
            tool="ops__restart_service",
        ),
    ),
)


async def main() -> None:
    config = {"configurable": {"thread_id": "incident-2041"}}
    async with MCPRuntime() as runtime:
        await runtime.register_mcp(
            "ops", _mcp_server.server, refresh=RefreshPolicy(on_register=True)
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            store = SQLiteRuntimeStore(Path(folder) / "agentbarrier.db")
            agent = create_agent(
                model=real_model(),
                mcp=runtime,
                pattern="react",
                policy=PolicyManager(
                    policy=OPS_POLICY,
                    store=store,
                    consequential={"ops__restart_service": "change_id"},
                    namespace="operations",
                ),
                checkpointer=InMemorySaver(),
                checkpoint_authorizer=lambda thread_id: thread_id.startswith(
                    "incident-"
                ),
            )

            runbook = await agent.aexecute_capability(
                "mcp:ops:resource:ops://runbooks/billing"
            )
            print("Runbook:", verified_text(runbook[0].text, "restart"))
            try:
                await agent.aexecute_capability(
                    "mcp:ops:prompt:incident_summary", {"service": "billing"}
                )
            except PolicyViolationError as denied:
                print(f"Denied: {denied}")
            else:
                raise AssertionError("prompts are not allowed by this policy")

            try:
                await agent.ainvoke(
                    "Billing is degraded. Check its status, then restart it under "
                    "change CHG-2041.",
                    config,
                )
            except ExecutionPaused as paused:
                approval = paused.interrupts[0].value
            else:
                raise AssertionError("restarts must wait for approval")
            assert approval["type"] == APPROVAL_INTERRUPT, approval
            print(f"Awaiting approval: {approval['tool']} {approval['arguments']}")
            store.decide(approval["action_id"], Decision.APPROVE, decided_by="sre")

            answer = await agent.aresume(config)
            print(verified_text(answer, "CHG-2041"))
            assert _mcp_server.RESTARTS == ["CHG-2041"], _mcp_server.RESTARTS
            store.close()


if __name__ == "__main__":
    asyncio.run(main())
