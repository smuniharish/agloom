"""A small FastMCP operations server used by the MCP examples.

It offers two tools, a runbook resource, and an incident prompt. Use it
in-process or run it over stdio (MCP clients start it; no banner on stderr):

    await runtime.register_mcp("ops", server, refresh=RefreshPolicy(on_register=True))

    uv run python examples/_mcp_server.py
"""

from __future__ import annotations

from fastmcp import FastMCP

server = FastMCP("ops")

STATUS = {
    "billing": "degraded: p95 latency is 2.4 s against a 1 s objective",
    "search": "healthy: p95 latency is 180 ms",
}
RESTARTS: list[str] = []
"""Restarts performed by this process, so examples can prove exactly-once effects."""


@server.tool
def service_status(service: str) -> str:
    """Return the current health of a service (billing or search)."""
    return f"{service}: {STATUS.get(service, 'unknown service')}"


@server.tool
def restart_service(service: str, change_id: str) -> str:
    """Restart a service under an approved change request."""
    RESTARTS.append(change_id)
    return f"{service} restarted under change {change_id}"


@server.resource("ops://runbooks/billing")
def billing_runbook() -> str:
    """The billing incident runbook."""
    return (
        "Billing runbook: when p95 latency exceeds 1 s, check the payment gateway, "
        "then restart the billing service under a change request."
    )


@server.prompt
def incident_summary(service: str) -> str:
    """Ask for a three-line incident summary of a service."""
    return f"Write a three-line incident summary for the {service} service."


if __name__ == "__main__":
    server.run(show_banner=False)
