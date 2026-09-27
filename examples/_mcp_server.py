from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Agloom example")


@mcp.tool()
def add(left: int, right: int) -> int:
    """Add two integers."""

    return left + right


@mcp.resource("agloom://guide")
def guide() -> str:
    """Return a small Agloom capability guide."""

    return "Agloom routes local, application, and MCP capabilities together."


@mcp.prompt()
def summarize(topic: str) -> str:
    """Create a request to summarize a topic."""

    return f"Summarize {topic} in one sentence."


if __name__ == "__main__":
    mcp.run(transport="stdio")
