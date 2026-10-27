"""Explainability: opt-in capture of what the agent did, for audit and explanation.

Pass a `langgraph_xai.XAIRuntime` as `explainability=` to instrument the
agent's model and graph runs: node executions, tool executions, and state
transitions are recorded per run. It is opt-in, so agents without it pay
nothing. Register an observability provider (for example
`LangfuseObservability`) on the runtime to export the records.

Run:
    uv run python examples/18_explainability.py
"""

from _support import real_model, verified_text
from langgraph_xai import XAIRuntime

from agloom import create_agent


def order_status(order_id: str) -> str:
    """Return the shipping status of an order."""
    return f"Order {order_id} left the Rotterdam warehouse and arrives Friday."


def main() -> None:
    xai = XAIRuntime(application_id="agloom-examples", graph_id="support-agent")
    agent = create_agent(
        model=real_model(), tools=[order_status], pattern="react", explainability=xai
    )

    with xai.collect_runs() as runs:
        answer = agent.invoke("Where is order A-1001 and when will it arrive?")

    print(verified_text(answer, "Friday"))
    tools = [tool.tool_name for run in runs for tool in run.execution.tools]
    nodes = [node.node_id for run in runs for node in run.execution.nodes]
    assert "order_status" in tools, tools
    assert {"model", "tools"} <= set(nodes), nodes
    print(f"Captured {len(runs)} run(s): nodes {nodes}, tools {tools}")

    agent.close()
    xai.run_sync(xai.close())


if __name__ == "__main__":
    main()
