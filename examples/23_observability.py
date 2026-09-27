from __future__ import annotations

import os

from _support import real_model
from langchain_core.tools import tool

from agloom import LangfuseObserver, LoggingObserver, PrometheusObserver, create_agent


@tool
def execution_modes() -> str:
    """Return Agloom's direct-execution terminology."""

    return "Agloom has eight topologies plus DIRECT execution mode."


prometheus = PrometheusObserver()
prometheus.start_http_server(
    port=int(os.getenv("AGLOOM_METRICS_PORT", "9464")),
    address=os.getenv("AGLOOM_METRICS_HOST", "127.0.0.1"),
)
langfuse = LangfuseObserver(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)
agent = create_agent(
    model=real_model(),
    tools=[execution_modes],
    observers=[LoggingObserver(), prometheus, langfuse],
)

try:
    print(
        agent.invoke("Use execution_modes and explain DIRECT in one sentence.").content
    )
finally:
    agent.close_observability()
