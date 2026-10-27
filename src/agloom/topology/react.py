"""ReAct: one tool-calling agent built with LangChain's `create_agent`."""

from __future__ import annotations

from agloom.models import TopologyName
from agloom.topology.base import (
    TopologyArtifact,
    TopologyBuildContext,
    langchain_agent,
)

_DEFAULT_INSTRUCTION = (
    "Use the available tools when they help. Stop when the request is fully answered."
)


class ReactTopology:
    """Reasoning interleaved with tool calls and observations.

    Graph nodes: `model` and `tools`. Input: `{"messages": [...]}`.
    """

    name = TopologyName.REACT

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        """Compile the ReAct agent graph."""
        graph = langchain_agent(
            context.model,
            name="agloom_react",
            tools=list(context.tools),
            system_prompt=context.system_prompt or _DEFAULT_INSTRUCTION,
            middleware=list(context.middleware),
            interrupt_before=list(context.interrupt_before) or None,
            interrupt_after=list(context.interrupt_after) or None,
        )
        return TopologyArtifact(self.name, graph)
