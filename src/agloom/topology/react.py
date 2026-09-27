"""ReAct topology implemented with LangChain's agent primitive."""

from __future__ import annotations

from langchain.agents import create_agent as create_langchain_agent

from agloom.models import TopologyName
from agloom.topology.base import TopologyArtifact, TopologyBuildContext


class ReactTopology:
    name = TopologyName.REACT

    def build(self, context: TopologyBuildContext) -> TopologyArtifact:
        graph = create_langchain_agent(
            model=context.model,
            tools=list(context.tools),
            middleware=context.middleware,
            system_prompt=context.system_prompt
            or "Use available tools when needed. Stop when the task is complete.",
            checkpointer=context.checkpointer,
            interrupt_before=list(context.interrupt_before) or None,
            interrupt_after=list(context.interrupt_after) or None,
        )
        return TopologyArtifact(self.name, graph)
