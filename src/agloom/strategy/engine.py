"""Deterministic execution selection from a structured task analysis."""

from __future__ import annotations

from collections.abc import Sequence

from agloom.errors import TopologySelectionError
from agloom.models import ExecutionDecision, ExecutionMode, TaskAnalysis, TopologyName


class StrategyEngine:
    """Maps a `TaskAnalysis` to DIRECT or exactly one topology.

    The preferred topology follows the dominant requirement, in priority
    order: composition (Hybrid), shared workspace (Blackboard), peer handoffs
    (Swarm), parallel subtasks (Planner), delegation to specialists
    (Supervisor), decomposition (Planner), validation of complex work
    (Reflection), ordered stages (Pipeline), and tool use (ReAct). Tasks
    without such requirements run DIRECT when DIRECT is allowed.

    When a developer allow-list does not contain the preferred topology, the
    candidate that satisfies the most requirements is chosen, preferring the
    allow-list order on ties.
    """

    def decide(
        self,
        analysis: TaskAnalysis,
        candidates: Sequence[TopologyName] | None = None,
        *,
        allow_direct: bool = True,
    ) -> ExecutionDecision:
        """Return the execution decision for one analyzed task.

        Raises:
            TopologySelectionError: If an empty allow-list is supplied, or no
                topology can be selected while DIRECT is not allowed.
        """
        if candidates is not None and not candidates:
            raise TopologySelectionError("no topology candidates are available")
        preferred = preferred_topology(analysis)
        if preferred is None and allow_direct:
            return ExecutionDecision(
                mode=ExecutionMode.DIRECT,
                reason="the task does not require orchestration",
            )
        if candidates is None:
            if preferred is None:
                raise TopologySelectionError(
                    "no topology satisfies the task and DIRECT is not allowed"
                )
            return ExecutionDecision(
                mode=ExecutionMode.TOPOLOGY,
                topology=preferred,
                reason="selected from the task requirements",
            )
        if preferred in candidates:
            return ExecutionDecision(
                mode=ExecutionMode.TOPOLOGY,
                topology=preferred,
                reason="selected from the task requirements within the allow-list",
            )
        selected = max(candidates, key=lambda candidate: score(candidate, analysis))
        return ExecutionDecision(
            mode=ExecutionMode.TOPOLOGY,
            topology=selected,
            reason="highest-scoring topology in the developer allow-list",
        )


def preferred_topology(analysis: TaskAnalysis) -> TopologyName | None:
    """Return the topology that matches the task's dominant requirement."""
    if analysis.requires_composition:
        return TopologyName.HYBRID
    if analysis.requires_shared_workspace:
        return TopologyName.BLACKBOARD
    if analysis.requires_handoffs:
        return TopologyName.SWARM
    if analysis.requires_parallelism:
        return TopologyName.PLANNER
    if analysis.requires_delegation:
        return TopologyName.SUPERVISOR
    if analysis.requires_decomposition:
        return TopologyName.PLANNER
    if analysis.requires_validation and analysis.complexity >= 4:
        return TopologyName.REFLECTION
    if analysis.requires_ordering and analysis.estimated_steps >= 3:
        return TopologyName.PIPELINE
    if analysis.requires_tools:
        return TopologyName.REACT
    return None


def score(candidate: TopologyName, analysis: TaskAnalysis) -> int:
    """Count the task requirements a candidate topology satisfies."""
    satisfied = {
        TopologyName.REACT: (analysis.requires_tools,),
        TopologyName.SUPERVISOR: (analysis.requires_delegation,),
        TopologyName.SWARM: (analysis.requires_handoffs,),
        TopologyName.PIPELINE: (analysis.requires_ordering,),
        TopologyName.PLANNER: (
            analysis.requires_decomposition,
            analysis.requires_parallelism,
        ),
        TopologyName.REFLECTION: (analysis.requires_validation,),
        TopologyName.BLACKBOARD: (analysis.requires_shared_workspace,),
        TopologyName.HYBRID: (analysis.requires_composition,),
    }
    return sum(satisfied[candidate])
