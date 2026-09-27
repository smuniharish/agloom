"""Structured strategy selection."""

from __future__ import annotations

from collections.abc import Sequence

from agloom.errors import TopologySelectionError
from agloom.models import (
    ExecutionDecision,
    ExecutionMode,
    TaskAnalysis,
    TopologyName,
)


class StrategyEngine:
    def decide(
        self,
        analysis: TaskAnalysis,
        candidates: Sequence[TopologyName] | None = None,
        *,
        allow_direct: bool = False,
    ) -> ExecutionDecision:
        if candidates is not None and not candidates:
            raise TopologySelectionError("no topology candidates are available")
        selected = self._select(analysis)
        if selected is None and allow_direct:
            return ExecutionDecision(
                mode=ExecutionMode.DIRECT,
                reason="child task does not require a topology",
            )
        if candidates is not None and selected not in candidates:
            selected = max(
                candidates, key=lambda candidate: self._score(candidate, analysis)
            )
            reason = "selected highest-scoring topology from developer allow-list"
        else:
            reason = "selected from structured task requirements"

        if selected is None:
            return ExecutionDecision(
                mode=ExecutionMode.DIRECT,
                reason="task does not require orchestration",
            )
        return ExecutionDecision(
            mode=ExecutionMode.TOPOLOGY,
            topology=selected,
            reason=reason,
        )

    @staticmethod
    def _select(analysis: TaskAnalysis) -> TopologyName | None:
        if analysis.requires_parallelism:
            return TopologyName.SWARM
        if analysis.requires_composition:
            return TopologyName.HYBRID
        if analysis.requires_shared_workspace:
            return TopologyName.BLACKBOARD
        if analysis.requires_delegation:
            return TopologyName.SUPERVISOR
        if analysis.requires_validation and analysis.complexity >= 4:
            return TopologyName.REFLECTION
        if analysis.requires_ordering and analysis.estimated_steps >= 3:
            return TopologyName.PIPELINE
        if analysis.requires_decomposition:
            return TopologyName.PLANNER
        if analysis.requires_tools:
            return TopologyName.REACT
        return None

    @classmethod
    def _score(cls, candidate: TopologyName, analysis: TaskAnalysis) -> int:
        preferred = cls._select(analysis)
        if candidate is preferred:
            return 100
        scores = {
            TopologyName.REACT: int(analysis.requires_tools),
            TopologyName.SUPERVISOR: int(analysis.requires_delegation),
            TopologyName.PIPELINE: int(analysis.requires_ordering),
            TopologyName.PLANNER: int(analysis.requires_decomposition),
            TopologyName.REFLECTION: int(analysis.requires_validation),
            TopologyName.SWARM: int(analysis.requires_parallelism),
            TopologyName.BLACKBOARD: int(
                analysis.requires_shared_workspace
                or analysis.requires_parallelism
                or analysis.requires_delegation
            ),
            TopologyName.HYBRID: int(
                analysis.requires_composition
                or (analysis.requires_decomposition and analysis.requires_delegation)
            ),
        }
        return scores[candidate]
