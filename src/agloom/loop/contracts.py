"""Extension contracts for Loop Engineering."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agloom.loop.models import (
    LoopBudget,
    LoopDecision,
    LoopEvaluation,
    LoopObservation,
    LoopPolicy,
    LoopState,
)
from agloom.models import TopologyName


class LoopObserver(Protocol):
    """Normalizes one iteration result into a `LoopObservation`."""

    def observe(
        self,
        *,
        iteration: int,
        result: Any = None,
        error: BaseException | None = None,
        topology: TopologyName | None = None,
        duration: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LoopObservation:
        """Return the observation of one iteration."""
        ...


class LoopEvaluator(Protocol):
    """Interprets an observation relative to the loop objective."""

    def evaluate(
        self, state: LoopState, observation: LoopObservation, policy: LoopPolicy
    ) -> LoopEvaluation:
        """Return the evaluation of one observation."""
        ...


@runtime_checkable
class AsyncLoopEvaluator(LoopEvaluator, Protocol):
    """A loop evaluator with a native asynchronous implementation."""

    async def aevaluate(
        self, state: LoopState, observation: LoopObservation, policy: LoopPolicy
    ) -> LoopEvaluation:
        """Asynchronously return the evaluation of one observation."""
        ...


class LoopController(Protocol):
    """Chooses the next loop action within the policy."""

    def decide(
        self,
        state: LoopState,
        observation: LoopObservation,
        evaluation: LoopEvaluation,
        policy: LoopPolicy,
        budget: LoopBudget,
    ) -> LoopDecision:
        """Return the decision for one evaluated iteration."""
        ...
