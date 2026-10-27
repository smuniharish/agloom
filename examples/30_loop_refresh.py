"""Refresh on stagnation: when progress stalls, refresh sources before giving up.

A grounding check (a `LoopObserver`) scores each answer against the system of
record. While the search index is stale, answers stay wrong and progress
stalls; after `stagnation_threshold` iterations without progress the
controller refreshes (`allow_refresh=True`) instead of stopping. The
refresh-engine re-indexes only the changed product, and the next iteration
succeeds. Agents with refresh use the async APIs.

Run:
    uv run python examples/30_loop_refresh.py
"""

import asyncio
from typing import Any, Literal

from _support import real_model, verified_text
from refresh_engine import (
    DiscoveryResult,
    PlanAction,
    RefreshEngine,
    RefreshRequest,
    Resource,
    ResourceSnapshot,
)

from agloom import (
    DefaultLoopObserver,
    LoopAction,
    LoopObservation,
    LoopPolicy,
    LoopTerminationReason,
    create_agent,
)

CATALOG = {"pro": "$49 per month", "team": "$99 per month"}
INDEX: dict[str, str] = {}


class Catalog:
    """The pricing system of record watched by refresh-engine."""

    async def discover(self) -> DiscoveryResult:
        return DiscoveryResult(resources=[Resource(plan) for plan in CATALOG])

    async def snapshot(self, resource: Resource) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource.resource_id, content=CATALOG[resource.resource_id]
        )


async def reindex(
    resource: Resource,
    snapshot: ResourceSnapshot | None,
    action: PlanAction,
    request: RefreshRequest,
) -> None:
    """Apply one catalog change to the search index."""
    if action is PlanAction.DELETE or snapshot is None:
        INDEX.pop(resource.resource_id, None)
    else:
        INDEX[resource.resource_id] = snapshot.content


def plan_price(plan: Literal["pro", "team"]) -> str:
    """Look up the price of a subscription plan in the search index."""
    return f"The {plan} plan costs {INDEX[plan]}."


class GroundingCheck(DefaultLoopObserver):
    """Scores an answer 1.0 when it quotes the system of record, else 0.0."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        price = CATALOG["pro"].split()[0]
        grounded = price in kwargs["result"].text
        return observation.model_copy(
            update={"objective_progress": 1.0 if grounded else 0.0}
        )


async def main() -> None:
    async with RefreshEngine(Catalog(), reindex) as engine:
        await engine.refresh()
        CATALOG["pro"] = "$59 per month"

        agent = create_agent(
            model=real_model(),
            tools=[plan_price],
            pattern="react",
            refresh=engine,
            loop_policy=LoopPolicy(
                max_iterations=5, stagnation_threshold=2, allow_refresh=True
            ),
            loop_observer=GroundingCheck(),
        )
        result = await agent.ainvoke_loop("How much does the pro plan cost?")

    state = result.state
    actions = [entry.decision.action for entry in state.history]
    assert actions == [LoopAction.CONTINUE, LoopAction.REFRESH, LoopAction.COMPLETE]
    assert state.termination_reason is LoopTerminationReason.OBJECTIVE_SATISFIED
    print(f"Decisions: {[action.value for action in actions]}")
    print(verified_text(result.result, "$59"))


if __name__ == "__main__":
    asyncio.run(main())
