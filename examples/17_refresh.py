"""Fresh sources: refresh-engine re-indexes changed data when the loop needs it.

Pass a `refresh_engine.RefreshEngine` as `refresh=` and allow refreshes with
`LoopPolicy(allow_refresh=True)`. When an iteration reports stale sources (here
a freshness probe compares the search index with the system of record), the
loop runs an incremental refresh (only changed documents are re-indexed) and
tries again with the fresh data. Agents with refresh use the async APIs.

Run:
    uv run python examples/17_refresh.py
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
    AgentEvent,
    DefaultLoopObserver,
    EventName,
    LoopAction,
    LoopObservation,
    LoopPolicy,
    LoopTerminationReason,
    TopologyName,
    create_agent,
)

POLICIES = {
    "returns": "Returns are accepted within 30 days of delivery.",
    "shipping": "Standard shipping takes 3 to 5 business days.",
}
INDEX: dict[str, str] = {}


class PolicySource:
    """The system of record that refresh-engine watches."""

    async def discover(self) -> DiscoveryResult:
        return DiscoveryResult(resources=[Resource(name) for name in POLICIES])

    async def snapshot(self, resource: Resource) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource.resource_id, content=POLICIES[resource.resource_id]
        )


async def reindex(
    resource: Resource,
    snapshot: ResourceSnapshot | None,
    action: PlanAction,
    request: RefreshRequest,
) -> None:
    """Apply one change to the search index."""
    if action is PlanAction.DELETE or snapshot is None:
        INDEX.pop(resource.resource_id, None)
    else:
        INDEX[resource.resource_id] = snapshot.content


def search_policies(topic: Literal["returns", "shipping"]) -> str:
    """Look up the current company policy for a topic."""
    return INDEX.get(topic, "No policy found for this topic.")


class FreshnessProbe(DefaultLoopObserver):
    """Reports stale sources when the index lags the system of record."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        if INDEX != POLICIES:
            return observation.model_copy(
                update={
                    "refresh_required": True,
                    "feedback": ("The policy index lags the system of record.",),
                }
            )
        return observation.model_copy(update={"validation_passed": True})


async def main() -> None:
    events: list[AgentEvent] = []
    async with RefreshEngine(PolicySource(), reindex) as engine:
        initial = await engine.refresh()
        print(f"Initial index: {initial.refreshed_count} documents")
        POLICIES["returns"] = "Returns are accepted within 45 days of delivery."

        agent = create_agent(
            model=real_model(),
            tools=[search_policies],
            pattern="react",
            refresh=engine,
            loop_policy=LoopPolicy(max_iterations=3, allow_refresh=True),
            loop_observer=FreshnessProbe(),
            observers=[events.append],
        )
        result = await agent.ainvoke_loop(
            "Look up the returns policy: how many days do customers have to "
            "return a product?"
        )

    state = result.state
    refreshed = [e.payload for e in events if e.name == EventName.LOOP_REFRESHED]
    assert [entry.decision.action for entry in state.history] == [
        LoopAction.REFRESH,
        LoopAction.COMPLETE,
    ], state.history
    assert refreshed, "the loop should refresh the stale index"
    assert refreshed[0]["refreshed"] == 1, refreshed
    assert state.termination_reason is LoopTerminationReason.VALIDATED
    assert state.active_topology is TopologyName.REACT
    print(verified_text(result.result, "45"))
    print(f"Refreshed {refreshed[0]['refreshed']} changed document mid-run.")


if __name__ == "__main__":
    asyncio.run(main())
