"""Topology adaptation: switch the execution structure when results fall short.

With `LoopPolicy(allow_topology_change=True)` and a `pattern` allow-list, a
result can request another topology for the next iteration
(`requested_topology`). Here a compliance check (a `LoopObserver`) rejects a
client email that lacks the mandatory compliance footer and asks for
Reflection (draft, review, revise) with the rule as feedback. Topology
changes stay inside the allow-list.

Run:
    uv run python examples/36_loop_topology_adaptation.py
"""

from typing import Any

from _support import real_model, verified_text

from agloom import (
    DefaultLoopObserver,
    LoopAction,
    LoopObservation,
    LoopPolicy,
    LoopTerminationReason,
    TopologyName,
    create_agent,
)

FOOTER = "Reviewed under Acme compliance policy CP-12."


def fund_performance(fund: str) -> str:
    """Return the 2025 performance of an investment fund."""
    return f"{fund}: +8.4% in 2025 versus +6.1% for its benchmark."


class ComplianceCheck(DefaultLoopObserver):
    """Client emails must end with the compliance footer."""

    def observe(self, **kwargs: Any) -> LoopObservation:
        observation = super().observe(**kwargs)
        if not observation.success:
            return observation
        if FOOTER in kwargs["result"].text:
            return observation.model_copy(update={"validation_passed": True})
        return observation.model_copy(
            update={
                "requested_topology": TopologyName.REFLECTION,
                "feedback": (f"Client emails must end with exactly: {FOOTER}",),
            }
        )


def main() -> None:
    agent = create_agent(
        model=real_model(),
        tools=[fund_performance],
        pattern=["react", "reflection"],
        loop_policy=LoopPolicy(max_iterations=3, allow_topology_change=True),
        loop_observer=ComplianceCheck(),
    )

    result = agent.invoke_loop(
        "Write a short client email summarizing the 2025 performance of the "
        "Global Equity fund."
    )

    state = result.state
    change = next(e for e in state.history if e.decision.next_topology is not None)
    assert change.decision.action is LoopAction.CHANGE_TOPOLOGY
    assert state.history[-1].observation.topology is TopologyName.REFLECTION
    assert state.termination_reason is LoopTerminationReason.VALIDATED
    topologies = [str(entry.observation.topology) for entry in state.history]
    print(f"Topologies per iteration: {topologies}")
    print(verified_text(result.result, FOOTER, "8.4"))


if __name__ == "__main__":
    main()
