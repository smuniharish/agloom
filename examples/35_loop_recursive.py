"""Recursive loops: each iteration may fan out into bounded child executions.

Loop Engineering composes with recursion. Every iteration runs the Planner,
whose steps run as recursive children (each analyzed and executed on its own,
here with ReAct and an internal reporting tool), and the model evaluator
judges the combined result. The recursion budget (`max_children`,
`max_execution_time`) covers the whole invocation, across all loop
iterations, so a loop of recursive runs stays bounded overall.

Run:
    uv run python examples/35_loop_recursive.py
"""

from typing import Literal

from _support import real_model, verified_text

from agloom import (
    AgentEvent,
    EventName,
    ExecutionLimits,
    LoopPolicy,
    LoopTerminationReason,
    create_agent,
)

REPORTS = {
    "north": "North region: Q3 revenue $4.2M, up 12% on Q2.",
    "south": "South region: Q3 revenue $3.1M, down 4% on Q2.",
    "west": "West region: Q3 revenue $5.6M, up 9% on Q2.",
}
looked_up: list[str] = []


def region_report(region: Literal["north", "south", "west"]) -> str:
    """Return the internal Q3 sales report of one sales region."""
    looked_up.append(region)
    return REPORTS[region]


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(),
        tools=[region_report],
        pattern="planner",
        limits=ExecutionLimits(
            max_depth=1, max_children=8, allowed_nested_topologies=("react",)
        ),
        loop_policy=LoopPolicy(max_iterations=2),
        observers=[events.append],
    )

    result = agent.invoke_loop(
        "Prepare one independent summary per sales region (north, south, and "
        "west): its Q3 revenue and its growth."
    )

    children = [
        event
        for event in events
        if event.name == EventName.TOPOLOGY_SELECTION_STARTED
        and event.payload.get("depth") == 1
    ]
    state = result.state
    assert set(looked_up) == set(REPORTS), looked_up
    assert children, "every planner step runs as a recursive child"
    assert state.termination_reason in {
        LoopTerminationReason.OBJECTIVE_SATISFIED,
        LoopTerminationReason.VALIDATED,
    }, state.termination_reason
    print(verified_text(result.result, "4.2", "3.1", "5.6"))
    print(f"{len(children)} recursive children in {state.iteration} iteration(s)")


if __name__ == "__main__":
    main()
