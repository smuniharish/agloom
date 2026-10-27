"""Recursion: Planner steps run as bounded recursive child executions.

With `ExecutionLimits(max_depth=1)`, every Planner step becomes a child
execution that is analyzed on its own and runs DIRECT or on a topology from
`allowed_nested_topologies`. Depth, child count, and wall-clock time are
bounded for the whole tree, so recursion can never run away. The observer
records each child's selection.

Run:
    uv run python examples/11_recursion.py
"""

from _support import real_model, verified_text

from agloom import AgentEvent, EventName, ExecutionLimits, create_agent


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(),
        pattern="planner",
        limits=ExecutionLimits(
            max_depth=1,
            max_children=4,
            max_execution_time=180,
            allowed_nested_topologies=("react", "reflection"),
        ),
        observers=[events.append],
    )

    answer = agent.invoke(
        "Answer two independent questions as separate steps: why does caching "
        "reduce API latency, and why does rate limiting protect an API? Give "
        "one sentence for each."
    )

    children = [
        event
        for event in events
        if event.name == EventName.TOPOLOGY_SELECTION_STARTED
        and event.payload.get("depth") == 1
    ]
    assert children, "every planner step should run as a recursive child"
    print(verified_text(answer, "cach", "rate limit"))
    print(f"Recursive child executions: {len(children)}")


if __name__ == "__main__":
    main()
