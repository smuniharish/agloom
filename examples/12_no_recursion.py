"""No recursion: the default keeps Planner steps as plain step agents.

`ExecutionLimits()` defaults to `max_depth=0`: steps of a Planner run as
tool-capable step agents and never start child executions, so there is no
per-step analysis call. The fan-out is still bounded by `max_workers`, and the
plan length by `max_plan_steps`.

Run:
    uv run python examples/12_no_recursion.py
"""

from _support import real_model, verified_text

from agloom import AgentEvent, EventName, ExecutionLimits, create_agent


def main() -> None:
    events: list[AgentEvent] = []
    agent = create_agent(
        model=real_model(),
        pattern="planner",
        limits=ExecutionLimits(max_plan_steps=3, max_workers=2),
        observers=[events.append],
    )

    answer = agent.invoke(
        "Answer two independent questions as separate steps: what is one "
        "benefit of unit tests, and what is one benefit of code review? Give "
        "one sentence for each."
    )

    assert not agent.limits.recursion_enabled
    assert EventName.TOPOLOGY_SELECTION_STARTED not in [e.name for e in events]
    print(verified_text(answer, "test", "review"))


if __name__ == "__main__":
    main()
