"""Multi-iteration loops: keep working until the evaluator confirms the objective.

Some objectives depend on the world, not on one answer. Here a release rolls
out region by region; each time the agent checks back, one more region is
live. With `LoopPolicy(max_iterations=5)`, a `ModelLoopEvaluator` judges every
report against explicit success criteria, and the loop continues, carrying the
previous result and the evaluator's feedback, until the rollout is confirmed.

Run:
    uv run python examples/26_loop_multi_iteration.py
"""

from _support import real_model, verified_text

from agloom import (
    AgentEvent,
    EventName,
    LoopAction,
    LoopPolicy,
    LoopTerminationReason,
    ModelLoopEvaluator,
    create_agent,
)

REGIONS = ("eu-west", "us-east", "ap-south")
live_regions = 0


def rollout_clock(event: AgentEvent) -> None:
    """Simulated time: one more region goes live before each attempt."""
    global live_regions
    if event.name == EventName.LOOP_ITERATION_STARTED:
        live_regions = min(int(event.payload["iteration"]), len(REGIONS))


def rollout_status() -> str:
    """Return the rollout status of release v2.4 in every region."""
    live, pending = REGIONS[:live_regions], REGIONS[live_regions:]
    if not pending:
        return f"Release v2.4 is live and healthy in all regions: {', '.join(live)}."
    return f"Release v2.4 is live in {', '.join(live)}; pending: {', '.join(pending)}."


def main() -> None:
    model = real_model()
    agent = create_agent(
        model=model,
        tools=[rollout_status],
        pattern="react",
        loop_policy=LoopPolicy(max_iterations=5),
        loop_evaluator=ModelLoopEvaluator(
            model,
            criteria=(
                "Complete only when the report says release v2.4 is live in all "
                "three regions."
            ),
        ),
        observers=[rollout_clock],
    )

    result = agent.invoke_loop(
        "Confirm that release v2.4 is live in every region. Check the rollout "
        "status once per attempt and report it."
    )

    state = result.state
    actions = [entry.decision.action for entry in state.history]
    assert actions == [LoopAction.CONTINUE, LoopAction.CONTINUE, LoopAction.COMPLETE]
    assert state.termination_reason is LoopTerminationReason.OBJECTIVE_SATISFIED
    for entry in state.history:
        progress = entry.evaluation.objective_progress or 0.0
        print(f"iteration {entry.observation.iteration}: progress {progress:.0%}")
    print(verified_text(result.result, "ap-south"))


if __name__ == "__main__":
    main()
