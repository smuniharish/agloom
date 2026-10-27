"""Loop Engineering basics: one-shot by default, early completion when judged done.

Every Agloom run is a loop. The default `LoopPolicy()` is one-shot: one
iteration and no evaluation call, so simple requests stay cheap. Raise
`max_iterations` and each result is judged against the objective by a
`ModelLoopEvaluator` (used automatically). The loop stops as soon as the
objective is satisfied instead of spending its whole budget.

Run:
    uv run python examples/25_loop_automatic_completion.py
"""

from _support import real_model, verified_text

from agloom import LoopPolicy, LoopTerminationReason, create_agent

OBJECTIVE = "List exactly three benefits of code review as a numbered list."


def main() -> None:
    model = real_model()

    one_shot = create_agent(model=model).invoke_loop(OBJECTIVE)
    assert one_shot.state.iteration == 1
    assert one_shot.state.termination_reason is LoopTerminationReason.ONE_SHOT_COMPLETE
    print(f"Default policy: {one_shot.state.termination_reason} after 1 iteration")

    agent = create_agent(model=model, loop_policy=LoopPolicy(max_iterations=4))
    result = agent.invoke_loop(OBJECTIVE)

    state = result.state
    assert state.termination_reason in {
        LoopTerminationReason.OBJECTIVE_SATISFIED,
        LoopTerminationReason.VALIDATED,
    }, state.termination_reason
    assert state.iteration < agent.loop_policy.max_iterations, state.iteration
    assert state.progress is not None, "the model evaluator scores every result"
    print(verified_text(result.result, "1", "2", "3"))
    print(
        f"Evaluated loop: {state.termination_reason} after {state.iteration} of "
        f"{agent.loop_policy.max_iterations} iterations (progress {state.progress:.0%})"
    )


if __name__ == "__main__":
    main()
