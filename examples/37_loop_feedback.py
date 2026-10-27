"""Loop feedback: evaluator feedback steers iterations and is recorded for review.

With `feedback=FeedbackManager(...)`, the loop records the evaluator's
feedback for every iteration (source `evaluator`, target the loop run), so
quality issues are auditable. Human feedback submitted for the conversation
thread is acknowledged and stands: it applies to every later iteration.
Here the evaluator holds acceptance criteria the writer does not see, and an
editor's correction arrives through the manager. Agents with feedback use the
async APIs.

Run:
    uv run python examples/37_loop_feedback.py
"""

import asyncio

from _support import real_model, verified_text
from feedback_manager import (
    FeedbackCategory,
    FeedbackManager,
    FeedbackQuery,
    FeedbackSource,
    FeedbackStatus,
    FeedbackTarget,
    FeedbackTargetType,
)

from agloom import LoopPolicy, ModelLoopEvaluator, create_agent

THREAD = "release-notes-1-0-1"


async def main() -> None:
    model = real_model()
    manager = FeedbackManager()
    agent = create_agent(
        model=model,
        loop_policy=LoopPolicy(max_iterations=4),
        loop_evaluator=ModelLoopEvaluator(
            model,
            criteria=(
                "The release note must cite the tickets AG-101 (policy engine), "
                "AG-102 (loop budgets), and AG-103 (MCP routing). If anything is "
                "missing, say exactly what to add."
            ),
        ),
        feedback=manager,
    )
    editor_note = await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.CORRECTION,
        target=FeedbackTarget(type=FeedbackTargetType.THREAD, id=THREAD),
        payload={"text": "Name the release 'Agloom 1.0.1' in the first bullet."},
    )

    result = await agent.ainvoke_loop(
        "Write a three-bullet release note covering the policy engine, loop "
        "budgets, and MCP capability routing.",
        {"configurable": {"thread_id": THREAD}},
    )

    state = result.state
    recorded = await manager.query(
        FeedbackQuery(source="evaluator", target_type="run", target_id=state.loop_id)
    )
    note = await manager.get(editor_note.feedback_id)
    assert state.iteration >= 2, "the first draft cannot know the ticket IDs"
    assert recorded, "the evaluator's feedback is recorded per iteration"
    assert note is not None
    assert note.status == FeedbackStatus.ACKNOWLEDGED, note.status
    print(verified_text(result.result, "AG-101", "AG-102", "AG-103", "1.0.1"))
    print(f"Editor note {note.status}; it applied to every iteration.")
    for event in recorded:
        print(f"Evaluator feedback, iteration {event.payload['iteration']}:")
        for item in event.payload["feedback"]:
            print(f"  - {item}")


if __name__ == "__main__":
    asyncio.run(main())
