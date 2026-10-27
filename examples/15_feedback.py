"""Human feedback: corrections on a thread reach the agent's next run.

Pass a `feedback_manager.FeedbackManager` as `feedback=`. Before each loop
iteration, Agloom collects the human feedback submitted for the conversation
thread, applies it as context, and acknowledges it, so reviewers can correct
an agent without redeploying it. The application owns the rest of the
lifecycle (handled, resolved). Agents with feedback use the async APIs.

Run:
    uv run python examples/15_feedback.py
"""

import asyncio

from _support import real_model, verified_text
from feedback_manager import (
    FeedbackCategory,
    FeedbackManager,
    FeedbackSource,
    FeedbackStatus,
    FeedbackTarget,
    FeedbackTargetType,
)

from agloom import create_agent


def support_hours() -> str:
    """Return the customer-support closing time from the published schedule."""
    return "Customer support closes at 5 PM."


async def main() -> None:
    manager = FeedbackManager()
    agent = create_agent(
        model=real_model(), tools=[support_hours], pattern="react", feedback=manager
    )
    config = {"configurable": {"thread_id": "support-desk-42"}}
    question = "When does customer support close today? Answer in one sentence."

    before = await agent.ainvoke(question, config)
    print("Before feedback:", verified_text(before, ("5 PM", "5:00", "5pm", "17:00")))

    correction = await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.CORRECTION,
        target=FeedbackTarget(type=FeedbackTargetType.THREAD, id="support-desk-42"),
        payload={
            "text": "Today support closes at 7 PM because of the product launch; "
            "the published schedule is outdated."
        },
        idempotency_key="support-hours-launch-day",
    )

    after = await agent.ainvoke(question, config)
    print("After feedback: ", verified_text(after, ("7 PM", "7:00", "7pm", "19:00")))

    applied = await manager.get(correction.feedback_id)
    assert applied is not None
    assert applied.status == FeedbackStatus.ACKNOWLEDGED, applied.status
    await manager.mark_handled(correction.feedback_id)
    resolved = await manager.resolve(
        correction.feedback_id, resolution={"schedule_updated": True}
    )
    print(f"Feedback lifecycle: acknowledged by the agent -> {resolved.status}")


if __name__ == "__main__":
    asyncio.run(main())
