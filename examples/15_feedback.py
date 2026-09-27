import asyncio

from _support import real_model, verified_text
from feedback_manager import (
    FeedbackCategory,
    FeedbackQuery,
    FeedbackSource,
    FeedbackTarget,
    FeedbackTargetType,
)
from langchain_core.tools import tool

from agloom import create_agent


async def main() -> None:
    tool_calls = 0

    @tool
    def support_hours() -> str:
        """Return the currently indexed customer-support closing time."""

        nonlocal tool_calls
        tool_calls += 1
        return "Customer support closes at 5 PM."

    agent = create_agent(
        model=real_model(),
        tools=[support_hours],
        pattern="react",
        feedback_options={},
    )
    response = agent.invoke("Use the support_hours tool and state its answer exactly.")
    assert tool_calls == 1
    answer = verified_text(response, "5 PM")

    manager = agent.capabilities.resolve("feedback_manager")
    received = await manager.submit(
        source=FeedbackSource.HUMAN,
        category=FeedbackCategory.CORRECTION,
        target=FeedbackTarget(
            type=FeedbackTargetType.GENERATION,
            id="support-hours-answer",
        ),
        payload={
            "original_text": answer,
            "corrected_text": "Customer support closes at 6 PM.",
        },
        idempotency_key="support-hours-correction",
    )
    matches = await manager.query(FeedbackQuery(target_id="support-hours-answer"))
    acknowledged = await manager.acknowledge(received.feedback_id)
    handled = await manager.mark_handled(acknowledged.feedback_id)
    resolved = await manager.resolve(
        handled.feedback_id,
        resolution={"applied_text": "Customer support closes at 6 PM."},
    )

    assert len(matches) == 1
    assert matches[0].payload["corrected_text"].endswith("6 PM.")
    assert resolved.status.value == "resolved"
    print(f"Original: {answer}")
    print(f"Correction: {matches[0].payload['corrected_text']}")
    print(f"Feedback status: {resolved.status.value}")


asyncio.run(main())
