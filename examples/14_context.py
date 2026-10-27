"""Context management: ContextSage summarizes long histories without losing facts.

`contextsage.IntelligentSummarizationMiddleware` replaces LangChain's
summarization middleware with an information-aware one: it compacts structured
content, writes a summary, and verifies that must-keep facts (IDs, numbers,
addresses) survived. Pass it in `middleware=`; Agloom applies middleware to
every agent it builds. The hook receives a content-free event per rewrite.

Run:
    uv run python examples/14_context.py
"""

from _support import real_model, verified_text
from contextsage import IntelligentSummarizationMiddleware, SummarizationEvent

from agloom import create_agent

CONVERSATION = [
    ("user", "Hi, I'm Dana. Order ORD-48213 arrived with a cracked screen."),
    ("assistant", "Sorry, Dana. I opened case CASE-7781 for order ORD-48213."),
    ("user", "The courier was TransFast, tracking number TF-55120."),
    ("assistant", "Thanks, I added tracking number TF-55120 to the case."),
    ("user", "I would prefer a replacement over a refund."),
    ("assistant", "Understood, we will send a replacement."),
    ("user", "Please ship it to my office at 12 Harbor Street, Leeds."),
    ("assistant", "Noted: the replacement ships to 12 Harbor Street, Leeds."),
    (
        "user",
        "Summarize my case in one sentence: the order number, the case number, "
        "and where the replacement ships.",
    ),
]


def main() -> None:
    model = real_model()
    rewrites: list[SummarizationEvent] = []
    summarizer = IntelligentSummarizationMiddleware(
        model,
        trigger=("messages", 6),
        keep=("messages", 2),
        observability_hook=rewrites.append,
    )
    agent = create_agent(model=model, pattern="react", middleware=[summarizer])

    answer = agent.invoke(CONVERSATION)

    assert rewrites, "the history should have been summarized"
    event = rewrites[-1]
    print(verified_text(answer, "ORD-48213", "CASE-7781", "Harbor"))
    print(
        f"Summarized {event.input_messages} -> {event.output_messages} messages; "
        f"must-preserve facts: {event.must_preserve_facts}; "
        f"validation: {event.validation_status}"
    )


if __name__ == "__main__":
    main()
