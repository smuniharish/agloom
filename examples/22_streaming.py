"""Streaming: model tokens and Agloom events as they happen.

`agent.stream()` is LangGraph streaming. `stream_mode="messages"` yields model
tokens and `stream_mode="custom"` yields Agloom events
(`{"agloom_event": ...}`) such as loop progress and topology selection, from
every topology graph of the run. With several modes, every item is a
`(mode, chunk)` pair. Internal structured decisions are never streamed as
tokens.

Run:
    uv run python examples/22_streaming.py
"""

from _support import real_model
from langchain_core.messages import AIMessageChunk

from agloom import EventName, create_agent


def main() -> None:
    agent = create_agent(model=real_model(), pattern="react")

    tokens: list[str] = []
    events: list[str] = []
    for mode, chunk in agent.stream(
        "Name the three primary colors of paint in one short sentence.",
        stream_mode=["messages", "custom"],
    ):
        if mode == "messages":
            token, _metadata = chunk
            if isinstance(token, AIMessageChunk) and token.text:
                tokens.append(token.text)
                print(token.text, end="", flush=True)
        else:
            events.append(chunk["agloom_event"])
    print()

    text = "".join(tokens).casefold()
    assert len(tokens) > 1, "the answer should arrive as several tokens"
    assert all(color in text for color in ("red", "blue", "yellow")), text
    assert events[0] == EventName.LOOP_STARTED, events
    assert events[-1] == EventName.LOOP_COMPLETED, events
    print(f"{len(tokens)} tokens; events: {', '.join(events)}")


if __name__ == "__main__":
    main()
