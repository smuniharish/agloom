from __future__ import annotations

import os
import sys

from langchain_openai import ChatOpenAI

if sys.stdout.encoding and sys.stdout.encoding.casefold() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def real_model() -> ChatOpenAI:
    """Create the OpenAI-compatible model used by every runnable example."""

    return ChatOpenAI(
        model=os.getenv("EXPLABS_MODEL", "gpt-5.6-luna"),
        base_url=os.getenv(
            "EXPLABS_BASE_URL",
            "https://api.experientiallabs.ai/v1",
        ),
        api_key=os.environ["EXPLABS_API_KEY"],
        timeout=120,
        max_retries=2,
        use_responses_api=False,
    )


def verified_text(
    value: object,
    *required: str,
    any_of: tuple[str, ...] = (),
) -> str:
    """Return text only when a live example produced its required facts."""

    content = getattr(value, "content", value)
    text = str(content)
    missing = [item for item in required if item.casefold() not in text.casefold()]
    if missing:
        raise AssertionError(f"live response omitted required values: {missing}")
    if any_of and not any(item.casefold() in text.casefold() for item in any_of):
        raise AssertionError(
            f"live response omitted every required alternative: {list(any_of)}"
        )
    return text
