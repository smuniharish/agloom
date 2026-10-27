"""Shared setup for the runnable examples.

Importing this module loads the repository's `.env` file (variables that are
already set win), switches standard output to UTF-8, and provides the live
model plus a helper that turns example answers into verified facts.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def real_model() -> ChatOpenAI:
    """Return the OpenAI-compatible chat model configured by `EXPLABS_*` variables."""
    missing = [
        name
        for name in ("EXPLABS_API_KEY", "EXPLABS_BASE_URL", "EXPLABS_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} in .env (see .env.example).")
    return ChatOpenAI(
        model=os.environ["EXPLABS_MODEL"],
        base_url=os.environ["EXPLABS_BASE_URL"],
        api_key=os.environ["EXPLABS_API_KEY"],
        use_responses_api=False,
        timeout=120,
        max_retries=2,
    )


def verified_text(answer: object, *facts: str | tuple[str, ...]) -> str:
    """Return the text of an answer after checking that it states every fact.

    A fact is a string or a tuple of alternative spellings, one of which must
    appear. Matching ignores case.

    Raises:
        AssertionError: If a fact is missing.
    """
    text = answer.text if isinstance(answer, BaseMessage) else str(answer)
    lowered = text.casefold()
    missing = [
        fact
        for fact in facts
        if not any(
            option.casefold() in lowered
            for option in ((fact,) if isinstance(fact, str) else fact)
        )
    ]
    if missing:
        raise AssertionError(f"the answer does not state {missing}:\n{text}")
    return text
