"""Internal structured decisions made with the application's chat model."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable


def structured_decision(
    model: BaseChatModel, schema: Any, *, name: str
) -> Runnable[Any, Any]:
    """Return a structured-output runnable for an internal decision.

    Decisions are never streamed to callers: the `nostream` tag hides their
    tokens, and a non-streaming copy of the model avoids streaming the request,
    which LangChain does whenever a streaming callback handler is attached.

    Raises:
        NotImplementedError: If the model does not support structured output.
    """
    quiet = model.model_copy(update={"disable_streaming": True})
    return quiet.with_structured_output(schema).with_config(
        run_name=name, tags=["nostream"]
    )
