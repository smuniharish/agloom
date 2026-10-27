"""Input normalization and result shaping for agent invocations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)

from agloom.errors import ConfigurationError

_TRUSTED_ROLES = frozenset({"human", "ai"})


def normalize_input(value: Any) -> tuple[str, list[BaseMessage], bool]:
    """Return the task, the conversation messages, and whether the input was a mapping.

    Accepts text, one message, a message sequence (LangChain messages,
    role/content mappings, or `(role, content)` pairs), or a mapping with
    `messages`, `task`, `input`, or `prompt`.

    Raises:
        ConfigurationError: If the input is empty, has an unsupported type, or
            contains system or tool messages. Trusted instructions belong in
            `system_prompt`.
    """
    if isinstance(value, str):
        return _task(value), [HumanMessage(content=value.strip())], False
    if isinstance(value, Mapping):
        if value.get("messages") is not None:
            messages = _messages(value["messages"])
            return _latest_task(messages), messages, True
        for key in ("task", "input", "prompt"):
            if value.get(key) is not None:
                text = _task(str(value[key]))
                return text, [HumanMessage(content=text)], True
        raise ConfigurationError(
            "input mappings require messages, task, input, or prompt"
        )
    if isinstance(value, BaseMessage) or (
        isinstance(value, Sequence) and not isinstance(value, bytes | bytearray)
    ):
        messages = _messages([value] if isinstance(value, BaseMessage) else value)
        return _latest_task(messages), messages, False
    raise ConfigurationError(
        f"unsupported input type {type(value).__name__}; "
        "expected text, messages, or a mapping"
    )


def shape_result(answer: AIMessage, original: Any, mapping_input: bool) -> Any:
    """Return the answer in the shape that matches the invocation input."""
    if not mapping_input:
        return answer
    if isinstance(original, Mapping) and original.get("messages") is not None:
        return {**original, "messages": [*_messages(original["messages"]), answer]}
    task, messages, _ = normalize_input(original)
    del task
    return {**dict(original), "messages": [*messages, answer]}


def answer_message(result: Any) -> AIMessage:
    """Return the answer of an iteration result as an `AIMessage`."""
    if isinstance(result, AIMessage):
        return result
    if isinstance(result, BaseMessage):
        return AIMessage(content=result.content)
    return AIMessage(content="" if result is None else str(result))


def _messages(raw: Any) -> list[BaseMessage]:
    try:
        messages = convert_to_messages(raw)
    except (TypeError, ValueError, NotImplementedError) as error:
        raise ConfigurationError(
            "messages must be LangChain messages, role/content mappings, "
            "or (role, content) pairs"
        ) from error
    for message in messages:
        if message.type not in _TRUSTED_ROLES:
            raise ConfigurationError(
                "input messages may contain only user and assistant messages; "
                "configure trusted instructions with system_prompt"
            )
    return messages


def _latest_task(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return _task(message.text)
    raise ConfigurationError("messages must contain at least one user message")


def _task(text: str) -> str:
    task = text.strip()
    if not task:
        raise ConfigurationError("the task must not be empty")
    return task
