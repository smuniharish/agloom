"""Helpers shared by the built-in topology builders."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from agloom._runtime import current_run
from agloom.errors import ExecutionError
from agloom.models import WorkerSpec

_CONVERSATION_MESSAGES = 10
_MESSAGE_CHARS = 2_000

State = Mapping[str, Any]


def node(
    name: str,
    build: Callable[[State], Any],
    runnable: Runnable[Any, Any],
    finish: Callable[[State, Any], dict[str, Any]],
    *,
    skip: Callable[[State], dict[str, Any] | None] | None = None,
) -> RunnableLambda[Any, Any]:
    """Return a graph node that maps state through one runnable, sync or async.

    When `skip` returns an update, the node returns it without calling the runnable.
    """

    def run(state: State, config: RunnableConfig) -> dict[str, Any]:
        ensure_active(config)
        if skip is not None and (update := skip(state)) is not None:
            return update
        return finish(state, runnable.invoke(build(state), config))

    async def arun(state: State, config: RunnableConfig) -> dict[str, Any]:
        ensure_active(config)
        if skip is not None and (update := skip(state)) is not None:
            return update
        return finish(state, await runnable.ainvoke(build(state), config))

    return RunnableLambda(run, afunc=arun, name=name)


def ensure_active(config: RunnableConfig | None) -> None:
    """Raise when the current invocation was cooperatively cancelled."""
    run = current_run(config)
    if run is not None:
        run.check_cancelled()


def model_input(system: str | None, prompt: str) -> list[BaseMessage]:
    """Return the messages for one plain model call."""
    messages: list[BaseMessage] = [SystemMessage(content=system)] if system else []
    messages.append(HumanMessage(content=prompt))
    return messages


def agent_input(prompt: str) -> dict[str, Any]:
    """Return the input of a LangChain agent for one prompt."""
    return {"messages": [HumanMessage(content=prompt)]}


def agent_text(result: Mapping[str, Any]) -> str:
    """Return the final text answer of a LangChain agent run."""
    message = final_message(result)
    return message.text if message is not None else ""


def final_message(value: Any) -> AIMessage | None:
    """Return the final answer message of a model call or a LangChain agent run."""
    if isinstance(value, AIMessage):
        return value
    if isinstance(value, Mapping):
        for message in reversed(value.get("messages") or ()):
            if isinstance(message, AIMessage) and message.text:
                return message
    return None


def answer_update(
    value: Any, transform: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """Return the `output` update for a final answer, keeping its loop signals.

    Loop signals travel in the answer message's `additional_kwargs["loop"]`;
    every topology preserves them so Loop Engineering observes them.
    """
    message = final_message(value)
    text = message.text if message is not None else ""
    signals = message.additional_kwargs.get("loop") if message is not None else None
    return {
        "output": transform(text) if transform is not None else text,
        "loop": dict(signals) if isinstance(signals, Mapping) else {},
    }


def structured_response(result: Mapping[str, Any]) -> Any:
    """Return the structured response of a LangChain agent run."""
    response = result.get("structured_response")
    if response is None:
        raise ExecutionError("the agent returned no structured response")
    return response


def request(state: State, *sections: tuple[str, str | None]) -> str:
    """Compose the prompt for one node from the request, sections, and loop context."""
    parts: list[str] = []
    history = conversation(state.get("messages") or ())
    if history:
        parts.append(f"Conversation so far:\n{history}")
    parts.append(f"Request:\n{state['task']}")
    parts.extend(f"{title}:\n{body}" for title, body in sections if body)
    context = state.get("context")
    if context:
        parts.append(context)
    return "\n\n".join(parts)


def conversation(messages: Iterable[BaseMessage]) -> str:
    """Render the bounded conversation before the latest user message."""
    prior = list(messages)[:-1][-_CONVERSATION_MESSAGES:]
    lines = []
    for message in prior:
        if message.type in {"human", "ai"} and message.text:
            role = "User" if message.type == "human" else "Assistant"
            lines.append(f"{role}: {message.text[:_MESSAGE_CHARS]}")
    return "\n".join(lines)


def node_name(prefix: str, index: int, name: str) -> str:
    """Return a unique, readable graph node name."""
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower() or prefix
    return f"{prefix}_{index}_{safe}"


def worker_instruction(worker: WorkerSpec) -> str:
    """Return the system instruction of one worker."""
    instruction = f"You are {worker.name}. Your scope: {worker.description}."
    if worker.instructions:
        instruction = f"{instruction}\n{worker.instructions}"
    return instruction


def choices(values: Sequence[str]) -> Any:
    """Return a `Literal` type restricted to values known only at runtime."""
    subscript = getattr(Literal, "__getitem__")  # noqa: B009 - runtime-built Literal
    return subscript(tuple(values))


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge reducer for dictionaries written by parallel branches."""
    return {**left, **right}


def transcript(entries: Sequence[Mapping[str, str]]) -> str:
    """Render worker contributions in execution order."""
    return "\n\n".join(f"[{entry['author']}] {entry['text']}" for entry in entries)


def with_context(
    messages: Sequence[BaseMessage], context: str | None
) -> list[BaseMessage]:
    """Append loop context to the latest user message of a conversation."""
    resolved = list(messages)
    if not context:
        return resolved
    for index in range(len(resolved) - 1, -1, -1):
        message = resolved[index]
        if message.type == "human":
            resolved[index] = HumanMessage(
                content=f"{message.text}\n\n{context}", id=message.id
            )
            return resolved
    resolved.append(HumanMessage(content=context))
    return resolved


def topology_input(
    react: bool, task: str, messages: Sequence[BaseMessage], context: str | None
) -> dict[str, Any]:
    """Return the input of a compiled topology graph."""
    if react:
        conversation_messages = list(messages) or [HumanMessage(content=task)]
        return {"messages": with_context(conversation_messages, context)}
    return {"task": task, "messages": list(messages), "context": context}


def topology_answer(react: bool, result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the `output` update, with loop signals, of a compiled topology graph."""
    if react:
        return answer_update(result)
    signals = result.get("loop")
    return {
        "output": str(result.get("output") or ""),
        "loop": dict(signals) if isinstance(signals, Mapping) else {},
    }
