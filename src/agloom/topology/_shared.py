"""Shared helpers used by the built-in LangGraph topology builders."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END
from langgraph.graph import StateGraph as StateGraphType

from agloom.errors import (
    CancellationError,
    ConfigurationError,
    ExecutionError,
    RecursionError,
)
from agloom.models import PipelineStage, TopologyName, WorkerSpec
from agloom.topology.base import ExecutionState, TopologyBuildContext


def compile_graph(context: TopologyBuildContext, builder: StateGraphType[Any]) -> Any:
    return builder.compile(
        checkpointer=context.checkpointer,
        interrupt_before=list(context.interrupt_before) or None,
        interrupt_after=list(context.interrupt_after) or None,
    )


def model_messages(
    instruction: str,
    task: str,
    previous: Any = None,
    *,
    system_prompt: str | None = None,
) -> list[Any]:
    if system_prompt:
        instruction = f"{system_prompt}\n\n{instruction}"
    messages: list[Any] = [SystemMessage(content=instruction)]
    if previous is not None:
        messages.append(SystemMessage(content=f"Prior result to consider:\n{previous}"))
    messages.append(HumanMessage(content=task))
    return messages


def model_text(
    context: TopologyBuildContext, instruction: str, state: ExecutionState
) -> AIMessage:
    return context.model.invoke(
        model_messages(
            instruction,
            state["task"],
            state.get("output"),
            system_prompt=context.system_prompt,
        )
    )


async def amodel_text(
    context: TopologyBuildContext, instruction: str, state: ExecutionState
) -> AIMessage:
    return await context.model.ainvoke(
        model_messages(
            instruction,
            state["task"],
            state.get("output"),
            system_prompt=context.system_prompt,
        )
    )


def node(
    sync_fn: Callable[[ExecutionState, RunnableConfig], dict[str, Any]],
    async_fn: Callable[[ExecutionState, RunnableConfig], Any],
) -> RunnableLambda:
    def checked(state: ExecutionState, config: RunnableConfig):
        check_cancelled(config)
        return sync_fn(state, config)

    async def achecked(state: ExecutionState, config: RunnableConfig):
        check_cancelled(config)
        return await async_fn(state, config)

    return RunnableLambda(checked, afunc=achecked)


def check_cancelled(config: RunnableConfig) -> None:
    event = config.get("configurable", {}).get("agloom_cancel_event")
    if event is not None and event.is_set():
        raise CancellationError("execution was cooperatively cancelled")


def parse_subtasks(response: AIMessage) -> list[str]:
    lines = []
    for line in str(response.content).splitlines():
        normalized = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if normalized:
            lines.append(normalized)
    if not lines:
        raise ExecutionError("planner produced no actionable subtasks")
    if len(lines) > 16:
        raise ExecutionError("planner exceeded the 16-subtask limit")
    return lines


def run_subtask(context: TopologyBuildContext, task: str) -> Any:
    return context.model.invoke(
        model_messages(
            "Complete this planned subtask precisely.",
            task,
            system_prompt=context.system_prompt,
        )
    ).content


def child_callback(config: RunnableConfig, *, async_mode: bool):
    configurable = config.get("configurable", {})
    callback_name = "agloom_achild_executor" if async_mode else "agloom_child_executor"
    callback = configurable.get(callback_name)
    if not callable(callback):
        raise RecursionError("recursive child executor was not configured")
    return callback


def require_workers(workers: tuple[WorkerSpec, ...], name: TopologyName) -> None:
    if not workers:
        raise ConfigurationError(
            f"{name.value} topology requires at least one WorkerSpec"
        )


def worker_instruction(worker: WorkerSpec) -> str:
    return (
        f"You are worker {worker.name!r}. Scope: {worker.description}. "
        f"Instructions: {worker.instructions or 'complete your assigned task'}."
    )


def worker_by_name(workers: tuple[WorkerSpec, ...], name: str) -> WorkerSpec:
    for worker in workers:
        if worker.name == name:
            return worker
    raise ExecutionError(f"supervisor selected unknown worker {name!r}")


def choose_worker(context: TopologyBuildContext, task: str) -> WorkerSpec:
    prompt = (
        "Select exactly one worker by its exact name to handle the task. "
        "Return only that name. Workers: "
        + "; ".join(
            f"{worker.name}: {worker.description}" for worker in context.workers
        )
    )
    response = context.model.invoke(
        model_messages(prompt, task, system_prompt=context.system_prompt)
    )
    return parse_worker(context.workers, str(response.content))


async def async_choose_worker(context: TopologyBuildContext, task: str) -> WorkerSpec:
    prompt = (
        "Select exactly one worker by its exact name to handle the task. "
        "Return only that name. Workers: "
        + "; ".join(
            f"{worker.name}: {worker.description}" for worker in context.workers
        )
    )
    response = await context.model.ainvoke(
        model_messages(prompt, task, system_prompt=context.system_prompt)
    )
    return parse_worker(context.workers, str(response.content))


def parse_worker(workers: tuple[WorkerSpec, ...], answer: str) -> WorkerSpec:
    choices = [worker for worker in workers if answer.strip() == worker.name]
    if len(choices) != 1:
        raise ExecutionError("supervisor did not return exactly one worker name")
    return choices[0]


def swarm_route(
    state: ExecutionState,
    config: RunnableConfig,
    workers: tuple[WorkerSpec, ...],
    ids: dict[str, str],
) -> str:
    max_handoffs = int(config.get("configurable", {}).get("agloom_max_handoffs", 2))
    if state.get("handoff_count", 0) >= max_handoffs:
        return END
    match = re.search(r"(?m)^\s*NEXT:\s*(.+?)\s*$", str(state.get("output", "")))
    if match is None:
        raise ExecutionError("swarm peer omitted its NEXT routing directive")
    target = match.group(1).strip()
    if target == "STOP":
        return END
    if target not in ids or target not in {worker.name for worker in workers}:
        raise ExecutionError(f"swarm peer selected unknown worker {target!r}")
    return ids[target]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_") or "stage"


def child_input(state: ExecutionState) -> dict[str, Any]:
    task = state["task"]
    if state.get("output") is not None:
        task = f"{task}\n\nPrior topology result:\n{state['output']}"
    return {"task": task, "messages": [HumanMessage(content=task)]}


def extract_output(result: Any) -> Any:
    if isinstance(result, dict):
        if "output" in result:
            return result["output"]
        messages = result.get("messages")
        if messages:
            return getattr(messages[-1], "content", messages[-1])
    return result


def without_checkpointer_thread(config: RunnableConfig) -> RunnableConfig:
    copied = cast(RunnableConfig, dict(config))
    raw_configurable = copied.get("configurable")
    configurable: dict[str, Any] = (
        dict(raw_configurable) if isinstance(raw_configurable, dict) else {}
    )
    configurable.pop("thread_id", None)
    copied["configurable"] = configurable
    return cast(RunnableConfig, copied)


def default_stage(name: str, instruction: str) -> PipelineStage:
    return PipelineStage(name=name, instruction=instruction)
