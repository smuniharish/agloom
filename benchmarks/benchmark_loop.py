"""Measure Loop Engineering overhead with deterministic models (no network).

Run from the repository root: `uv run python benchmarks/benchmark_loop.py`.
"""

from __future__ import annotations

import argparse
import sys
import tracemalloc
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agloom import (
    Agent,
    DefaultLoopEvaluator,
    LoopPolicy,
    PipelineStage,
    create_agent,
)
from helpers import Call, ScriptedChatModel, allow_threads, analysis


def signals(text: str, **loop: Any) -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"loop": loop})


def counting_model(answer: Callable[[int], AIMessage]) -> ScriptedChatModel:
    calls = {"n": 0}

    def respond(call: Call) -> Any:
        if call.wants("TaskAnalysis"):
            return analysis()
        calls["n"] += 1
        return answer(calls["n"])

    return ScriptedChatModel(responder=respond)


def agent(
    policy: LoopPolicy, answer: Callable[[int], AIMessage], **options: Any
) -> Agent:
    return create_agent(
        model=counting_model(answer),
        pattern="pipeline",
        stages=[PipelineStage(name="work", instruction="Do the work.")],
        loop_policy=policy,
        loop_evaluator=DefaultLoopEvaluator(),
        **options,
    )


def eight(step: int) -> AIMessage:
    position = (step - 1) % 8 + 1
    return signals(
        "step", objective_progress=position / 8, objective_complete=position == 8
    )


def replan(step: int) -> AIMessage:
    if step % 2:
        return signals("first", replan_required=True)
    return signals("second", objective_complete=True)


ONE_SHOT = agent(LoopPolicy(), lambda step: AIMessage(content="done"))
EIGHT = agent(LoopPolicy(max_iterations=8, history_limit=3), eight)
REPLAN = agent(LoopPolicy(max_iterations=2, allow_replan=True, max_replans=1), replan)
CHECKPOINTED = agent(
    LoopPolicy(),
    lambda step: AIMessage(content="done"),
    checkpointer=InMemorySaver(),
    checkpoint_authorizer=allow_threads,
)


def one_shot() -> None:
    ONE_SHOT.invoke("complete once")


def eight_iterations() -> None:
    assert EIGHT.invoke_loop("complete eight steps").state.iteration == 8


def bounded_replan() -> None:
    assert REPLAN.invoke_loop("replan once").state.budget.replans == 1


def checkpointed(counter: list[int] = [0]) -> None:  # noqa: B006 - benchmark counter
    counter[0] += 1
    CHECKPOINTED.invoke("persist", {"configurable": {"thread_id": f"t{counter[0]}"}})


def concurrent() -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: ONE_SHOT.invoke("parallel"), range(16)))


CASES: dict[str, Callable[[], None]] = {
    "one-shot loop": one_shot,
    "eight iterations": eight_iterations,
    "bounded replan": bounded_replan,
    "checkpointed": checkpointed,
    "16 concurrent": concurrent,
}


def benchmark(name: str, operation: Callable[[], None], count: int) -> None:
    operation()
    durations: list[float] = []
    tracemalloc.start()
    for _ in range(count):
        started = perf_counter()
        operation()
        durations.append(perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{name:18} median_ms={median(durations) * 1000:8.3f} "
        f"peak_kib={peak / 1024:8.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    for name, operation in CASES.items():
        benchmark(name, operation, args.iterations)


if __name__ == "__main__":
    main()
