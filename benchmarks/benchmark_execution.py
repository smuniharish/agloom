from __future__ import annotations

import argparse
import tracemalloc
from collections.abc import Callable
from statistics import median
from time import perf_counter
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agloom import RecursionPolicy, WorkerSpec, create_agent
from agloom.compiler.compiler import ArchitectureCompiler

DIRECT_ANALYSIS = '{"complexity":0,"estimated_steps":1}'
PLANNER_ANALYSIS = '{"complexity":3,"estimated_steps":2,"requires_decomposition":true}'


class CountingCompiler(ArchitectureCompiler):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def compile(self, spec, config):
        self.calls += 1
        return super().compile(spec, config)


def _model(*responses: str) -> FakeListChatModel:
    return FakeListChatModel(responses=list(responses))


def _direct() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model(DIRECT_ANALYSIS, "Hello."),
        tools=[],
        compiler=compiler,
    )
    agent.invoke("Hi")
    return agent, compiler


def _automatic() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model(
            PLANNER_ANALYSIS,
            "1. Complete the request",
            "Completed.",
        ),
        tools=[],
        compiler=compiler,
    )
    agent.invoke("Plan how to complete one task.")
    return agent, compiler


def _planner() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model("1. Complete the request", "Completed."),
        tools=[],
        pattern="planner",
        compiler=compiler,
    )
    agent.invoke("Complete the request.")
    return agent, compiler


def _supervisor() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model("researcher", "Facts collected."),
        tools=[],
        pattern="supervisor",
        workers=[WorkerSpec(name="researcher", description="Collect facts.")],
        compiler=compiler,
    )
    agent.invoke("Collect facts.")
    return agent, compiler


def _multiple_workers() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model("Finding A.", "Finding B.", "Combined findings."),
        tools=[],
        pattern="blackboard",
        workers=[
            WorkerSpec(name="analyst_a", description="Analyze the first aspect."),
            WorkerSpec(name="analyst_b", description="Analyze the second aspect."),
        ],
        compiler=compiler,
    )
    agent.invoke("Analyze two aspects and combine findings.")
    return agent, compiler


def _hybrid() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model(
            "Analysis.", "Draft.", "Evaluation.", "Revision.", "Review.", "Final."
        ),
        tools=[],
        pattern="hybrid",
        composition=["pipeline", "reflection"],
        compiler=compiler,
    )
    agent.invoke("Analyze and revise the response.")
    return agent, compiler


def _recursive() -> tuple[Any, CountingCompiler]:
    compiler = CountingCompiler()
    agent = create_agent(
        model=_model(
            "1. Complete the child task",
            DIRECT_ANALYSIS,
            "Child complete.",
        ),
        tools=[],
        pattern="planner",
        recursion=True,
        recursion_policy=RecursionPolicy(max_depth=1, max_subtasks=2),
        compiler=compiler,
    )
    agent.invoke("Complete one task.")
    return agent, compiler


CASES: dict[str, Callable[[], tuple[Any, CountingCompiler]]] = {
    "DIRECT": _direct,
    "automatic": _automatic,
    "Planner": _planner,
    "Supervisor": _supervisor,
    "multiple workers": _multiple_workers,
    "Hybrid": _hybrid,
    "recursive": _recursive,
}


def benchmark(
    name: str, operation: Callable[[], tuple[Any, CountingCompiler]], count: int
):
    durations = []
    compile_calls = 0
    tracemalloc.start()
    for _ in range(count):
        started = perf_counter()
        _, compiler = operation()
        durations.append(perf_counter() - started)
        compile_calls += compiler.calls
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{name:12} median_ms={median(durations) * 1000:.3f} "
        f"peak_kib={peak / 1024:.1f} topology_compiles={compile_calls}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    for name, operation in CASES.items():
        benchmark(name, operation, args.iterations)


if __name__ == "__main__":
    main()
