"""Measure execution overhead with deterministic models (no network).

Run from the repository root: `uv run python benchmarks/benchmark_execution.py`.

* cold: create the agent and run one request (comparable to earlier releases)
* warm: reuse one agent for every request (the production pattern)
"""

from __future__ import annotations

import argparse
import sys
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from agloom import (
    Agent,
    ArchitectureCompiler,
    ArchitectureSpec,
    ExecutionLimits,
    TopologyArtifact,
    TopologyBuildContext,
    WorkerSpec,
    create_agent,
)
from helpers import Call, ScriptedChatModel, analysis, tool_call

PLAN = {"steps": [{"id": "a", "task": "Complete the request"}]}
WORKERS = [
    WorkerSpec(name="analyst_a", description="Analyze the first aspect."),
    WorkerSpec(name="analyst_b", description="Analyze the second aspect."),
]


class CountingCompiler(ArchitectureCompiler):
    """Counts topology compilations."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def compile(
        self, spec: ArchitectureSpec, context: TopologyBuildContext
    ) -> TopologyArtifact:
        self.calls += 1
        return super().compile(spec, context)


def respond(call: Call) -> Any:
    if call.wants("TaskAnalysis"):
        if "Plan" in call.prompt:
            return analysis(
                complexity=3, estimated_steps=2, requires_decomposition=True
            )
        return analysis()
    if call.wants("Plan"):
        return tool_call("Plan", PLAN)
    if call.wants("SupervisorRoute"):
        if "[researcher]" in call.prompt:
            return tool_call(
                "SupervisorRoute", {"next": "FINISH", "final_answer": "Done."}
            )
        return tool_call(
            "SupervisorRoute", {"next": "researcher", "assignment": "Collect."}
        )
    if call.wants("Review"):
        return tool_call("Review", {"approved": True})
    return "Completed."


def factory(**options: Any) -> Callable[[CountingCompiler], Agent]:
    def build(compiler: CountingCompiler) -> Agent:
        return create_agent(
            model=ScriptedChatModel(responder=respond), compiler=compiler, **options
        )

    return build


CASES: dict[str, tuple[Callable[[CountingCompiler], Agent], str]] = {
    "DIRECT": (factory(), "Hi"),
    "automatic": (factory(), "Plan how to complete one task."),
    "Planner": (factory(pattern="planner"), "Complete the request."),
    "Supervisor": (
        factory(
            pattern="supervisor",
            workers=[WorkerSpec(name="researcher", description="Collect facts.")],
        ),
        "Collect facts.",
    ),
    "multiple workers": (factory(pattern="blackboard", workers=WORKERS), "Analyze."),
    "Hybrid": (
        factory(pattern="hybrid", composition=["pipeline", "reflection"]),
        "Analyze and revise the response.",
    ),
    "recursive": (
        factory(pattern="planner", limits=ExecutionLimits(max_depth=1, max_children=2)),
        "Complete one task.",
    ),
}


def measure(operation: Callable[[], None], count: int) -> tuple[float, float]:
    durations: list[float] = []
    tracemalloc.start()
    for _ in range(count):
        started = perf_counter()
        operation()
        durations.append(perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return median(durations) * 1000, peak / 1024


def benchmark(
    name: str, build: Callable[[CountingCompiler], Agent], task: str, count: int
) -> None:
    cold_compiler = CountingCompiler()
    cold_ms, cold_kib = measure(lambda: build(cold_compiler).invoke(task), count)
    warm_compiler = CountingCompiler()
    agent = build(warm_compiler)
    agent.invoke(task)
    warm_ms, warm_kib = measure(lambda: agent.invoke(task), count)
    print(
        f"{name:17} cold_ms={cold_ms:8.3f} warm_ms={warm_ms:8.3f} "
        f"peak_kib={max(cold_kib, warm_kib):8.1f} "
        f"compiles_cold={cold_compiler.calls} compiles_warm={warm_compiler.calls}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    for name, (build, task) in CASES.items():
        benchmark(name, build, task, args.iterations)


if __name__ == "__main__":
    main()
