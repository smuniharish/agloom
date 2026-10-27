"""Deterministic execution selection."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agloom import (
    ExecutionMode,
    StrategyEngine,
    TaskAnalysis,
    TopologyName,
    TopologySelectionError,
)
from agloom.strategy.engine import preferred_topology, score

FLAGS = (
    "requires_tools",
    "requires_decomposition",
    "requires_parallelism",
    "requires_delegation",
    "requires_handoffs",
    "requires_shared_workspace",
    "requires_ordering",
    "requires_validation",
    "requires_composition",
)

analyses = st.builds(
    TaskAnalysis,
    complexity=st.integers(0, 10),
    estimated_steps=st.integers(1, 128),
    **{flag: st.booleans() for flag in FLAGS},
)
allow_lists = st.lists(
    st.sampled_from(list(TopologyName)), min_size=1, max_size=8, unique=True
)


def analysis(**flags: object) -> TaskAnalysis:
    return TaskAnalysis.model_validate({"complexity": 1, "estimated_steps": 1, **flags})


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"requires_composition": True, "requires_tools": True}, TopologyName.HYBRID),
        (
            {"requires_shared_workspace": True, "requires_handoffs": True},
            TopologyName.BLACKBOARD,
        ),
        ({"requires_handoffs": True, "requires_parallelism": True}, TopologyName.SWARM),
        (
            {"requires_parallelism": True, "requires_delegation": True},
            TopologyName.PLANNER,
        ),
        (
            {"requires_delegation": True, "requires_decomposition": True},
            TopologyName.SUPERVISOR,
        ),
        ({"requires_decomposition": True}, TopologyName.PLANNER),
        ({"requires_validation": True, "complexity": 4}, TopologyName.REFLECTION),
        ({"requires_ordering": True, "estimated_steps": 3}, TopologyName.PIPELINE),
        ({"requires_tools": True}, TopologyName.REACT),
        ({"requires_validation": True, "complexity": 3}, None),
        ({"requires_ordering": True, "estimated_steps": 2}, None),
        ({}, None),
    ],
)
def test_preferred_topology_priority(
    flags: dict[str, object], expected: TopologyName | None
) -> None:
    assert preferred_topology(analysis(**flags)) is expected


def test_direct_when_nothing_is_required() -> None:
    decision = StrategyEngine().decide(analysis())
    assert decision.mode is ExecutionMode.DIRECT
    assert decision.topology is None


def test_requirements_select_preferred_topology() -> None:
    decision = StrategyEngine().decide(analysis(requires_tools=True))
    assert decision.topology is TopologyName.REACT
    assert decision.reason == "selected from the task requirements"


def test_no_direct_and_no_requirement_fails_without_allow_list() -> None:
    with pytest.raises(TopologySelectionError, match="DIRECT is not allowed"):
        StrategyEngine().decide(analysis(), allow_direct=False)


def test_empty_allow_list_fails() -> None:
    with pytest.raises(TopologySelectionError, match="no topology candidates"):
        StrategyEngine().decide(analysis(), [])


def test_allow_list_containing_preferred() -> None:
    decision = StrategyEngine().decide(
        analysis(requires_tools=True), [TopologyName.PLANNER, TopologyName.REACT]
    )
    assert decision.topology is TopologyName.REACT
    assert "within the allow-list" in decision.reason


def test_allow_list_falls_back_to_best_score_then_order() -> None:
    engine = StrategyEngine()
    decision = engine.decide(
        analysis(requires_tools=True, requires_validation=True),
        [TopologyName.PIPELINE, TopologyName.REFLECTION],
    )
    assert decision.topology is TopologyName.REFLECTION
    tie = engine.decide(
        analysis(), [TopologyName.SWARM, TopologyName.PIPELINE], allow_direct=False
    )
    assert tie.topology is TopologyName.SWARM
    assert "highest-scoring" in tie.reason


def test_score_counts_satisfied_requirements() -> None:
    task = analysis(requires_decomposition=True, requires_parallelism=True)
    assert score(TopologyName.PLANNER, task) == 2
    assert score(TopologyName.REACT, task) == 0


@given(analyses, allow_lists, st.booleans())
def test_decisions_never_escape_the_allow_list(
    task: TaskAnalysis, candidates: list[TopologyName], allow_direct: bool
) -> None:
    decision = StrategyEngine().decide(task, candidates, allow_direct=allow_direct)
    if decision.mode is ExecutionMode.DIRECT:
        assert allow_direct
        assert preferred_topology(task) is None
    else:
        assert decision.topology in candidates


@given(analyses)
def test_direct_only_without_requirements(task: TaskAnalysis) -> None:
    decision = StrategyEngine().decide(task)
    assert (decision.mode is ExecutionMode.DIRECT) == (preferred_topology(task) is None)
    if decision.mode is ExecutionMode.TOPOLOGY:
        assert decision.topology is preferred_topology(task)
