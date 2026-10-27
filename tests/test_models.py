"""Configuration and execution-selection models."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from agloom import (
    ArchitectureSpec,
    ConfigurationError,
    ExecutionBlueprint,
    ExecutionDecision,
    ExecutionLimits,
    ExecutionMode,
    PipelineStage,
    SuggestedStage,
    SuggestedWorker,
    TaskAnalysis,
    TopologyName,
    WorkerSpec,
)
from agloom.models import parse_composition, parse_pattern, parse_topology

TOPOLOGIES = st.sampled_from(list(TopologyName))
WORKER = WorkerSpec(name="researcher", description="Finds facts.")


class TestExecutionLimits:
    def test_defaults_disable_recursion(self) -> None:
        limits = ExecutionLimits()
        assert not limits.recursion_enabled
        assert limits.allowed_nested_topologies == tuple(TopologyName)

    def test_positive_depth_enables_recursion(self) -> None:
        assert ExecutionLimits(max_depth=2).recursion_enabled

    @pytest.mark.parametrize(
        "field",
        ["max_depth", "max_children", "max_workers", "max_plan_steps", "max_handoffs"],
    )
    def test_rejects_out_of_range_integers(self, field: str) -> None:
        with pytest.raises(ValidationError):
            ExecutionLimits.model_validate({field: 10_000})

    def test_rejects_empty_and_duplicate_nested_topologies(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            ExecutionLimits(allowed_nested_topologies=())
        with pytest.raises(ValidationError, match="must be unique"):
            ExecutionLimits(
                allowed_nested_topologies=(TopologyName.REACT, TopologyName.REACT)
            )

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionLimits().max_depth = 3  # type: ignore[misc]


class TestWorkersAndStages:
    def test_worker_text_is_stripped(self) -> None:
        worker = WorkerSpec(name="  a  ", description=" b ", instructions=" c ")
        assert (worker.name, worker.description, worker.instructions) == ("a", "b", "c")

    def test_worker_rejects_blank_text(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            WorkerSpec(name="   ", description="scope")

    def test_stage_rejects_blank_text(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            PipelineStage(name="draft", instruction="   ")

    def test_stage_keeps_transform(self) -> None:
        stage = PipelineStage(
            name=" draft ", instruction=" Write ", transform=str.upper
        )
        assert stage.name == "draft"
        assert stage.transform is not None
        assert stage.transform("x") == "X"


class TestBlueprintAndAnalysis:
    def test_blueprint_rejects_duplicate_workers_and_stages(self) -> None:
        worker = SuggestedWorker(name="w", description="d")
        stage = SuggestedStage(name="s", instruction="i")
        with pytest.raises(ValidationError, match="worker names must be unique"):
            ExecutionBlueprint(workers=(worker, worker))
        with pytest.raises(ValidationError, match="stage names must be unique"):
            ExecutionBlueprint(stages=(stage, stage))

    def test_blueprint_validates_composition(self) -> None:
        with pytest.raises(ValidationError, match="two to four"):
            ExecutionBlueprint(composition=(TopologyName.REACT,))
        blueprint = ExecutionBlueprint(
            composition=(TopologyName.PLANNER, TopologyName.REFLECTION)
        )
        assert blueprint.composition[0] is TopologyName.PLANNER

    def test_analysis_normalizes_missing_capabilities(self) -> None:
        analysis = TaskAnalysis(
            complexity=1,
            estimated_steps=1,
            missing_capabilities=(" email ", "email", "sms"),
        )
        assert analysis.missing_capabilities == ("email", "sms")

    def test_analysis_rejects_blank_missing_capability(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            TaskAnalysis(complexity=1, estimated_steps=1, missing_capabilities=(" ",))


class TestExecutionDecision:
    def test_topology_mode_requires_topology(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionDecision(mode=ExecutionMode.TOPOLOGY, reason="r")
        with pytest.raises(ValidationError):
            ExecutionDecision(
                mode=ExecutionMode.DIRECT, topology=TopologyName.REACT, reason="r"
            )

    def test_valid_decisions(self) -> None:
        assert ExecutionDecision(mode=ExecutionMode.DIRECT, reason="r").topology is None
        decision = ExecutionDecision(
            mode=ExecutionMode.TOPOLOGY, topology=TopologyName.REACT, reason="r"
        )
        assert decision.topology is TopologyName.REACT


class TestArchitectureSpec:
    @pytest.mark.parametrize(
        "topology",
        [TopologyName.SUPERVISOR, TopologyName.SWARM, TopologyName.BLACKBOARD],
    )
    def test_worker_topologies_require_workers(self, topology: TopologyName) -> None:
        with pytest.raises(ValidationError, match="requires workers"):
            ArchitectureSpec(topology=topology)
        assert ArchitectureSpec(topology=topology, workers=(WORKER,)).workers

    def test_composition_only_for_hybrid(self) -> None:
        with pytest.raises(ValidationError, match="only valid for the Hybrid"):
            ArchitectureSpec(
                topology=TopologyName.REACT,
                composition=(TopologyName.REACT, TopologyName.PLANNER),
            )

    def test_hybrid_with_worker_child_requires_workers(self) -> None:
        composition = (TopologyName.PLANNER, TopologyName.SUPERVISOR)
        with pytest.raises(ValidationError, match="requires workers"):
            ArchitectureSpec(topology=TopologyName.HYBRID, composition=composition)
        spec = ArchitectureSpec(
            topology=TopologyName.HYBRID, composition=composition, workers=(WORKER,)
        )
        assert spec.composition == composition

    def test_cache_key_is_hashable_and_distinguishes_transforms(self) -> None:
        first = ArchitectureSpec(
            topology=TopologyName.PIPELINE,
            stages=(PipelineStage(name="a", instruction="b", transform=str.upper),),
        )
        second = ArchitectureSpec(
            topology=TopologyName.PIPELINE,
            stages=(PipelineStage(name="a", instruction="b", transform=str.lower),),
        )
        assert first.cache_key != second.cache_key
        assert hash(first.cache_key) == hash(first.model_copy().cache_key)


class TestParsing:
    def test_parse_topology_accepts_names_and_members(self) -> None:
        assert parse_topology(" Planner ") is TopologyName.PLANNER
        assert parse_topology(TopologyName.SWARM) is TopologyName.SWARM

    @pytest.mark.parametrize("value", ["direct", "unknown", 3])
    def test_parse_topology_rejects_unknown_values(self, value: object) -> None:
        with pytest.raises(ConfigurationError, match="unknown topology"):
            parse_topology(value)  # type: ignore[arg-type]

    def test_parse_pattern(self) -> None:
        assert parse_pattern(None) is None
        assert parse_pattern("react") == (TopologyName.REACT,)
        assert parse_pattern(["react", TopologyName.PLANNER]) == (
            TopologyName.REACT,
            TopologyName.PLANNER,
        )

    @pytest.mark.parametrize(
        ("pattern", "message"),
        [
            ([], "must not be an empty sequence"),
            (["react", "react"], "must be unique"),
            (42, "topology name or a sequence"),
        ],
    )
    def test_parse_pattern_rejects_invalid(self, pattern: object, message: str) -> None:
        with pytest.raises(ConfigurationError, match=message):
            parse_pattern(pattern)  # type: ignore[arg-type]

    def test_parse_composition(self) -> None:
        assert parse_composition(None) == ()
        assert parse_composition(["react", "reflection"]) == (
            TopologyName.REACT,
            TopologyName.REFLECTION,
        )

    @pytest.mark.parametrize(
        ("composition", "message"),
        [
            ("react", "sequence of topology names"),
            (["react"], "two to four"),
            (["react", "hybrid"], "cannot contain Hybrid"),
            (["react", "react"], "must be unique"),
        ],
    )
    def test_parse_composition_rejects_invalid(
        self, composition: object, message: str
    ) -> None:
        with pytest.raises(ConfigurationError, match=message):
            parse_composition(composition)  # type: ignore[arg-type]


@given(st.lists(TOPOLOGIES, min_size=1, max_size=8, unique=True))
def test_unique_patterns_round_trip(names: list[TopologyName]) -> None:
    assert parse_pattern([name.value.upper() for name in names]) == tuple(names)


@given(st.lists(TOPOLOGIES, min_size=0, max_size=6))
def test_composition_is_valid_exactly_when_rules_hold(
    names: list[TopologyName],
) -> None:
    valid = (
        2 <= len(names) <= 4
        and TopologyName.HYBRID not in names
        and len(set(names)) == len(names)
    )
    if valid:
        assert parse_composition(names) == tuple(names)
    else:
        with pytest.raises(ConfigurationError):
            parse_composition(names)
