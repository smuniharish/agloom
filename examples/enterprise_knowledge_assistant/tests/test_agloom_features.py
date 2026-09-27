import asyncio

from examples.agloom_feature_regression import (
    assert_all_execution_modes,
    assert_all_integration_combinations,
    assert_complete_parameter_surface,
    assert_semantic_runtime_features,
)


def test_complete_create_agent_surface() -> None:
    assert_complete_parameter_surface("enterprise-knowledge-assistant")


def test_all_integration_combinations() -> None:
    assert_all_integration_combinations()


def test_all_execution_modes() -> None:
    assert_all_execution_modes()


def test_semantic_runtime_features() -> None:
    asyncio.run(assert_semantic_runtime_features("enterprise-knowledge-assistant"))
