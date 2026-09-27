from examples.agloom_feature_regression import (
    assert_all_execution_modes,
    assert_all_integration_combinations,
    assert_complete_parameter_surface,
    assert_semantic_runtime_features,
)


def test_complete_parameter_surface() -> None:
    assert_complete_parameter_surface("engineering-change-investigator")


def test_all_integration_combinations() -> None:
    assert_all_integration_combinations()


def test_execution_modes() -> None:
    assert_all_execution_modes()


async def test_semantic_runtime_features() -> None:
    await assert_semantic_runtime_features("engineering-change-investigator")
