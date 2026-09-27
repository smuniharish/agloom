from examples.agloom_feature_regression import (
    assert_all_execution_modes,
    assert_all_integration_combinations,
    assert_complete_parameter_surface,
    assert_semantic_runtime_features,
)


def test_complete_parameter_surface():
    assert_complete_parameter_surface("web-compliance-auditor")


def test_all_integration_combinations():
    assert_all_integration_combinations()


def test_execution_modes():
    assert_all_execution_modes()


async def test_semantic_runtime_features():
    await assert_semantic_runtime_features("web-compliance-auditor")
