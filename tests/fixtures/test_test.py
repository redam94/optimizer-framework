from atlas.core.interfaces import OptimizationResult


def test_optimization_result():
    result = OptimizationResult(optimal_budget={"param1": 0.5, "param2": 1.5}, optimal_value=42.0)
    assert result.optimal_budget == {"param1": 0.5, "param2": 1.5}
    assert result.optimal_value == 42.0
