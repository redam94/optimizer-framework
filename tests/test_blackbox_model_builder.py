# tests/test_blackbox_model_builder.py
"""
Tests and examples for the Black Box Model Builder system.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from atlas.models.blackbox.model_builder import (
    BlackBoxModel,
    BlackBoxModelBuilder,
    BlackBoxModelConfig,
    BlackBoxModelConfigBuilder,
    DataVarType,
    OptimizationLayer,
    create_simple_model,
)


class TestBlackBoxModelConfig:
    """Test configuration builder functionality."""

    def test_basic_config_creation(self):
        """Test creating a basic configuration."""
        config = (
            BlackBoxModelConfigBuilder()
            .model_info(
                name="test_model", version="1.0.0", description="Test model", author="Test Author"
            )
            .add_required_var("input1", DataVarType.CONTINUOUS, bounds=(0, 100), units="units")
            .add_dimension("time", required=True)
            .add_output("output", dims=["time"])
            .build()
        )

        assert config.model_name == "test_model"
        assert config.model_version == "1.0.0"
        assert "input1" in config.data_vars
        assert config.data_vars["input1"].required
        assert "time" in config.dimensions
        assert "output" in config.outputs

    def test_optional_variables(self):
        """Test adding optional variables with defaults."""
        config = (
            BlackBoxModelConfigBuilder()
            .add_optional_var(
                "optional_var", DataVarType.CONTINUOUS, default_value=1.0, bounds=(0, 10)
            )
            .build()
        )

        assert "optional_var" in config.data_vars
        assert not config.data_vars["optional_var"].required
        assert config.data_vars["optional_var"].default_value == 1.0

    def test_optimization_layer_mapping(self):
        """Test mapping variables to optimization layers."""
        config = (
            BlackBoxModelConfigBuilder()
            .add_required_var("budget1", optimization_layer=OptimizationLayer.BUDGET_ALLOCATION)
            .add_required_var("budget2", optimization_layer=OptimizationLayer.BUDGET_ALLOCATION)
            .add_required_var("constraint1", optimization_layer=OptimizationLayer.CONSTRAINT_LAYER)
            .map_budget_allocation(["budget1", "budget2"])
            .build()
        )

        budget_vars = config.optimization_mappings.get(OptimizationLayer.BUDGET_ALLOCATION, [])
        assert "budget1" in budget_vars
        assert "budget2" in budget_vars

    def test_config_serialization(self):
        """Test saving and loading configuration."""
        config = (
            BlackBoxModelConfigBuilder()
            .model_info("test_model", "1.0.0")
            .add_required_var("var1", DataVarType.CONTINUOUS)
            .add_dimension("dim1")
            .add_output("out1", dims=["dim1"])
            .build()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test YAML
            yaml_path = Path(tmpdir) / "config.yaml"
            config.save(yaml_path, format="yaml")
            loaded_config = BlackBoxModelConfig.load(yaml_path)
            assert loaded_config.model_name == config.model_name

            # Test JSON
            json_path = Path(tmpdir) / "config.json"
            config.save(json_path, format="json")
            loaded_config = BlackBoxModelConfig.load(json_path)
            assert loaded_config.model_name == config.model_name


class TestBlackBoxModel:
    """Test black box model functionality."""

    def create_test_model(self):
        """Create a test model for use in tests."""

        def predict_fn(data: xr.Dataset) -> xr.DataArray:
            return xr.DataArray(
                data["x1"].values + 2 * data["x2"].values, dims=data["x1"].dims, name="output"
            )

        config = (
            BlackBoxModelConfigBuilder()
            .model_info("test_model", "1.0.0")
            .add_required_var("x1", DataVarType.CONTINUOUS, bounds=(0, 10))
            .add_required_var("x2", DataVarType.CONTINUOUS, bounds=(0, 10))
            .add_dimension("time")
            .add_output("output", dims=["time"])
            .map_budget_allocation(["x1", "x2"])
            .build()
        )

        return BlackBoxModel(predict_fn, config)

    def test_model_prediction(self):
        """Test basic model prediction."""
        model = self.create_test_model()

        data = xr.Dataset(
            {
                "x1": xr.DataArray([1, 2, 3], dims=["time"]),
                "x2": xr.DataArray([4, 5, 6], dims=["time"]),
                "time": [0, 1, 2],
            }
        )

        predictions = model.predict(data)
        expected = np.array([9, 12, 15])  # x1 + 2*x2
        np.testing.assert_array_equal(predictions.values, expected)

    def test_input_validation(self):
        """Test input validation."""
        model = self.create_test_model()

        # Missing required variable
        invalid_data = xr.Dataset({"x1": xr.DataArray([1, 2, 3], dims=["time"]), "time": [0, 1, 2]})

        with pytest.raises(ValueError, match="Missing required variables"):
            model.predict(invalid_data)

        # Values outside bounds
        invalid_data = xr.Dataset(
            {
                "x1": xr.DataArray([1, 2, 3], dims=["time"]),
                "x2": xr.DataArray([4, 5, 11], dims=["time"]),  # 11 > 10
                "time": [0, 1, 2],
            }
        )

        with pytest.raises(ValueError, match="outside bounds"):
            model.predict(invalid_data)

    def test_default_values(self):
        """Test handling of optional variables with defaults."""

        def predict_fn(data: xr.Dataset) -> xr.DataArray:
            return xr.DataArray(
                data["x1"].values * data["multiplier"].values, dims=data["x1"].dims, name="output"
            )

        config = (
            BlackBoxModelConfigBuilder()
            .add_required_var("x1", DataVarType.CONTINUOUS)
            .add_optional_var("multiplier", DataVarType.CONTINUOUS, default_value=2.0)
            .add_dimension("time")
            .add_output("output", dims=["time"])
            .build()
        )

        model = BlackBoxModel(predict_fn, config)

        # Without optional variable - should use default
        data = xr.Dataset({"x1": xr.DataArray([1, 2, 3], dims=["time"]), "time": [0, 1, 2]})

        predictions = model.predict(data)
        expected = np.array([2, 4, 6])  # x1 * 2.0 (default)
        np.testing.assert_array_equal(predictions.values, expected)

    def test_contributions(self):
        """Test contribution calculation."""
        model = self.create_test_model()

        data = xr.Dataset(
            {
                "x1": xr.DataArray([5], dims=["time"]),
                "x2": xr.DataArray([3], dims=["time"]),
                "time": [0],
            }
        )

        contributions = model.contributions(data)

        # Should have contributions for budget allocation variables
        assert "x1" in contributions
        assert "x2" in contributions

        # Contributions should sum to prediction
        prediction = model.predict(data)
        total_contribution = sum(float(contributions[var]) for var in ["x1", "x2"])
        assert abs(total_contribution - float(prediction)) < 1e-6

    def test_custom_contribution_function(self):
        """Test using a custom contribution function."""

        def predict_fn(data: xr.Dataset) -> xr.DataArray:
            return xr.DataArray([100], dims=["contribution"], name="output")

        def contribution_fn(data: xr.Dataset) -> xr.Dataset:
            return xr.Dataset(
                {
                    "x1": xr.DataArray([30], dims=["contribution"]),
                    "x2": xr.DataArray([70], dims=["contribution"]),
                }
            )

        config = BlackBoxModelConfigBuilder().add_required_var("x1").add_required_var("x2").build()

        model = BlackBoxModel(predict_fn, config, contribution_fn=contribution_fn)

        data = xr.Dataset({"x1": xr.DataArray([1]), "x2": xr.DataArray([1])})

        contributions = model.contributions(data)
        assert float(contributions["x1"]) == 30
        assert float(contributions["x2"]) == 70


class TestBlackBoxModelBuilder:
    """Test the model builder functionality."""

    def test_builder_pattern(self):
        """Test building a model using the builder pattern."""

        def predict_fn(data: xr.Dataset) -> xr.DataArray:
            return xr.DataArray([42], dims=["output"])

        model = (
            BlackBoxModelBuilder().set_name("test_model").set_predict_function(predict_fn).build()
        )

        assert model.name == "test_model"
        assert model.model_type == "blackbox_test_model"

    def test_builder_validation(self):
        """Test builder validation."""
        builder = BlackBoxModelBuilder()

        # Should fail without prediction function
        with pytest.raises(ValueError, match="Prediction function must be set"):
            builder.build()


class TestRealWorldExamples:
    """Test real-world usage patterns."""

    def test_marketing_mix_model(self):
        """Test a marketing mix model with saturation effects."""

        def mmm_predict(data: xr.Dataset) -> xr.DataArray:
            # Adstock transformation
            def adstock(spend, decay=0.7):
                result = np.zeros_like(spend)
                result[0] = spend[0]
                for i in range(1, len(spend)):
                    result[i] = spend[i] + decay * result[i - 1]
                return result

            # Saturation transformation (Hill transformation)
            def saturation(x, alpha=2.0, gamma=0.5):
                return x**alpha / (x**alpha + gamma**alpha)

            # Apply transformations
            tv_transformed = saturation(adstock(data["tv_spend"].values))
            digital_transformed = saturation(adstock(data["digital_spend"].values, decay=0.5))

            # Base sales + media effects
            base_sales = 1000
            revenue = base_sales + 0.3 * tv_transformed + 0.5 * digital_transformed

            return xr.DataArray(revenue, dims=data["tv_spend"].dims, name="revenue")

        # Build configuration
        config = (
            BlackBoxModelConfigBuilder()
            .model_info(
                name="advanced_mmm",
                version="2.0.0",
                description="Marketing mix model with adstock and saturation",
            )
            .add_required_var(
                "tv_spend",
                DataVarType.CONTINUOUS,
                bounds=(0, 1_000_000),
                units="USD",
                optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            )
            .add_required_var(
                "digital_spend",
                DataVarType.CONTINUOUS,
                bounds=(0, 1_000_000),
                units="USD",
                optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            )
            .add_dimension("week", required=True, coordinate_type="time")
            .add_output("revenue", dims=["week"], units="USD")
            .map_budget_allocation(["tv_spend", "digital_spend"])
            .add_metadata("adstock_decay_tv", 0.7)
            .add_metadata("adstock_decay_digital", 0.5)
            .add_metadata("saturation_alpha", 2.0)
            .add_metadata("saturation_gamma", 0.5)
            .build()
        )

        model = BlackBoxModel(mmm_predict, config)

        # Test with weekly data
        weeks = pd.date_range("2024-01-01", periods=4, freq="W")
        data = xr.Dataset(
            {
                "tv_spend": xr.DataArray([100_000, 150_000, 120_000, 80_000], dims=["week"]),
                "digital_spend": xr.DataArray([200_000, 250_000, 300_000, 280_000], dims=["week"]),
                "week": weeks,
            }
        )

        predictions = model.predict(data)
        assert len(predictions) == 4
        assert all(predictions.values > 1000)  # All should be above base sales

    def test_multi_geography_model(self):
        """Test a model with multiple geographies and time dimensions."""

        def geo_predict(data: xr.Dataset) -> xr.DataArray:
            # Different effectiveness by geography
            geo_multipliers = {"US": 1.2, "UK": 1.0, "DE": 0.9}

            result = np.zeros(data["spend"].shape)

            for i, geo in enumerate(data["geography"].values):
                multiplier = geo_multipliers.get(geo, 1.0)
                result[i, :] = data["spend"].isel(geography=i).values * multiplier

            return xr.DataArray(
                result,
                dims=["geography", "month"],
                coords={"geography": data["geography"], "month": data["month"]},
                name="sales",
            )

        config = (
            BlackBoxModelConfigBuilder()
            .model_info("geo_model", "1.0.0")
            .add_required_var(
                "spend",
                DataVarType.CONTINUOUS,
                bounds=(0, float("inf")),
                optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            )
            .add_dimension("geography", required=True, coordinate_type="geography")
            .add_dimension("month", required=True, coordinate_type="time")
            .add_output("sales", dims=["geography", "month"])
            .build()
        )

        model = BlackBoxModel(geo_predict, config)

        # Test data
        data = xr.Dataset(
            {
                "spend": xr.DataArray(
                    [[100, 110, 120], [100, 110, 120], [100, 110, 120]],  # US  # UK  # DE
                    dims=["geography", "month"],
                ),
                "geography": ["US", "UK", "DE"],
                "month": pd.date_range("2024-01", periods=3, freq="M"),
            }
        )

        predictions = model.predict(data)

        # US should have highest sales due to multiplier
        assert predictions.sel(geography="US").sum() > predictions.sel(geography="UK").sum()
        assert predictions.sel(geography="UK").sum() > predictions.sel(geography="DE").sum()

    def test_constraint_aware_model(self):
        """Test a model with constraint variables."""

        def constrained_predict(data: xr.Dataset) -> xr.DataArray:
            spend = data["spend"].values
            max_capacity = data["max_capacity"].values

            # Apply capacity constraint
            effective_spend = np.minimum(spend, max_capacity)

            return xr.DataArray(
                effective_spend * 0.1,  # 10% conversion
                dims=data["spend"].dims,
                coords=data["spend"].coords,
                name="conversions",
            )

        config = (
            BlackBoxModelConfigBuilder()
            .model_info("constrained_model", "1.0.0")
            .add_required_var(
                "spend",
                DataVarType.CONTINUOUS,
                optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            )
            .add_required_var(
                "max_capacity",
                DataVarType.CONTINUOUS,
                optimization_layer=OptimizationLayer.CONSTRAINT_LAYER,
            )
            .add_dimension("channel")
            .add_output("conversions", dims=["channel"])
            .map_budget_allocation(["spend"])
            .map_constraints(["max_capacity"])
            .build()
        )

        model = BlackBoxModel(constrained_predict, config)

        data = xr.Dataset(
            {
                "spend": xr.DataArray([1000, 2000, 3000], dims=["channel"]),
                "max_capacity": xr.DataArray([1500, 1500, 1500], dims=["channel"]),
                "channel": ["A", "B", "C"],
            }
        )

        predictions = model.predict(data)

        # Check that capacity constraints are respected
        assert float(predictions.sel(channel="A")) == 100  # 1000 * 0.1
        assert float(predictions.sel(channel="B")) == 150  # 1500 * 0.1 (capped)
        assert float(predictions.sel(channel="C")) == 150  # 1500 * 0.1 (capped)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_simple_model(self):
        """Test the create_simple_model convenience function."""

        def predict_fn(data: xr.Dataset) -> xr.DataArray:
            return xr.DataArray(data["x"].values + data["y"].values, dims=["time"], name="result")

        model = create_simple_model(
            predict_fn=predict_fn, input_vars=["x", "y"], output_name="result", dimensions=["time"]
        )

        data = xr.Dataset(
            {
                "x": xr.DataArray([1, 2, 3], dims=["time"]),
                "y": xr.DataArray([4, 5, 6], dims=["time"]),
                "time": [0, 1, 2],
            }
        )

        predictions = model.predict(data)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(predictions.values, expected)


if __name__ == "__main__":
    # Run example usage
    from atlas.models.blackbox.model_builder import example_usage

    example_usage()
