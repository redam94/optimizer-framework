"""
Black Box Model Builder for Atlas Framework

This module provides a flexible system for wrapping arbitrary functions
as Atlas-compatible models with full configuration support.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from atlas.core.interfaces import AbstractModel


class DataVarType(Enum):
    """Types of data variables in the model."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    INTEGER = "integer"


class OptimizationLayer(Enum):
    """Types of optimization layers."""

    BUDGET_ALLOCATION = "budget_allocation"
    RESOURCE_ALLOCATION = "resource_allocation"
    CONSTRAINT_LAYER = "constraint_layer"
    OBJECTIVE_LAYER = "objective_layer"
    TRANSFORMATION_LAYER = "transformation_layer"


@dataclass
class DataVarSpec:
    """Specification for a data variable."""

    name: str
    dtype: DataVarType
    required: bool = True
    default_value: Optional[Any] = None
    bounds: Optional[Tuple[float, float]] = None
    description: Optional[str] = None
    units: Optional[str] = None
    optimization_layer: Optional[OptimizationLayer] = None
    transformation: Optional[str] = None  # e.g., "log", "sqrt", "normalize"


@dataclass
class DimensionSpec:
    """Specification for a dimension."""

    name: str
    required: bool = True
    default_size: Optional[int] = None
    description: Optional[str] = None
    coordinate_type: Optional[str] = None  # e.g., "time", "channel", "geography"


@dataclass
class OutputSpec:
    """Specification for model output."""

    name: str
    dims: List[str]
    dtype: str = "float64"
    description: Optional[str] = None
    units: Optional[str] = None
    aggregation_method: Optional[str] = None  # e.g., "sum", "mean", "max"


class BlackBoxModelConfig:
    """Configuration for black box models."""

    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.description: Optional[str] = None
        self.author: Optional[str] = None
        self.created_at: datetime = datetime.now()

        self.data_vars: Dict[Hashable, DataVarSpec] = {}
        self.dimensions: Dict[Hashable, DimensionSpec] = {}
        self.outputs: Dict[Hashable, OutputSpec] = {}

        self.optimization_mappings: Dict[OptimizationLayer, List[str]] = {}
        self.metadata: Dict[Hashable, Any] = {}
        self.preprocessing_steps: List[Dict[str, Any]] = []
        self.postprocessing_steps: List[Dict[str, Any]] = []

    def add_data_var(self, spec: DataVarSpec) -> "BlackBoxModelConfig":
        """Add a data variable specification."""
        self.data_vars[spec.name] = spec
        return self

    def add_dimension(self, spec: DimensionSpec) -> "BlackBoxModelConfig":
        """Add a dimension specification."""
        self.dimensions[spec.name] = spec
        return self

    def add_output(self, spec: OutputSpec) -> "BlackBoxModelConfig":
        """Add an output specification."""
        self.outputs[spec.name] = spec
        return self

    def map_optimization_layer(
        self, layer: OptimizationLayer, data_vars: List[str]
    ) -> "BlackBoxModelConfig":
        """Map data variables to optimization layers."""
        if layer not in self.optimization_mappings:
            self.optimization_mappings[layer] = []
        self.optimization_mappings[layer].extend(data_vars)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "data_vars": {
                name: {
                    "dtype": spec.dtype.value,
                    "required": spec.required,
                    "default_value": spec.default_value,
                    "bounds": spec.bounds,
                    "description": spec.description,
                    "units": spec.units,
                    "optimization_layer": (
                        spec.optimization_layer.value if spec.optimization_layer else None
                    ),
                    "transformation": spec.transformation,
                }
                for name, spec in self.data_vars.items()
            },
            "dimensions": {
                name: {
                    "required": spec.required,
                    "default_size": spec.default_size,
                    "description": spec.description,
                    "coordinate_type": spec.coordinate_type,
                }
                for name, spec in self.dimensions.items()
            },
            "outputs": {
                name: {
                    "dims": spec.dims,
                    "dtype": spec.dtype,
                    "description": spec.description,
                    "units": spec.units,
                    "aggregation_method": spec.aggregation_method,
                }
                for name, spec in self.outputs.items()
            },
            "optimization_mappings": {
                layer.value: vars_list for layer, vars_list in self.optimization_mappings.items()
            },
            "metadata": self.metadata,
            "preprocessing_steps": self.preprocessing_steps,
            "postprocessing_steps": self.postprocessing_steps,
        }

    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()

        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BlackBoxModelConfig":
        """Load configuration from file."""
        path = Path(path)

        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        config = cls()
        config.model_name = config_dict.get("model_name")
        config.model_version = config_dict.get("model_version")
        config.description = config_dict.get("description")
        config.author = config_dict.get("author")
        config.metadata = config_dict.get("metadata", {})

        # Load data vars
        for name, spec_dict in config_dict.get("data_vars", {}).items():
            spec = DataVarSpec(
                name=name,
                dtype=DataVarType(spec_dict["dtype"]),
                required=spec_dict.get("required", True),
                default_value=spec_dict.get("default_value"),
                bounds=tuple(spec_dict["bounds"]) if spec_dict.get("bounds") else None,
                description=spec_dict.get("description"),
                units=spec_dict.get("units"),
                optimization_layer=(
                    OptimizationLayer(spec_dict["optimization_layer"])
                    if spec_dict.get("optimization_layer")
                    else None
                ),
                transformation=spec_dict.get("transformation"),
            )
            config.add_data_var(spec)

        return config


class BlackBoxModelConfigBuilder:
    """Builder for creating black box model configurations."""

    def __init__(self) -> None:
        self.config = BlackBoxModelConfig()

    def model_info(
        self,
        name: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        author: Optional[str] = None,
    ) -> "BlackBoxModelConfigBuilder":
        """Set basic model information."""
        self.config.model_name = name
        self.config.model_version = version
        self.config.description = description
        self.config.author = author
        return self

    def add_required_var(
        self,
        name: str,
        dtype: DataVarType = DataVarType.CONTINUOUS,
        bounds: Optional[Tuple[float, float]] = None,
        units: Optional[str] = None,
        optimization_layer: Optional[OptimizationLayer] = None,
        description: Optional[str] = None,
    ) -> "BlackBoxModelConfigBuilder":
        """Add a required data variable."""
        spec = DataVarSpec(
            name=name,
            dtype=dtype,
            required=True,
            bounds=bounds,
            units=units,
            optimization_layer=optimization_layer,
            description=description,
        )
        self.config.add_data_var(spec)
        return self

    def add_optional_var(
        self,
        name: str,
        dtype: DataVarType = DataVarType.CONTINUOUS,
        default_value: Any = None,
        bounds: Optional[Tuple[float, float]] = None,
        units: Optional[str] = None,
        optimization_layer: Optional[OptimizationLayer] = None,
        description: Optional[str] = None,
    ) -> "BlackBoxModelConfigBuilder":
        """Add an optional data variable."""
        spec = DataVarSpec(
            name=name,
            dtype=dtype,
            required=False,
            default_value=default_value,
            bounds=bounds,
            units=units,
            optimization_layer=optimization_layer,
            description=description,
        )
        self.config.add_data_var(spec)
        return self

    def add_dimension(
        self,
        name: str,
        required: bool = True,
        default_size: Optional[int] = None,
        coordinate_type: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "BlackBoxModelConfigBuilder":
        """Add a dimension specification."""
        spec = DimensionSpec(
            name=name,
            required=required,
            default_size=default_size,
            coordinate_type=coordinate_type,
            description=description,
        )
        self.config.add_dimension(spec)
        return self

    def add_output(
        self,
        name: str,
        dims: List[str],
        dtype: str = "float64",
        units: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "BlackBoxModelConfigBuilder":
        """Add an output specification."""
        spec = OutputSpec(
            name=name,
            dims=dims,
            dtype=dtype,
            units=units,
            aggregation_method=aggregation_method,
            description=description,
        )
        self.config.add_output(spec)
        return self

    def map_budget_allocation(self, data_vars: List[str]) -> "BlackBoxModelConfigBuilder":
        """Map data variables to budget allocation layer."""
        self.config.map_optimization_layer(OptimizationLayer.BUDGET_ALLOCATION, data_vars)
        return self

    def map_constraints(self, data_vars: List[str]) -> "BlackBoxModelConfigBuilder":
        """Map data variables to constraint layer."""
        self.config.map_optimization_layer(OptimizationLayer.CONSTRAINT_LAYER, data_vars)
        return self

    def add_preprocessing(self, step: Dict[str, Any]) -> "BlackBoxModelConfigBuilder":
        """Add a preprocessing step."""
        self.config.preprocessing_steps.append(step)
        return self

    def add_metadata(self, key: str, value: Any) -> "BlackBoxModelConfigBuilder":
        """Add metadata."""
        self.config.metadata[key] = value
        return self

    def build(self) -> BlackBoxModelConfig:
        """Build and return the configuration."""
        return self.config


class BlackBoxModel(AbstractModel):
    """Black box model wrapper for arbitrary functions."""

    def __init__(
        self,
        predict_fn: Callable[[xr.Dataset], xr.DataArray],
        config: BlackBoxModelConfig,
        contribution_fn: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize black box model.

        Args:
            predict_fn: Function that takes xr.Dataset and returns xr.DataArray
            config: Model configuration
            contribution_fn: Optional function for calculating contributions
            name: Optional model name override
        """
        self.predict_fn = predict_fn
        self.config = config
        self.contribution_fn = contribution_fn
        self.name = name or config.model_name or "BlackBoxModel"

        # Cache for validation
        self._required_vars = {name for name, spec in config.data_vars.items() if spec.required}
        self._required_dims = {name for name, spec in config.dimensions.items() if spec.required}
        self._optional_vars = {name for name, spec in config.data_vars.items() if not spec.required}

    def _validate_input(self, x: xr.Dataset) -> None:
        """Validate input dataset against configuration."""
        # Check required variables
        missing_vars = self._required_vars - set(x.data_vars)
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Check required dimensions
        all_dims: set[Hashable] = set()
        for var in x.data_vars:
            all_dims.update(x[var].dims)

        missing_dims = self._required_dims - all_dims
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")

        # Validate data types and bounds
        for var_name in x.data_vars:
            if var_name in self.config.data_vars:
                spec = self.config.data_vars[var_name]

                # Check bounds if specified
                if spec.bounds:
                    min_val, max_val = spec.bounds
                    data = x[var_name].values
                    if np.any(data < min_val) or np.any(data > max_val):
                        raise ValueError(
                            f"Variable '{var_name}' has values outside bounds {spec.bounds}"
                        )

    def _apply_preprocessing(self, x: xr.Dataset) -> xr.Dataset:
        """Apply preprocessing steps."""
        x_processed: xr.Dataset = x.copy()

        # Apply transformations
        for var_name, spec in self.config.data_vars.items():
            if var_name in x_processed and spec.transformation:
                if spec.transformation == "log":
                    x_processed[var_name] = np.log(x_processed[var_name] + 1e-8)
                elif spec.transformation == "sqrt":
                    x_processed[var_name] = np.sqrt(x_processed[var_name])
                elif spec.transformation == "normalize":
                    data = x_processed[var_name]
                    x_processed[var_name] = (data - data.mean()) / (data.std() + 1e-8)

        # Apply custom preprocessing steps
        for step in self.config.preprocessing_steps:
            step_type = step.get("type")
            if step_type == "scale":
                factor = step.get("factor", 1.0)
                vars_to_scale = step.get("variables", [])
                for var in vars_to_scale:
                    if var in x_processed:
                        x_processed[var] *= factor

        return x_processed

    def _add_defaults(self, x: xr.Dataset) -> xr.Dataset:
        """Add default values for optional variables."""
        x_with_defaults: xr.Dataset = x.copy()

        for var_name, spec in self.config.data_vars.items():
            if (
                not spec.required
                and var_name not in x_with_defaults
                and spec.default_value is not None
            ):
                # Create default array with appropriate shape
                shape = [x_with_defaults.dims[dim] for dim in x_with_defaults.dims]
                default_array = np.full(shape, spec.default_value)

                x_with_defaults[var_name] = xr.DataArray(
                    default_array, dims=list(x_with_defaults.dims), name=var_name
                )

        return x_with_defaults

    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Generate predictions from input data."""
        # Validate input
        self._validate_input(x)

        # Add defaults
        x_with_defaults = self._add_defaults(x)

        # Apply preprocessing
        x_processed = self._apply_preprocessing(x_with_defaults)

        # Call prediction function
        predictions = self.predict_fn(x_processed)

        # Validate output
        if not isinstance(predictions, xr.DataArray):
            raise TypeError(
                f"Prediction function must return xr.DataArray, got {type(predictions)}"
            )

        return predictions

    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Calculate feature contributions."""
        if self.contribution_fn:
            return self.contribution_fn(x)

        # Default: equal contributions
        predictions = self.predict(x)
        total = float(predictions.sum())

        contributions = xr.Dataset()
        optimizable_vars = []

        # Get variables mapped to budget allocation
        if OptimizationLayer.BUDGET_ALLOCATION in self.config.optimization_mappings:
            optimizable_vars = self.config.optimization_mappings[
                OptimizationLayer.BUDGET_ALLOCATION
            ]

        if not optimizable_vars:
            # Fallback: use all continuous variables
            optimizable_vars = [
                str(name)
                for name, spec in self.config.data_vars.items()
                if spec.dtype == DataVarType.CONTINUOUS
            ]

        # Calculate contributions
        for var in optimizable_vars:
            if var in x:
                weight = float(x[var].sum()) / sum(
                    float(x[v].sum()) for v in optimizable_vars if v in x
                )
                contributions[var] = xr.DataArray(
                    [total * weight], dims=["contribution"], name=f"{var}_contribution"
                )

        return contributions

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return f"blackbox_{self.name}"

    @property
    def required_dimensions(self) -> List[str]:
        """Return required dimensions."""
        return list(str(dim) for dim in self._required_dims)

    def get_optimization_variables(self, layer: OptimizationLayer) -> List[str]:
        """Get variables mapped to a specific optimization layer."""
        return self.config.optimization_mappings.get(layer, [])

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the model configuration."""
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "required_variables": list(self._required_vars),
            "optional_variables": list(self._optional_vars),
            "required_dimensions": list(self._required_dims),
            "outputs": list(self.config.outputs.keys()),
            "optimization_layers": {
                layer.value: vars_list
                for layer, vars_list in self.config.optimization_mappings.items()
            },
        }


class BlackBoxModelBuilder:
    """Builder for creating black box models."""

    def __init__(self) -> None:
        self.config_builder = BlackBoxModelConfigBuilder()
        self.predict_fn: Optional[Callable] = None
        self.contribution_fn: Optional[Callable] = None
        self.name: Optional[str] = None

    def from_config(self, config: BlackBoxModelConfig) -> "BlackBoxModelBuilder":
        """Initialize builder from existing configuration."""
        self.config_builder.config = config
        return self

    def set_predict_function(
        self, fn: Callable[[xr.Dataset], xr.DataArray]
    ) -> "BlackBoxModelBuilder":
        """Set the prediction function."""
        self.predict_fn = fn
        return self

    def set_contribution_function(
        self, fn: Callable[[xr.Dataset], xr.Dataset]
    ) -> "BlackBoxModelBuilder":
        """Set the contribution calculation function."""
        self.contribution_fn = fn
        return self

    def set_name(self, name: str) -> "BlackBoxModelBuilder":
        """Set the model name."""
        self.name = name
        self.config_builder.config.model_name = name
        return self

    def build(self) -> BlackBoxModel:
        """Build and return the black box model."""
        if not self.predict_fn:
            raise ValueError("Prediction function must be set")

        config = self.config_builder.build()

        return BlackBoxModel(
            predict_fn=self.predict_fn,
            config=config,
            contribution_fn=self.contribution_fn,
            name=self.name,
        )


# Convenience functions for common patterns


def create_simple_model(
    predict_fn: Callable,
    input_vars: List[str],
    output_name: str = "prediction",
    dimensions: Optional[List[str]] = None,
) -> BlackBoxModel:
    """Create a simple black box model with minimal configuration."""
    builder = BlackBoxModelBuilder()

    # Add input variables
    for var in input_vars:
        builder.config_builder.add_required_var(
            var, DataVarType.CONTINUOUS, optimization_layer=OptimizationLayer.BUDGET_ALLOCATION
        )

    # Add dimensions
    if dimensions:
        for dim in dimensions:
            builder.config_builder.add_dimension(dim)

    # Add output
    builder.config_builder.add_output(output_name, dims=dimensions or [], dtype="float64")

    # Map budget allocation
    builder.config_builder.map_budget_allocation(input_vars)

    # Set prediction function
    builder.set_predict_function(predict_fn)

    return builder.build()


# Example usage function
def example_usage() -> BlackBoxModel:
    """Example of using the black box model builder."""

    # Define a simple prediction function
    def my_predict_fn(data: xr.Dataset) -> xr.DataArray:
        # Simple linear combination with diminishing returns
        tv_effect = np.sqrt(data["tv_spend"].values)
        digital_effect = np.sqrt(data["digital_spend"].values)
        radio_effect = np.sqrt(data["radio_spend"].values)

        # Add seasonality if present
        if "seasonality" in data:
            seasonal_mult: Union[float, np.ndarray] = data["seasonality"].values
        else:
            seasonal_mult = 1.0

        prediction = (0.5 * tv_effect + 0.3 * digital_effect + 0.2 * radio_effect) * seasonal_mult

        return xr.DataArray(prediction, dims=data["tv_spend"].dims, name="revenue")

    # Build configuration
    config = (
        BlackBoxModelConfigBuilder()
        .model_info(
            name="marketing_mix_model",
            version="1.0.0",
            description="Marketing mix model with diminishing returns",
            author="Data Science Team",
        )
        # Required variables
        .add_required_var(
            "tv_spend",
            DataVarType.CONTINUOUS,
            bounds=(0, 1_000_000),
            units="USD",
            optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            description="Television advertising spend",
        )
        .add_required_var(
            "digital_spend",
            DataVarType.CONTINUOUS,
            bounds=(0, 1_000_000),
            units="USD",
            optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            description="Digital advertising spend",
        )
        .add_required_var(
            "radio_spend",
            DataVarType.CONTINUOUS,
            bounds=(0, 500_000),
            units="USD",
            optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
            description="Radio advertising spend",
        )
        # Optional variables
        .add_optional_var(
            "seasonality",
            DataVarType.CONTINUOUS,
            default_value=1.0,
            bounds=(0.5, 2.0),
            description="Seasonal multiplier",
        )
        # Dimensions
        .add_dimension("time", required=True, coordinate_type="time")
        .add_dimension("geography", required=False, coordinate_type="geography")
        # Output
        .add_output(
            "revenue",
            dims=["time"],
            units="USD",
            aggregation_method="sum",
            description="Predicted revenue",
        )
        # Optimization mappings
        .map_budget_allocation(["tv_spend", "digital_spend", "radio_spend"])
        # Metadata
        .add_metadata("model_type", "marketing_mix")
        .add_metadata("training_date", "2024-01-01")
        .build()
    )

    # Create model
    model = BlackBoxModel(predict_fn=my_predict_fn, config=config, name="marketing_mix_model")

    # Test the model
    test_data = xr.Dataset(
        {
            "tv_spend": xr.DataArray([100_000], dims=["time"]),
            "digital_spend": xr.DataArray([200_000], dims=["time"]),
            "radio_spend": xr.DataArray([50_000], dims=["time"]),
            "time": pd.date_range("2024-01-01", periods=1),
        }
    )

    predictions = model.predict(test_data)
    contributions = model.contributions(test_data)

    print(f"Predictions: {predictions}")
    print(f"Contributions: {contributions}")
    print(f"Config Summary: {model.get_config_summary()}")

    # Save configuration
    config.save("marketing_mix_config.yaml")

    return model
