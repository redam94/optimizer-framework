# Black Box Model Integration Guide

## Overview

The Black Box Model Builder provides a flexible system for wrapping arbitrary functions into Atlas-compatible models. This guide covers common usage patterns and best practices.

## Key Components

### 1. **BlackBoxModelConfig**
Defines the complete specification of your model including:
- Required and optional data variables
- Dimensions (time, geography, etc.)
- Output specifications
- Optimization layer mappings
- Metadata and preprocessing steps

### 2. **BlackBoxModelConfigBuilder**
A fluent API for building configurations:
```python
config = (BlackBoxModelConfigBuilder()
    .model_info(name="my_model", version="1.0.0")
    .add_required_var("spend", DataVarType.CONTINUOUS)
    .add_dimension("time")
    .add_output("revenue", dims=["time"])
    .build()
)
```

### 3. **BlackBoxModel**
The actual model wrapper that:
- Validates inputs against configuration
- Applies preprocessing
- Calls your prediction function
- Calculates contributions

## Quick Start Examples

### Example 1: Simple Linear Model

```python
import xarray as xr
import numpy as np
from atlas import create_simple_blackbox_model

# Define prediction function
def linear_predict(data: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        0.5 * data["tv_spend"].values + 0.3 * data["digital_spend"].values,
        dims=data["tv_spend"].dims,
        name="revenue"
    )

# Create model with minimal configuration
model = create_simple_blackbox_model(
    predict_fn=linear_predict,
    input_vars=["tv_spend", "digital_spend"],
    output_name="revenue",
    dimensions=["time"]
)

# Use with optimizer
from atlas import OptimizerFactory

optimizer = OptimizerFactory.create("scipy", model=model)
result = optimizer.optimize(initial_budget, constraints)
```

### Example 2: Marketing Mix Model with Saturation

```python
from atlas import (
    BlackBoxModelBuilder,
    BlackBoxModelConfigBuilder,
    DataVarType,
    OptimizationLayer
)

def mmm_with_saturation(data: xr.Dataset) -> xr.DataArray:
    """Marketing mix model with diminishing returns."""
    # Hill saturation function
    def hill_transform(x, alpha=2.5, gamma=0.5):
        return x**alpha / (x**alpha + gamma**alpha)
    
    # Apply saturation to each channel
    tv_effect = hill_transform(data["tv_spend"].values) * 0.4
    digital_effect = hill_transform(data["digital_spend"].values) * 0.6
    
    # Add base sales
    base_sales = 1000
    revenue = base_sales + tv_effect + digital_effect
    
    return xr.DataArray(revenue, dims=data["tv_spend"].dims, name="revenue")

# Build comprehensive configuration
config = (BlackBoxModelConfigBuilder()
    .model_info(
        name="mmm_saturation",
        version="1.0.0",
        description="Marketing mix model with Hill saturation",
        author="Data Science Team"
    )
    # Define variables with bounds and metadata
    .add_required_var(
        "tv_spend",
        DataVarType.CONTINUOUS,
        bounds=(0, 500_000),
        units="USD",
        optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
        description="Television advertising spend"
    )
    .add_required_var(
        "digital_spend", 
        DataVarType.CONTINUOUS,
        bounds=(0, 500_000),
        units="USD",
        optimization_layer=OptimizationLayer.BUDGET_ALLOCATION,
        description="Digital advertising spend"
    )
    # Add dimensions
    .add_dimension("week", required=True, coordinate_type="time")
    # Define output
    .add_output(
        "revenue",
        dims=["week"],
        units="USD",
        aggregation_method="sum"
    )
    # Map optimization layers
    .map_budget_allocation(["tv_spend", "digital_spend"])
    # Add metadata
    .add_metadata("saturation_alpha", 2.5)
    .add_metadata("saturation_gamma", 0.5)
    .build()
)

# Create model
model = BlackBoxModel(mmm_with_saturation, config)
```

### Example 3: Multi-Geography Model with Constraints

```python
def geo_aware_model(data: xr.Dataset) -> xr.DataArray:
    """Model with geography-specific effectiveness and constraints."""
    # Geography-specific multipliers
    geo_effectiveness = {
        "US": {"digital": 1.2, "tv": 1.0},
        "UK": {"digital": 1.0, "tv": 1.1},
        "DE": {"digital": 0.9, "tv": 0.8}
    }
    
    # Calculate revenue by geography and channel
    revenue = xr.zeros_like(data["digital_spend"])
    
    for geo in data["geography"].values:
        geo_idx = data["geography"] == geo
        digital_mult = geo_effectiveness.get(str(geo), {}).get("digital", 1.0)
        tv_mult = geo_effectiveness.get(str(geo), {}).get("tv", 1.0)
        
        # Apply market size constraints if available
        if "market_size" in data:
            market_cap = data["market_size"].sel(geography=geo).values
        else:
            market_cap = float('inf')
        
        geo_revenue = (
            data["digital_spend"].sel(geography=geo) * digital_mult * 0.15 +
            data["tv_spend"].sel(geography=geo) * tv_mult * 0.10
        )
        
        # Cap by market size
        revenue.loc[{"geography": geo}] = np.minimum(geo_revenue, market_cap)
    
    return revenue.rename("revenue")

# Configuration with optional constraint variable
config = (BlackBoxModelConfigBuilder()
    .model_info("geo_model", "1.0.0")
    # Budget variables
    .add_required_var(
        "digital_spend",
        optimization_layer=OptimizationLayer.BUDGET_ALLOCATION
    )
    .add_required_var(
        "tv_spend",
        optimization_layer=OptimizationLayer.BUDGET_ALLOCATION
    )
    # Optional constraint
    .add_optional_var(
        "market_size",
        DataVarType.CONTINUOUS,
        default_value=float('inf'),
        optimization_layer=OptimizationLayer.CONSTRAINT_LAYER,
        description="Maximum market potential by geography"
    )
    # Multiple dimensions
    .add_dimension("geography", required=True, coordinate_type="geography")
    .add_dimension("time", required=True, coordinate_type="time")
    # Output
    .add_output("revenue", dims=["geography", "time"])
    .build()
)

model = BlackBoxModel(geo_aware_model, config)
```

## Advanced Features

### Custom Contribution Calculation

By default, the black box model calculates proportional contributions. You can provide a custom contribution function:

```python
def custom_contributions(data: xr.Dataset) -> xr.Dataset:
    """Calculate marginal contributions using finite differences."""
    predictions = model.predict(data)
    contributions = xr.Dataset()
    
    # Calculate marginal contribution for each variable
    for var in ["tv_spend", "digital_spend"]:
        # Create perturbed data
        delta = 1000  # $1000 change
        data_plus = data.copy()
        data_plus[var] = data[var] + delta
        
        # Calculate marginal effect
        pred_plus = model.predict(data_plus)
        marginal = (pred_plus - predictions) / delta
        
        contributions[f"{var}_marginal"] = marginal
        contributions[f"{var}_total"] = marginal * data[var]
    
    return contributions

# Use with model
model = BlackBoxModel(
    predict_fn=mmm_with_saturation,
    config=config,
    contribution_fn=custom_contributions
)
```

### Preprocessing and Transformations

Add preprocessing steps to your configuration:

```python
config = (BlackBoxModelConfigBuilder()
    # ... other configuration ...
    
    # Add variable-level transformations
    .add_required_var(
        "price",
        transformation="log",  # Will apply log(x + 1e-8)
        description="Log-transformed price"
    )
    
    # Add custom preprocessing steps
    .add_preprocessing({
        "type": "scale",
        "factor": 1000,
        "variables": ["spend_in_thousands"]
    })
    .add_preprocessing({
        "type": "normalize",
        "method": "z-score",
        "variables": ["temperature", "humidity"]
    })
    .build()
)
```

### Working with Time Series

Handle time-based features and carryover effects:

```python
def time_series_model(data: xr.Dataset) -> xr.DataArray:
    """Model with time-based features."""
    # Extract time features
    time_coords = pd.to_datetime(data["time"].values)
    
    # Seasonal factors
    month = time_coords.month
    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
    
    # Trend component
    days_since_start = (time_coords - time_coords[0]).days
    trend = 1 + 0.001 * days_since_start
    
    # Adstock transformation for carryover
    def adstock(x, decay=0.7):
        result = np.zeros_like(x)
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = x[i] + decay * result[i-1]
        return result
    
    # Apply transformations
    tv_adstock = adstock(data["tv_spend"].values)
    
    # Calculate revenue
    revenue = (
        1000 * trend * seasonal_factor +  # Base with trend and seasonality
        0.1 * tv_adstock  # Media effect with carryover
    )
    
    return xr.DataArray(revenue, dims=["time"], name="revenue")

# Configuration
config = (BlackBoxModelConfigBuilder()
    .model_info("time_series_model", "1.0.0")
    .add_required_var("tv_spend", optimization_layer=OptimizationLayer.BUDGET_ALLOCATION)
    .add_dimension("time", required=True, coordinate_type="time")
    .add_output("revenue", dims=["time"])
    .add_metadata("adstock_decay", 0.7)
    .add_metadata("has_seasonality", True)
    .add_metadata("has_trend", True)
    .build()
)
```

## Integration with Atlas Optimization

### Using with Different Optimizers

```python
# Create model
model = BlackBoxModel(predict_fn, config)

# Use with SciPy optimizer
scipy_optimizer = OptimizerFactory.create("scipy", model=model)
scipy_result = scipy_optimizer.optimize(initial_budget, constraints)

# Use with Optuna optimizer for black-box optimization
optuna_optimizer = OptimizerFactory.create(
    "optuna",
    model=model,
    config={
        "n_trials": 1000,
        "sampler": "TPE",
        "n_jobs": -1
    }
)
optuna_result = optuna_optimizer.optimize(initial_budget, constraints)
```

### Multi-Objective Optimization

```python
# Create multiple models for different objectives
revenue_model = BlackBoxModel(revenue_predict_fn, revenue_config)
awareness_model = BlackBoxModel(awareness_predict_fn, awareness_config)

# Use multi-objective optimizer
from atlas.optimizers import MultiObjectiveOptimizer

multi_optimizer = MultiObjectiveOptimizer(
    models={"revenue": revenue_model, "awareness": awareness_model},
    weights={"revenue": 0.7, "awareness": 0.3}
)

pareto_results = multi_optimizer.optimize(initial_budget, constraints)
```

## Best Practices

### 1. **Validate Your Functions**
Always test your prediction function with sample data before wrapping:
```python
# Test function directly
test_data = xr.Dataset({...})
test_output = my_predict_fn(test_data)
assert isinstance(test_output, xr.DataArray)
```

### 2. **Use Meaningful Variable Names**
Choose descriptive names that match your business domain:
```python
.add_required_var("tv_grp", description="Television Gross Rating Points")
.add_required_var("digital_impressions", description="Digital ad impressions")
```

### 3. **Document Metadata**
Include important model information:
```python
.add_metadata("training_period", "2023-01-01 to 2023-12-31")
.add_metadata("model_assumptions", ["linear additivity", "no interaction effects"])
.add_metadata("data_sources", ["Nielsen", "Google Analytics"])
```

### 4. **Set Reasonable Bounds**
Define realistic bounds for optimization:
```python
.add_required_var(
    "spend",
    bounds=(0, 1_000_000),  # Realistic budget constraints
    description="Channel spend in USD"
)
```

### 5. **Save and Version Configurations**
```python
# Save configuration
config.save("models/configs/mmm_v2.1.0.yaml")

# Load and reuse
loaded_config = BlackBoxModelConfig.load("models/configs/mmm_v2.1.0.yaml")
model = BlackBoxModel(predict_fn, loaded_config)
```

## Troubleshooting

### Common Issues

1. **Missing Variables Error**
   ```python
   # Error: Missing required variables: {'var1', 'var2'}
   # Solution: Ensure all required variables are in your input dataset
   ```

2. **Dimension Mismatch**
   ```python
   # Error: Missing required dimensions: {'time'}
   # Solution: Add required dimensions to your data
   data = data.expand_dims({"time": [0]})
   ```

3. **Output Shape Issues**
   ```python
   # Ensure output has correct dimensions
   return xr.DataArray(
       predictions,
       dims=data[input_var].dims,  # Match input dimensions
       name=output_name
   )
   ```

## Next Steps

1. Start with the `create_simple_blackbox_model` function for quick prototypes
2. Build comprehensive configurations for production models
3. Integrate with Atlas optimizers for budget optimization
4. Use the configuration system to document and version your models
5. Implement custom contribution functions for better interpretability

For more examples, see the test file `test_blackbox_model_builder.py` which includes additional patterns and use cases.