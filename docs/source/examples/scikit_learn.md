# Scikit-learn Integration with Atlas

This module provides seamless integration between scikit-learn regression models and the Atlas optimization framework.

## Overview

The integration consists of three main components:

1. **SklearnModelWrapper**: Wraps scikit-learn models to work with Atlas's xarray-based interface
2. **SklearnModelFactory**: Factory pattern for creating model wrappers
3. **ModelConfigBuilder**: Configuration builder for model setup

## Installation

```bash
# Assuming Atlas is already installed
pip install scikit-learn xarray pandas numpy
pip install xgboost lightgbm  # Optional, for advanced models
```

## Quick Start

### 1. Loading an Existing Model

```python
from atlas import OptimizerFactory
from atlas.models.sklearn_wrapper import SklearnModelWrapper

# Load pre-trained model
model_wrapper = SklearnModelWrapper(
    model_path='path/to/model.pkl',
    feature_names=['tv_spend', 'digital_spend', 'radio_spend'],
    target_name='revenue',
    scaler_path='path/to/scaler.pkl'  # Optional
)

# Create optimizer
optimizer = OptimizerFactory.create('scipy', model=model_wrapper)

# Define budget data
import xarray as xr
budget = xr.Dataset({
    'tv_spend': xr.DataArray([100000]),
    'digital_spend': xr.DataArray([200000]),
    'radio_spend': xr.DataArray([50000])
})

# Optimize
constraints = {
    'total_budget': 350000,
    'bounds': {
        'tv_spend': (50000, 200000),
        'digital_spend': (100000, 300000),
        'radio_spend': (25000, 100000)
    }
}

result = optimizer.optimize(budget, constraints)
print(f"Optimal allocation: {result.optimal_budget}")
print(f"Expected outcome: {result.optimal_value}")
```

### 2. Using the Factory Pattern

```python
from atlas.models.sklearn_factory import SklearnModelFactory

# Create from saved model
model_wrapper = SklearnModelFactory.create(
    model_path='path/to/model.pkl',
    feature_names=['feature1', 'feature2', 'feature3'],
    scaler_path='path/to/scaler.pkl'
)

# Create new model
model_wrapper = SklearnModelFactory.create(
    model_type='random_forest',
    feature_names=['tv', 'digital', 'radio'],
    model_params={
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
)
```

### 3. Configuration-Based Setup

```python
from atlas.models.sklearn_factory import ModelConfigBuilder

# Build configuration
config = (ModelConfigBuilder()
    .model_path('models/trained_model.pkl')
    .features(['tv', 'digital', 'radio', 'social'])
    .target('conversions')
    .scaler('models/scaler.pkl')
    .dimensions(time_dim='week', channel_dim='channel')
    .contribution_method('feature_importance')
    .build()
)

# Save configuration
builder = ModelConfigBuilder()
# ... configure ...
builder.save('config.yaml')

# Load from configuration
model_wrapper = SklearnModelFactory.from_config('config.yaml')
```

## Supported Models

The factory supports the following scikit-learn models out of the box:

- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet
- **Tree-Based**: RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor
- **Other**: SVR (Support Vector Regression)
- **External**: XGBoost, LightGBM

### Registering Custom Models

```python
from sklearn.base import BaseEstimator
from atlas.models.sklearn_factory import SklearnModelFactory

class CustomModel(BaseEstimator):
    def fit(self, X, y):
        # Implementation
        pass
    
    def predict(self, X):
        # Implementation
        pass

# Register the model
SklearnModelFactory.register_model('custom', CustomModel)

# Use it
model_wrapper = SklearnModelFactory.create(
    model_type='custom',
    feature_names=['f1', 'f2']
)
```

## Feature Contributions

The wrapper automatically handles feature contribution calculations based on model type:

- **Linear models**: Uses coefficients
- **Tree-based models**: Uses feature importances
- **Other models**: Uses permutation importance or equal distribution

```python
# Get contributions
budget_data = xr.Dataset({...})
contributions = model_wrapper.contributions(budget_data)

# Get feature importance
importance = model_wrapper.get_feature_importance()
print(f"Feature importance: {importance}")
```

## Working with Time Series

For time-based optimization:

```python
import pandas as pd

# Create time series data
dates = pd.date_range('2024-01-01', periods=12, freq='M')
budget = xr.Dataset({
    'tv': xr.DataArray(
        [100000] * 12,
        dims=['time'],
        coords={'time': dates}
    ),
    'digital': xr.DataArray(
        [200000] * 12,
        dims=['time'],
        coords={'time': dates}
    )
})

# Wrapper with time dimension
model_wrapper = SklearnModelWrapper(
    model_path='model.pkl',
    feature_names=['tv', 'digital'],
    time_dim='time'
)

# Optimize for each time period
for t in dates:
    period_budget = budget.sel(time=t)
    result = optimizer.optimize(period_budget, constraints)
    print(f"{t}: {result.optimal_budget}")
```

## Best Practices

### 1. Feature Scaling

Always use a scaler for better optimization results:

```python
from sklearn.preprocessing import StandardScaler

# During training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# Save both
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Use in wrapper
model_wrapper = SklearnModelWrapper(
    model_path='model.pkl',
    scaler_path='scaler.pkl',
    feature_names=feature_names
)
```

### 2. Model Validation

Validate your model before optimization:

```python
# Check model type
model_wrapper.validate_input(budget_data)

# Test predictions
test_pred = model_wrapper.predict(budget_data)
print(f"Prediction shape: {test_pred.shape}")
print(f"Prediction range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
```

### 3. Contribution Methods

Choose the appropriate contribution method:

```python
# For linear models
wrapper = SklearnModelWrapper(
    model=linear_model,
    contribution_method='coef'
)

# For tree-based models
wrapper = SklearnModelWrapper(
    model=rf_model,
    contribution_method='feature_importance'
)

# For black-box models
wrapper = SklearnModelWrapper(
    model=complex_model,
    contribution_method='permutation'
)
```

## Troubleshooting

### Common Issues

1. **Missing Features Error**
   ```python
   # Ensure all required features are present
   required = model_wrapper.feature_names
   provided = list(budget_data.data_vars)
   missing = set(required) - set(provided)
   ```

2. **Dimension Mismatch**
   ```python
   # Check dimensions match expectations
   print(f"Data dimensions: {list(budget_data.dims)}")
   print(f"Expected time dim: {model_wrapper.time_dim}")
   ```

3. **Scaling Issues**
   ```python
   # Ensure scaler matches training
   # If predictions seem off, check if scaling was applied during training
   ```

## Advanced Usage

### Multi-Objective Optimization

```python
# Wrap multiple models
revenue_model = SklearnModelWrapper(model_path='revenue.pkl', ...)
cost_model = SklearnModelWrapper(model_path='cost.pkl', ...)

# Use with multi-objective optimizer
from atlas import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    models={'revenue': revenue_model, 'cost': cost_model},
    objectives={
        'revenue': {'direction': 'maximize', 'weight': 0.7},
        'cost': {'direction': 'minimize', 'weight': 0.3}
    }
)
```

### Custom Preprocessing

```python
class CustomSklearnWrapper(SklearnModelWrapper):
    def _xarray_to_features(self, x: xr.Dataset) -> np.ndarray:
        # Custom feature extraction
        features = super()._xarray_to_features(x)
        
        # Add derived features
        tv_digital_interaction = features[:, 0] * features[:, 1]
        features = np.column_stack([features, tv_digital_interaction])
        
        return features
```

## Performance Tips

1. **Use appropriate model complexity** - Simpler models often optimize faster
2. **Cache predictions** when doing multiple optimizations
3. **Parallelize** time series optimizations when possible
4. **Profile** your model to identify bottlenecks

## Next Steps

- Explore Atlas's visualization tools for optimization results
- Implement custom constraints for your business rules
- Consider ensemble approaches for robust optimization
- Integrate with your data pipeline for automated optimization