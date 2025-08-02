# 4 Core Concepts

Atlas is a unified optimization framework that enables data-driven budget allocation across diverse models and scenarios. This document covers the four fundamental concepts that make Atlas powerful and flexible.

## Overview

Atlas operates on four core concepts that work together to solve complex optimization problems:

1. **[Models](#models)** - Predictive models that estimate outcomes from budget allocations
2. **[Optimization](#optimization)** - Algorithms and strategies for finding optimal solutions
3. **[Constraints](#constraints)** - Business rules and limitations that solutions must satisfy
4. **[Data](#data)** - Multi-dimensional data structures and management

---

## Data

Data management in Atlas handles the complex, multi-dimensional nature of budget allocation problems. The framework uses Xarray as its foundation for powerful and flexible data structures. See [Data](data.md) for more detail.

### Data Architecture

Atlas is built on Xarray, which provides:
- **Multi-dimensional arrays** with labeled axes
- **Automatic alignment** across different data sources
- **Broadcasting** for operations across dimensions
- **Missing data handling** with interpolation options
- **Efficient computation** with NumPy backend

### Core Data Structures

#### **Budget Allocations**
Budget data is represented as Xarray DataArrays or Datasets with labeled dimensions:

```python
import xarray as xr
import numpy as np

# Simple budget allocation (2D: channels × time)
budget = xr.DataArray(
    data=np.array([
        [100_000, 120_000, 110_000, 130_000],  # Digital by month
        [200_000, 180_000, 190_000, 210_000],  # TV by month
        [50_000, 60_000, 55_000, 65_000]       # Radio by month
    ]),
    dims=['channel', 'month'],
    coords={
        'channel': ['digital', 'tv', 'radio'],
        'month': ['jan', 'feb', 'mar', 'apr']
    }
)

# Complex budget allocation (4D: channels × regions × time × products)
complex_budget = xr.Dataset({
    'budget': xr.DataArray(
        data=np.random.rand(3, 2, 12, 4) * 1_000_000,
        dims=['channel', 'region', month', 'product'],
        coords={
            'channel': ['digital', 'tv', 'radio'],
            'region': ['north', 'south'],
            'month': range(1, 13),
            'product': ['product_a', 'product_b', 'product_c', 'product_d']
        }
    )
})
```

#### **Model Outputs**
Model predictions follow the same structure for consistency:

```python
# Revenue predictions with confidence intervals
revenue_predictions = xr.Dataset({
    'revenue': xr.DataArray(
        data=predicted_revenue,
        dims=['channel', 'month'],
        coords={'channel': channels, 'month': months}
    ),
    'revenue_lower': xr.DataArray(
        data=revenue_lower_bound,
        dims=['channel', 'month'],
        coords={'channel': channels, 'month': months}
    ),
    'revenue_upper': xr.DataArray(
        data=revenue_upper_bound,
        dims=['channel', 'month'],
        coords={'channel': channels, 'month': months}
    )
})
```

#### **Historical Data**
Historical performance data for model training and validation:

```python
historical_data = xr.Dataset({
    'spend': xr.DataArray(
        data=historical_spend_data,
        dims=['date', 'channel'],
        coords={'date': date_range, 'channel': channels}
    ),
    'impressions': xr.DataArray(
        data=impressions_data,
        dims=['date', 'channel'],
        coords={'date': date_range, 'channel': channels}
    ),
    'conversions': xr.DataArray(
        data=conversion_data,
        dims=['date', 'channel'],
        coords={'date': date_range, 'channel': channels}
    ),
    'revenue': xr.DataArray(
        data=revenue_data,
        dims=['date'],
        coords={'date': date_range}
    )
})
```

### Data Operations

#### **Budget Aggregation**
```python
# Total budget by channel across all time periods
total_by_channel = budget.sum(dim='month')

# Monthly totals across all channels
monthly_totals = budget.sum(dim='channel')

# Budget allocation percentages
budget_percentages = budget / budget.sum(dim='channel')
```

#### **Data Alignment**
Xarray automatically handles alignment across different data sources:

```python
# Automatically aligns budget and cost data
cost_per_impression = budget / impressions

# Handles missing values gracefully
roi = revenue / budget.where(budget > 0)
```

#### **Resampling and Interpolation**
```python
# Resample daily data to monthly
monthly_data = daily_budget.resample(date='M').mean()

# Interpolate missing values
complete_budget = budget.interpolate_na(dim='month', method='linear')

# Fill forward for categorical data
filled_budget = budget.fillna(method='ffill')
```

### Data Validation

Atlas includes comprehensive data validation to ensure data quality:

```python
from atlas.data import DataValidator

validator = DataValidator()

# Validate data structure
validation_result = validator.validate_structure(
    data=budget_data,
    expected_dims=['channel', 'month'],
    required_coords=['channel', 'month']
)

# Check for data quality issues
quality_report = validator.check_data_quality(
    data=budget_data,
    checks=['missing_values', 'negative_values', 'outliers', 'consistency']
)

# Validate business rules
business_validation = validator.validate_business_rules(
    data=budget_data,
    rules={
        'positive_budgets': 'budget >= 0',
        'reasonable_scale': 'budget <= 10_000_000',
        'minimum_spend': 'budget.sum() >= 100_000'
    }
)
```

### Data Transformation

Atlas provides utilities for common data transformations:

#### **Unit Conversion**
```python
from atlas.data import UnitConverter

converter = UnitConverter()

# Convert between currencies
budget_usd = converter.convert_currency(
    budget_local,
    from_currency='EUR',
    to_currency='USD',
    exchange_rates=exchange_rate_data
)

# Normalize spend by market size
normalized_budget = converter.normalize_by_market_size(
    budget=budget_data,
    market_sizes=market_size_by_region
)
```

#### **Time Aggregation**
```python
from atlas.data import TimeAggregator

aggregator = TimeAggregator()

# Aggregate daily data to weekly
weekly_budget = aggregator.aggregate_time(
    data=daily_budget,
    target_frequency='W',
    method='sum'
)

# Create rolling windows
rolling_budget = aggregator.rolling_window(
    data=budget_data,
    window=4,  # 4-week rolling average
    method='mean'
)
```

#### **Feature Engineering**
```python
from atlas.data import FeatureEngineer

engineer = FeatureEngineer()

# Create lagged features
lagged_features = engineer.create_lags(
    data=budget_data,
    lags=[1, 2, 4],  # 1-week, 2-week, 4-week lags
    variables=['budget', 'impressions']
)

# Calculate moving averages
moving_averages = engineer.moving_average(
    data=budget_data,
    windows=[4, 8, 12],  # 4, 8, 12-week averages
    center=True
)

# Create interaction terms
interactions = engineer.create_interactions(
    data=budget_data,
    interaction_pairs=[('digital', 'tv'), ('radio', 'print')]
)
```

### Data Import/Export

Atlas supports various data formats and sources:

#### **File Formats**
```python
from atlas.data import DataLoader, DataExporter

loader = DataLoader()

# Load from CSV
budget_data = loader.from_csv(
    'budget_data.csv',
    index_cols=['date'],
    parse_dates=['date']
)

# Load from Excel with multiple sheets
excel_data = loader.from_excel(
    'marketing_data.xlsx',
    sheets=['budget', 'performance', 'costs'],
    header_row=2
)

# Load from Parquet (efficient for large datasets)
large_dataset = loader.from_parquet('historical_data.parquet')

# Export results
exporter = DataExporter()
exporter.to_excel(
    optimization_results,
    'optimization_results.xlsx',
    include_metadata=True
)
```

#### **Database Connections**
```python
from atlas.data import DatabaseConnector

db = DatabaseConnector(
    connection_string="postgresql://user:pass@localhost/marketing"
)

# Load budget data from database
budget_query = """
    SELECT date, channel, spend, impressions, conversions
    FROM marketing_spend
    WHERE date >= %s AND date <= %s
"""

budget_data = db.query_to_xarray(
    query=budget_query,
    params=[start_date, end_date],
    index_cols=['date'],
    data_vars=['spend', 'impressions', 'conversions']
)
```

#### **API Integration**
```python
from atlas.data import APIConnector

api = APIConnector(
    base_url="https://api.marketing-platform.com",
    auth_token="your-token"
)

# Fetch performance data
performance_data = api.get_performance_data(
    start_date=start_date,
    end_date=end_date,
    metrics=['impressions', 'clicks', 'conversions'],
    dimensions=['channel', 'campaign']
)
```

### Data Performance

Atlas includes optimizations for handling large datasets:

#### **Lazy Loading**
```python
# Load large datasets lazily (don't load into memory until needed)
large_dataset = xr.open_dataset(
    'huge_marketing_data.nc',
    chunks={'date': 1000, 'channel': 10}  # Dask chunks
)

# Computations are lazy until explicitly computed
result = large_dataset.groupby('channel').mean()
computed_result = result.compute()  # Now actually compute
```

#### **Efficient Storage**
```python
# Save with compression for smaller file sizes
budget_data.to_netcdf(
    'budget_data.nc',
    encoding={'budget': {'zlib': True, 'complevel': 9}}
)

# Use efficient data types
optimized_data = budget_data.astype({
    'budget': 'float32',    # Reduce precision if appropriate
    'channel': 'category'   # Use categorical for string data
})
```

---

## Models

Models in Atlas are the predictive engines that estimate outcomes (revenue, awareness, conversions, etc.) based on actions that can be taken. The framework is designed to be completely model-agnostic, supporting any type of predictive model. [Models](models.md)

### Model Types

Atlas supports integration with various model types:

#### **Machine Learning Models**
- **Scikit-learn models**: Random Forest, XGBoost, Neural Networks
- **Deep learning frameworks**: TensorFlow, PyTorch models
- **Time series models**: ARIMA, Prophet, seasonal decomposition
- **Custom ML pipelines**: End-to-end prediction workflows

#### **Statistical Models**
- **Regression models**: Linear, logistic, mixed effects
- **Bayesian models**: PyMC, Stan implementations
- **Econometric models**: Marketing mix models, attribution models

#### **External Services**
- **APIs**: Third-party prediction services, cloud ML platforms
- **Legacy systems**: Existing business intelligence tools
- **Excel models**: Business rule-based spreadsheet models

#### **Custom Models**
- **Business rules**: Custom logic implementations
- **Hybrid models**: Combinations of multiple approaches
- **Rule engines**: Decision trees and business logic

### Model Integration

Atlas provides multiple integration patterns to accommodate different model architectures:

#### **1. Direct Python Integration**
```python
from atlas.models import BaseModel

class MyRevenueModel(BaseModel):
    def __init__(self, trained_model):
        self.model = trained_model
    
    def predict(self, budget_allocation):
        """
        Predict revenue from budget allocation.
        
        Args:
            budget_allocation: xarray.Dataset with budget by channel/time
            
        Returns:
            xarray.Dataset with predicted outcomes
        """
        # Transform budget to model features
        features = self.prepare_features(budget_allocation)
        
        # Generate predictions
        predictions = self.model.predict(features)
        
        # Return structured results
        return self.format_predictions(predictions)
    
    def prepare_features(self, budget_allocation):
        # Feature engineering logic
        pass
    
    def format_predictions(self, raw_predictions):
        # Format output as xarray Dataset
        pass
```

#### **2. Docker Container Integration**
```python
from atlas.models import DockerModel

# For models deployed as containerized services
model = DockerModel(
    image="mycompany/mmm-model:v1.2",
    port=8080,
    health_endpoint="/health",
    prediction_endpoint="/predict"
)
```

#### **3. API Integration**
```python
from atlas.models import APIModel

# For external prediction services
model = APIModel(
    endpoint="https://api.mycompany.com/revenue-model/predict",
    auth_token="your-auth-token",
    timeout=30,
    retry_config={'max_retries': 3, 'backoff_factor': 1.5}
)
```

### Model Requirements

All models must implement the `BaseModel` interface with these key methods:

#### **Required Methods**
- `predict(budget_allocation)`: Core prediction method
- `validate_input(budget_allocation)`: Input validation
- `get_feature_names()`: Return expected input features

#### **Optional Methods**
- `predict_confidence(budget_allocation)`: Uncertainty estimates
- `explain_prediction(budget_allocation)`: Model interpretability
- `health_check()`: Model availability status

### Model Validation

Atlas includes comprehensive validation to ensure model reliability:

```python
from atlas.validation import ModelValidator

validator = ModelValidator()

# Validate model interface compliance
is_valid, errors = validator.validate_interface(model)

# Test model predictions
test_results = validator.test_predictions(
    model=model,
    test_cases=sample_budgets,
    expected_properties=['positive_revenue', 'monotonic_response']
)

# Performance benchmarking
benchmarks = validator.benchmark_performance(
    model=model,
    budget_samples=performance_test_data
)
```

For a technical guide to implementing a model yourself see: [Model Integration](../guides/model_integration.md)

---

## Optimization

Optimization in Atlas finds the best levers to pull given your model predictions, business constraints, and objectives. The framework supports multiple optimization approaches and algorithms. [Optimization](optimization.md)

### Optimization Backends

Atlas provides several optimization engines, each suited for different problem types:

#### **SciPy Optimizer**
Best for: Continuous variables, gradient-based optimization, well-behaved objective functions

```python
from atlas.optimizers import ScipyOptimizer

optimizer = ScipyOptimizer(
    model=revenue_model,
    method='trust-constr',  # L-BFGS-B, SLSQP, trust-constr
    config={
        'maxiter': 1000,
        'tol': 1e-8,
        'ftol': 1e-9
    }
)
```

**Supported methods:**
- `L-BFGS-B`: Bounded optimization, good for smooth functions
- `SLSQP`: Sequential quadratic programming, handles constraints well
- `trust-constr`: Trust region, most robust for complex constraints

#### **Optuna Optimizer**
Best for: Hyperparameter tuning, discrete variables, black-box optimization, parallel execution

```python
from atlas.optimizers import OptunaOptimizer

optimizer = OptunaOptimizer(
    model=mmm_model,
    config={
        'n_trials': 1000,
        'sampler': 'TPE',  # Tree-structured Parzen Estimator
        'pruner': 'MedianPruner',
        'n_jobs': -1  # Use all CPU cores
    }
)
```

**Key features:**
- Bayesian optimization with TPE sampler
- Early stopping with pruning
- Parallel execution support
- Built-in hyperparameter suggestions

#### **CVXPY Optimizer**
Best for: Convex optimization, linear/quadratic programming, mathematical guarantees

```python
from atlas.optimizers import CVXPYOptimizer

optimizer = CVXPYOptimizer(
    model=linear_model,
    solver='ECOS',  # ECOS, SCS, MOSEK
    config={
        'verbose': True,
        'max_iters': 10000,
        'abstol': 1e-8
    }
)
```

**Supported problem types:**
- Linear programming (LP)
- Quadratic programming (QP)
- Second-order cone programming (SOCP)
- Semidefinite programming (SDP)

### Multi-Objective Optimization

Atlas supports optimizing multiple objectives simultaneously:

```python
from atlas.optimizers import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    objectives={
        'revenue': {
            'model': revenue_model,
            'weight': 0.5,
            'direction': 'maximize'
        },
        'awareness': {
            'model': awareness_model,
            'weight': 0.3,
            'direction': 'maximize'
        },
        'cost_efficiency': {
            'model': cost_model,
            'weight': 0.2,
            'direction': 'minimize'
        }
    },
    method='weighted_sum'  # weighted_sum, pareto_frontier, lexicographic
)

# Generate Pareto frontier
pareto_solutions = optimizer.generate_pareto_frontier(
    n_points=20,
    initial_budget=base_allocation
)
```

### Optimization Strategies

#### **Single-Point Optimization**
Find one optimal solution:

```python
result = optimizer.optimize(
    initial_budget=starting_allocation,
    constraints=business_constraints
)

print(f"Optimal allocation: {result.optimal_budget}")
print(f"Expected return: {result.optimal_value}")
print(f"Optimization time: {result.optimization_time}")
```

#### **Multi-Start Optimization**
Improve robustness by trying multiple starting points:

```python
result = optimizer.multistart_optimize(
    n_starts=10,
    initial_budget_range={
        'digital': (100_000, 500_000),
        'tv': (200_000, 800_000),
        'radio': (50_000, 300_000)
    },
    constraints=constraints
)
```

#### **Scenario Optimization**
Optimize across multiple scenarios:

```python
scenarios = {
    'optimistic': {'market_growth': 1.1, 'competition': 0.9},
    'base': {'market_growth': 1.0, 'competition': 1.0},
    'pessimistic': {'market_growth': 0.9, 'competition': 1.1}
}

robust_result = optimizer.scenario_optimize(
    scenarios=scenarios,
    risk_preference='robust',  # robust, risk_neutral, risk_seeking
    confidence_level=0.95
)
```

### Performance Optimization

Atlas includes several features to improve optimization performance:

#### **Caching**
```python
from atlas.caching import ModelCache

cached_model = ModelCache(
    base_model=expensive_model,
    cache_size=1000,
    ttl=3600  # Cache for 1 hour
)
```

#### **Parallel Execution**
```python
# Automatic parallelization for applicable optimizers
optimizer = OptunaOptimizer(
    model=model,
    config={'n_jobs': -1}  # Use all available cores
)
```

#### **Warm Starting**
```python
# Use previous solution as starting point
result = optimizer.optimize(
    initial_budget=previous_optimal_budget,
    warm_start=True
)
```

---

## Constraints

Constraints in Atlas define the business rules and limitations that any optimal solution must satisfy. They ensure that optimization results are feasible and aligned with business requirements. [Constraints](constraints.md)

### Constraint Types

Atlas supports various types of constraints to model real-world business requirements:

#### **Budget Constraints**

**Total Budget Limits:**
```python
constraints = {
    'total_budget': 1_000_000,  # Cannot exceed $1M total spend
    'min_total_budget': 800_000  # Must spend at least $800K
}
```

**Channel-Specific Budgets:**
```python
constraints = {
    'bounds': {
        'digital': (100_000, 500_000),  # Digital: $100K-$500K
        'tv': (200_000, 800_000),       # TV: $200K-$800K
        'radio': (0, 300_000),          # Radio: $0-$300K
        'print': (50_000, 200_000)      # Print: $50K-$200K
    }
}
```

**Percentage Allocations:**
```python
constraints = {
    'percentage_bounds': {
        'digital': (0.3, 0.6),    # 30-60% of total budget
        'traditional': (0.2, 0.5) # 20-50% for traditional channels
    }
}
```

#### **Business Rule Constraints**

**Minimum Spend Requirements:**
```python
constraints = {
    'min_spend_rules': {
        'digital': 150_000,      # Must spend at least $150K on digital
        'brand_building': 300_000 # Must allocate $300K to brand activities
    }
}
```

**Channel Dependencies:**
```python
constraints = {
    'dependency_rules': [
        {
            'type': 'if_then',
            'condition': 'tv_budget > 500000',
            'requirement': 'digital_budget >= 200000'
        },
        {
            'type': 'mutual_exclusive',
            'channels': ['premium_tv', 'basic_tv'],
            'max_active': 1
        }
    ]
}
```

**Geographic Constraints:**
```python
constraints = {
    'geographic_rules': {
        'north_region': {'min_budget': 200_000, 'max_budget': 600_000},
        'south_region': {'min_budget': 150_000, 'max_budget': 500_000},
        'total_regional_balance': 0.1  # Regions within 10% of each other
    }
}
```

#### **Temporal Constraints**

**Seasonal Restrictions:**
```python
constraints = {
    'seasonal_rules': {
        'q4_boost': {
            'months': [10, 11, 12],
            'min_increase': 0.2  # 20% increase in Q4
        },
        'summer_reduction': {
            'months': [6, 7, 8],
            'max_spend_ratio': 0.8  # Reduce spend by 20% in summer
        }
    }
}
```

**Budget Smoothing:**
```python
constraints = {
    'smoothing_rules': {
        'max_month_to_month_change': 0.15,  # Max 15% change between months
        'max_quarter_variance': 0.25        # Quarters within 25% of average
    }
}
```

#### **Performance Constraints**

**ROI Requirements:**
```python
constraints = {
    'performance_targets': {
        'min_roi': 3.0,           # Minimum 3:1 ROI
        'min_channel_roi': {
            'digital': 4.0,       # Higher requirement for digital
            'traditional': 2.5    # Lower requirement for traditional
        }
    }
}
```

**Market Share Limits:**
```python
def market_share_constraint(budget_allocation):
    """Custom constraint function for market share limits."""
    total_market_spend = 50_000_000
    our_spend = budget_allocation.sum()
    market_share = our_spend / total_market_spend
    return 0.25 - market_share  # Must be <= 0 (max 25% market share)

constraints = {
    'custom_constraints': [
        {'type': 'ineq', 'fun': market_share_constraint}
    ]
}
```

### Custom Constraints

Atlas supports custom constraint functions for complex business rules:

```python
def cross_channel_synergy_constraint(budget):
    """Ensure minimum synergy between digital and TV."""
    digital_spend = budget['digital'].sum()
    tv_spend = budget['tv'].sum()
    
    # Require balanced investment for synergy
    ratio = digital_spend / (tv_spend + 1e-6)
    return 2.0 - ratio  # Digital spend should not exceed 2x TV spend

def brand_safety_constraint(budget):
    """Ensure brand safety requirements are met."""
    premium_channels = ['tv', 'premium_digital', 'print']
    premium_spend = sum(budget[ch].sum() for ch in premium_channels if ch in budget)
    total_spend = budget.sum()
    
    premium_ratio = premium_spend / total_spend
    return premium_ratio - 0.4  # At least 40% in premium channels

constraints = {
    'custom_constraints': [
        {'type': 'ineq', 'fun': cross_channel_synergy_constraint},
        {'type': 'ineq', 'fun': brand_safety_constraint}
    ]
}
```

### Constraint Validation

Atlas provides tools to validate and debug constraints:

```python
from atlas.constraints import ConstraintValidator

validator = ConstraintValidator()

# Check constraint feasibility
is_feasible, violations = validator.check_feasibility(
    constraints=constraints,
    budget_bounds=channel_bounds
)

if not is_feasible:
    print("Constraint violations found:")
    for violation in violations:
        print(f"- {violation['constraint']}: {violation['description']}")

# Test constraint with sample budget
test_budget = {'digital': 300_000, 'tv': 500_000, 'radio': 200_000}
constraint_results = validator.evaluate_constraints(
    budget=test_budget,
    constraints=constraints
)
```

### Constraint Relaxation

When constraints are too restrictive, Atlas provides relaxation strategies:

```python
from atlas.constraints import ConstraintRelaxation

relaxer = ConstraintRelaxation()

# Automatic constraint relaxation
relaxed_constraints = relaxer.relax_constraints(
    original_constraints=strict_constraints,
    relaxation_method='penalty',  # penalty, elastic, priority
    relaxation_factor=0.05  # Allow 5% violation
)

# Priority-based relaxation
relaxed_constraints = relaxer.priority_relaxation(
    constraints=constraints,
    priorities={
        'total_budget': 'hard',      # Never relax
        'channel_bounds': 'medium',   # Moderate relaxation allowed
        'roi_targets': 'soft'        # Can be relaxed significantly
    }
)
```


---

## Integration Example

Here's how all four concepts work together in a complete optimization workflow:

```python
from atlas import (
    ModelFactory, OptimizerFactory, 
    ConstraintBuilder, DataLoader
)

# 1. DATA: Load and prepare multi-dimensional data
loader = DataLoader()
historical_data = loader.from_database(
    query="SELECT * FROM marketing_performance",
    dims=['date', 'channel', 'region']
)

# 2. MODEL: Create and validate model
model = ModelFactory.create(
    model_type='xgboost',
    training_data=historical_data,
    target_variable='revenue'
)

# 3. CONSTRAINTS: Define business rules
constraints = ConstraintBuilder() \
    .total_budget(max_budget=5_000_000) \
    .channel_bounds(digital=(500_000, 2_000_000)) \
    .roi_threshold(min_roi=2.5) \
    .build()

# 4. OPTIMIZATION: Find optimal allocation
optimizer = OptimizerFactory.create(
    optimizer_type='optuna',
    model=model,
    n_trials=1000
)

result = optimizer.optimize(
    initial_budget=current_allocation,
    constraints=constraints,
    objectives=['revenue', 'brand_awareness']
)

print(f"Optimal allocation: {result.optimal_budget}")
print(f"Expected outcomes: {result.predictions}")
print(f"Constraint satisfaction: {result.constraint_status}")
```

These core concepts make Atlas the titan of modeling and optimization.