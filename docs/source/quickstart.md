# Quick Start Guide

This guide will get you up and running with Atlas in just a few minutes. By the end, you'll have optimized your first budget allocation!

## Prerequisites

- Python 3.10 or higher
- Basic familiarity with Python and data analysis
- (Optional) Docker for containerized model deployment

## Installation

### Install from PyPI

```bash
pip install optimizer-framework
```

### Install from Source

```bash
git clone https://github.com/redam94/optimizer-framework.git
cd optimizer-framework
pip install -e ".[dev]"
```

## Your First Optimization

Let's optimize a simple marketing budget allocation across three channels: TV, Digital, and Radio.

### Step 1: Import Required Components

```python
from optimizer_framework import (
    OptimizationService,
    ModelFactory,
    OptimizerFactory,
    ConfigurationManager
)
from optimizer_framework.models import SimpleLinearModel
import numpy as np
```

### Step 2: Create a Simple Model

For this example, we'll use a built-in linear model. In practice, you'd use your own predictive model.

```python
# Define channel coefficients (impact per dollar spent)
coefficients = {
    'tv': 1.5,      # $1.50 return per $1 spent
    'digital': 2.0,  # $2.00 return per $1 spent
    'radio': 1.2     # $1.20 return per $1 spent
}

# Create model
model = SimpleLinearModel(coefficients)
```

### Step 3: Define Optimization Constraints

Set up your business constraints and budget limits:

```python
# Total budget constraint
total_budget = 1_000_000  # $1M total

# Channel-specific constraints
constraints = {
    'total_budget': total_budget,
    'bounds': {
        'tv': (100_000, 500_000),      # TV: $100k - $500k
        'digital': (200_000, 600_000),  # Digital: $200k - $600k
        'radio': (50_000, 200_000)      # Radio: $50k - $200k
    },
    'business_rules': {
        'min_digital_percentage': 0.3,   # At least 30% to digital
        'max_traditional_percentage': 0.5 # At most 50% to traditional media
    }
}
```

### Step 4: Run Optimization

```python
# Create optimizer
optimizer = OptimizerFactory.create(
    optimizer_type='scipy',
    model=model,
    config={
        'method': 'SLSQP',
        'maxiter': 1000
    }
)

# Run optimization
result = optimizer.optimize(
    initial_budget={'tv': 300_000, 'digital': 400_000, 'radio': 100_000},
    constraints=constraints
)

# Display results
print("Optimal Budget Allocation:")
print(f"TV: ${result.optimal_budget['tv']:,.0f}")
print(f"Digital: ${result.optimal_budget['digital']:,.0f}")
print(f"Radio: ${result.optimal_budget['radio']:,.0f}")
print(f"\nExpected Return: ${result.optimal_value:,.0f}")
```

### Expected Output

```
Optimal Budget Allocation:
TV: $250,000
Digital: $600,000
Radio: $150,000

Expected Return: $1,830,000
```

## Real-World Example: Multi-KPI Optimization

Now let's tackle a more realistic scenario where we optimize for multiple objectives:

```python
from optimizer_framework.strategies import MultiObjectiveStrategy
from optimizer_framework.models import ModelWrapper
import xarray as xr

# Step 1: Load your existing model
class YourRevenueModel(ModelWrapper):
    def __init__(self, model_path):
        # Load your trained model (sklearn, xgboost, etc.)
        self.model = joblib.load(model_path)
    
    def predict(self, budget_data):
        # Your prediction logic
        features = self._prepare_features(budget_data)
        return self.model.predict(features)

# Step 2: Define multiple objectives
objectives = {
    'revenue': {
        'model': YourRevenueModel('models/revenue_model.pkl'),
        'weight': 0.6,
        'direction': 'maximize'
    },
    'brand_awareness': {
        'model': YourAwarenessModel('models/awareness_model.pkl'),
        'weight': 0.3,
        'direction': 'maximize'
    },
    'customer_acquisition': {
        'model': YourCAModel('models/acquisition_model.pkl'),
        'weight': 0.1,
        'direction': 'minimize'  # Minimize cost per acquisition
    }
}

# Step 3: Create multi-objective optimizer
multi_optimizer = MultiObjectiveStrategy(
    objectives=objectives,
    method='weighted_sum'  # or 'pareto' for Pareto optimization
)

# Step 4: Add time dimension for weekly optimization
time_periods = pd.date_range('2024-01-01', periods=52, freq='W')
channels = ['tv', 'digital', 'radio', 'social', 'print']

# Create optimization request with time dimension
optimization_request = {
    'dimensions': {
        'time': time_periods,
        'channel': channels
    },
    'constraints': {
        'total_budget_per_week': 250_000,
        'min_spend_per_channel': 10_000,
        'seasonality': {
            'Q4': 1.5,  # 50% increase in Q4
            'Q1': 0.8   # 20% decrease in Q1
        }
    }
}

# Step 5: Optimize
result = multi_optimizer.optimize(optimization_request)

# Step 6: Visualize results
result.plot_allocation_heatmap()
result.plot_objective_tradeoffs()
result.generate_executive_summary()
```

## Using Docker Models

If your model is containerized, integration is just as simple:

```python
from optimizer_framework.models import DockerModelWrapper

# Connect to dockerized model
model = DockerModelWrapper(
    name="revenue-model",
    version="2.0",
    port=8001
)

# Use exactly the same as any other model
optimizer = OptimizerFactory.create(
    optimizer_type='optuna',
    model=model,
    config={'n_trials': 1000}
)

result = optimizer.optimize(initial_budget, constraints)
```

## Advanced Features Quick Examples

### 1. Scenario Analysis

```python
# Define scenarios
scenarios = {
    'conservative': {
        'total_budget': 800_000,
        'risk_tolerance': 'low',
        'constraints': {'max_channel_percentage': 0.4}
    },
    'aggressive': {
        'total_budget': 1_500_000,
        'risk_tolerance': 'high',
        'constraints': {'min_digital_percentage': 0.6}
    },
    'balanced': {
        'total_budget': 1_000_000,
        'risk_tolerance': 'medium'
    }
}

# Run scenario analysis
scenario_results = optimizer.analyze_scenarios(scenarios)

# Compare results
comparison = scenario_results.compare(
    metrics=['expected_return', 'risk_score', 'efficiency']
)
print(comparison.to_dataframe())
```

### 2. Real-Time Optimization

```python
from optimizer_framework.realtime import RealtimeOptimizer

# Setup real-time optimizer
rt_optimizer = RealtimeOptimizer(
    model=model,
    monitoring_config={
        'metrics': ['ctr', 'conversion_rate', 'spend_pace'],
        'update_frequency': 'hourly',
        'reallocation_threshold': 0.1
    }
)

# Start monitoring and auto-reallocation
rt_optimizer.start(initial_budget, constraints)

# Get current status
status = rt_optimizer.get_status()
print(f"Current performance: {status['performance_vs_target']}")
print(f"Recommended reallocation: {status['recommended_changes']}")
```

### 3. Model Registry Usage

```python
from optimizer_framework import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register your models
registry.register(
    model=YourRevenueModel('model_v2.pkl'),
    name="revenue_model",
    version="2.0",
    tags=["production", "q4_2024"],
    metrics={'rmse': 0.05, 'mape': 0.03}
)

# Use registered model
optimizer = OptimizerFactory.create(
    optimizer_type='scipy',
    model=registry.get_model("revenue_model", version="2.0")
)
```

## Configuration Management

For production deployments, use configuration files:

```yaml
# config/optimizer_config.yaml
model:
  type: "docker"
  name: "revenue-model"
  version: "2.0"
  endpoint: "http://model-service:8000"

optimizer:
  type: "optuna"
  settings:
    n_trials: 2000
    n_jobs: -1
    sampler: "TPE"

constraints:
  total_budget: 1000000
  channels:
    tv:
      min: 100000
      max: 400000
    digital:
      min: 200000
      max: 600000
    radio:
      min: 50000
      max: 200000
  
  business_rules:
    - type: "percentage"
      channel: "digital"
      min: 0.3
    - type: "ratio"
      channels: ["tv", "radio"]
      max_ratio: 3.0
```

Load and use configuration:

```python
from optimizer_framework import ConfigurationManager

# Load configuration
config = ConfigurationManager.load('config/optimizer_config.yaml')

# Create optimizer from config
optimizer = OptimizerFactory.from_config(config)

# Run optimization
result = optimizer.optimize()
```

## Next Steps

Now that you've completed the quick start:

1. **Integrate Your Models**: See the [Model Integration Guide](guides/model_integration.md)
2. **Customize Optimization**: Read the [Optimization Strategies Guide](guides/optimization_strategies.md)
3. **Deploy to Production**: Check out Docker Integration (coming soon)
4. **Explore Examples**: Browse our Example Gallery (coming soon)

## Getting Help

- **Documentation**: Full documentation at [atlas.mattreda.pro](https://atlas.mattreda.pro)
- **GitHub Issues**: Report bugs or request features
- **Community Forum**: Ask questions and share experiences

## Common Issues

### Issue: ImportError
```python
# Solution: Ensure you've installed all dependencies
pip install optimizer-framework[all]
```

### Issue: Model Not Found
```python
# Solution: Check model registration
registry.list_models()  # See all registered models
```

### Issue: Optimization Not Converging
```python
# Solution: Adjust optimizer settings
optimizer = OptimizerFactory.create(
    optimizer_type='scipy',
    model=model,
    config={
        'method': 'trust-constr',  # Try different method
        'maxiter': 5000,           # Increase iterations
        'tol': 1e-8                # Adjust tolerance
    }
)
```

Happy optimizing! 🚀