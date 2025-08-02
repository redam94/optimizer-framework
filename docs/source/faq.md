# Frequently Asked Questions (FAQ)

## General Questions

### What is Atlas?

Atlas is a unified platform for optimizing business decisions across diverse models and business scenarios. It enables data-driven organizations to maximize ROI through intelligent resource allocation, supporting any type of predictive model (ML, statistical, or API-based).

### Who should use Atlas?

The framework is designed for:
- **Data Scientists** building optimization solutions
- **Marketing Teams** optimizing budget allocation
- **Operations Managers** allocating resources efficiently
- **Business Analysts** performing scenario analysis
- **Technology Teams** implementing scalable analytics

### How is this different from other optimization tools?

Key differentiators:
- **Model Agnostic**: Works with any predictive model
- **Multi-Objective**: Optimize multiple KPIs simultaneously
- **Production Ready**: Docker support and horizontal scaling
- **Extensible**: Plugin architecture for custom features
- **Industry Specific**: Pre-built strategies for common use cases

### What are the system requirements?

- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space
- Linux, macOS, or Windows 10/11

## Installation and Setup

### How do I install the framework?

```bash
# Basic installation
pip install atlas-optimizer

# Full installation with all features
pip install atlas-optimizer[all]
```

### Can I use this with my existing models?

Yes! The framework is designed to work with any model:

```python
from atlas import ModelWrapper

# Wrap your existing model
wrapped_model = ModelWrapper(
    predict_func=your_model.predict,
    model_type="custom"
)
```

### Do I need Docker?

Docker is optional but recommended for:
- Model isolation and versioning
- Scaling across multiple machines
- Language-agnostic model integration
- Production deployments

### How do I handle authentication for cloud models?

```python
# Example: AWS SageMaker integration
model = CloudModelWrapper(
    endpoint="https://runtime.sagemaker.region.amazonaws.com/endpoints/your-endpoint",
    auth_config={
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region_name": "us-east-1"
    }
)
```

## Model Integration

### How do I integrate a scikit-learn model?

```python
from atlas.models import SklearnModelWrapper
import joblib

# Load your trained model
sklearn_model = joblib.load("model.pkl")

# Create wrapper
model = SklearnModelWrapper(
    model=sklearn_model,
    feature_names=['tv', 'digital', 'radio'],
    target_name='revenue'
)
```

### Can I use R models?

Yes, through the R-Python bridge or Docker:

```python
# Option 1: Using rpy2
from atlas.models import RModelWrapper

r_model = RModelWrapper(
    model_path="model.rds",
    predict_script="predict.R"
)

# Option 2: Docker container
docker_model = DockerModelWrapper(
    image="my-r-model:latest",
    port=8000
)
```

### How do I handle models with preprocessing?

Create a pipeline wrapper:

```python
class PreprocessedModel(AbstractModel):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        # Apply preprocessing
        processed = self.preprocessor.transform(x)
        # Make predictions
        return self.model.predict(processed)
```

### What about deep learning models?

```python
# TensorFlow/Keras example
from atlas.models import TensorFlowModelWrapper

tf_model = TensorFlowModelWrapper(
    model_path="saved_model/",
    input_signature={
        'tv': tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        'digital': tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    }
)

# PyTorch example
from atlas.models import PyTorchModelWrapper

torch_model = PyTorchModelWrapper(
    model_path="model.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

## Optimization

### How do I choose the right optimizer?

| Use Case | Recommended Optimizer | Why |
|----------|---------------------|-----|
| Convex problems | SciPy | Fast, deterministic |
| Black-box models | Optuna | No gradients needed |
| Linear constraints | CVXPY | Guaranteed global optimum |
| Expensive models | Bayesian | Fewer evaluations |
| Multiple objectives | NSGA-II | Pareto frontier |

### Can I optimize over time periods?

Yes, the framework supports temporal optimization:

```python
# Optimize weekly budgets for a quarter
time_periods = pd.date_range('2024-01-01', '2024-03-31', freq='W')

result = optimizer.optimize_temporal(
    periods=time_periods,
    total_budget=12_000_000,
    constraints={
        'weekly_min': 500_000,
        'weekly_max': 1_500_000,
        'carryover_allowed': True
    }
)
```

### How do I handle non-linear constraints?

```python
# Define custom constraint function
def market_share_constraint(budget):
    """Ensure we don't exceed 30% market share"""
    total_market = 10_000_000
    our_spend = sum(budget.values())
    return 0.3 - (our_spend / total_market)  # <= 0 for feasibility

# Add to constraints
constraints = {
    'custom_constraints': [
        {'type': 'ineq', 'fun': market_share_constraint}
    ]
}
```

### What if optimization doesn't converge?

Try these solutions:

1. **Increase iterations**:
   ```python
   config = {'maxiter': 5000, 'tol': 1e-6}
   ```

2. **Change algorithm**:
   ```python
   # Try different methods
   optimizer = ScipyOptimizer(model, method='trust-constr')
   ```

3. **Relax constraints**:
   ```python
   # Add slack variables
   constraints['slack_percentage'] = 0.05  # Allow 5% violation
   ```

4. **Use multi-start**:
   ```python
   # Try multiple starting points
   result = optimizer.multistart_optimize(
       n_starts=10,
       initial_budget=budget,
       constraints=constraints
   )
   ```

## Performance

### How can I speed up optimization?

1. **Use parallel processing**:
   ```python
   optimizer = OptunaOptimizer(
       model=model,
       config={'n_jobs': -1}  # Use all CPU cores
   )
   ```

2. **Enable caching**:
   ```python
   from atlas.utils import CachedModel
   
   cached_model = CachedModel(
       base_model=your_model,
       cache_size=1000
   )
   ```

3. **Reduce model complexity**:
   ```python
   # Use surrogate model for optimization
   surrogate = SurrogateModel(
       base_model=complex_model,
       n_samples=100
   )
   ```

### How much data can the framework handle?

The framework uses Xarray for efficient multi-dimensional data handling:
- Tested with datasets up to 10GB
- Supports lazy loading for larger datasets
- Can optimize 1000+ variables
- Handles millions of constraint evaluations

### Can I use GPU acceleration?

Yes, for supported operations:

```python
# Enable GPU for applicable models
config = {
    'device': 'cuda',
    'gpu_batch_size': 1024
}

# For neural network models
model = NeuralModel(device='cuda')
```

## Deployment

### How do I deploy to production?

1. **Using Docker**:
   ```bash
   docker build -t optimizer-service .
   docker run -p 8000:8000 optimizer-service
   ```

2. **Using Kubernetes**:
   ```yaml
   kubectl apply -f optimizer-deployment.yaml
   ```

3. **As a service**:
   ```python
   from atlas.server import create_app
   
   app = create_app(model, optimizer)
   app.run(host='0.0.0.0', port=8000)
   ```

### How do I monitor optimization jobs?

```python
# Enable monitoring
from atlas.monitoring import MetricsCollector

collector = MetricsCollector()
optimizer.add_callback(collector)

# View metrics
metrics = collector.get_metrics()
print(f"Average optimization time: {metrics['avg_time']}")
print(f"Success rate: {metrics['success_rate']}")
```

### Can I schedule regular optimizations?

Yes, using the scheduling module:

```python
from atlas.scheduling import OptimizationScheduler

scheduler = OptimizationScheduler(optimizer)

# Run daily at 2 AM
scheduler.schedule_daily(
    time="02:00",
    config=optimization_config,
    notification_email="team@company.com"
)
```

## Troubleshooting

### Common Error Messages

#### "Model prediction shape mismatch"
```python
# Solution: Ensure output dimensions match
def predict(self, x):
    predictions = self.model.predict(x)
    return xr.DataArray(
        predictions,
        dims=x.dims,  # Match input dimensions
        coords=x.coords
    )
```

#### "Constraint violation in optimal solution"
```python
# Solution: Check constraint formulation
# Ensure constraints are feasible
validator = ConstraintValidator()
is_feasible = validator.check_feasibility(constraints, initial_budget)
```

#### "Memory error during optimization"
```python
# Solution: Use chunked processing
optimizer = ChunkedOptimizer(
    model=model,
    chunk_size=1000,
    backend='dask'
)
```

### How do I debug optimization issues?

Enable detailed logging:

```python
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('atlas')

# Enable optimization trace
optimizer = OptimizerFactory.create(
    optimizer_type='scipy',
    model=model,
    config={'disp': True, 'trace': True}
)

# Save optimization history
result = optimizer.optimize(budget, constraints)
result.save_trace('optimization_trace.json')
```

## Best Practices

### Should I normalize my data?

Yes, normalization often improves optimization:

```python
from atlas.preprocessing import BudgetNormalizer

normalizer = BudgetNormalizer(method='minmax')
normalized_budget = normalizer.fit_transform(budget)

# Optimize with normalized values
result = optimizer.optimize(normalized_budget, constraints)

# Transform back
optimal_budget = normalizer.inverse_transform(result.optimal_budget)
```

### How often should I retrain models?

Depends on your domain:
- **Stable markets**: Monthly or quarterly
- **Dynamic markets**: Weekly or daily
- **Real-time systems**: Continuous learning

Set up monitoring to detect model drift:

```python
from atlas.monitoring import ModelDriftDetector

detector = ModelDriftDetector(
    reference_data=historical_data,
    threshold=0.1
)

if detector.detect_drift(current_data):
    retrain_model()
```

### What's the best way to handle multiple currencies?

```python
from atlas.utils import CurrencyConverter

converter = CurrencyConverter(base_currency='USD')

# Convert all budgets to base currency
normalized_budgets = {}
for channel, amount in budgets.items():
    currency = channel_currencies[channel]
    normalized_budgets[channel] = converter.convert(
        amount, from_currency=currency, to_currency='USD'
    )

# Optimize in base currency
result = optimizer.optimize(normalized_budgets, constraints)
```

## Advanced Topics

### Can I implement custom optimization algorithms?

Yes, extend the BaseOptimizer class:

```python
from atlas.optimizers import BaseOptimizer

class MyCustomOptimizer(BaseOptimizer):
    def optimize(self, initial_budget, constraints):
        # Your optimization logic here
        return OptimizationResult(
            optimal_budget=optimal,
            optimal_value=value
        )

# Register your optimizer
OptimizerFactory.register('custom', MyCustomOptimizer)
```

### How do I handle uncertainty in predictions?

Use stochastic optimization:

```python
from atlas.stochastic import StochasticOptimizer

# Define uncertainty model
uncertainty_model = GaussianUncertainty(std_dev=0.1)

stochastic_opt = StochasticOptimizer(
    model=model,
    uncertainty=uncertainty_model,
    n_scenarios=1000
)

# Get robust solution
result = stochastic_opt.optimize(
    initial_budget,
    constraints,
    confidence_level=0.95
)
```

### Can I use the framework for non-marketing applications?

Absolutely! The framework is domain-agnostic:

```python
# Supply chain optimization
model = InventoryModel(warehouses, products)
optimizer.optimize(
    initial_allocation={'warehouse_a': 1000, 'warehouse_b': 1500},
    constraints={'total_capacity': 5000}
)

# Resource allocation
model = StaffingModel(departments, skills)
optimizer.optimize(
    initial_staffing={'engineering': 50, 'sales': 30},
    constraints={'total_headcount': 100}
)
```

## Support

### Where can I get help?

1. **Documentation**: [Readthedocs](https://atlas.mattreda.pro)
2. **GitHub Issues**: Report bugs and request features


### How do I report a bug?

1. Check existing issues on GitHub
2. Create a minimal reproducible example
3. Include system information:
   ```python
   import atlas
   atlas.show_versions()
   ```
4. Submit issue with clear description

### Can I contribute to the project?

Yes! We welcome contributions:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See our [Contributing Guide](contributing.md) for details.