# Core Classes

This reference covers the main classes and methods in Atlas. For complete API documentation, see the auto-generated docs.

## Table of Contents

- [Models](#models)
  - [AbstractModel](#abstractmodel)
  - [ModelWrapper](#modelwrapper)
  - [DockerModelWrapper](#dockermodelwrapper)
- [Optimizers](#optimizers)
  - [BaseOptimizer](#baseoptimizer)
  - [ScipyOptimizer](#scipyoptimizer)
  - [OptunaOptimizer](#optunaoptimizer)
- [Strategies](#strategies)
  - [BaseOptimizationStrategy](#baseoptimizationstrategy)
  - [MultiObjectiveStrategy](#multiobjectivestrategy)
- [Configuration](#configuration)
  - [ConfigurationManager](#configurationmanager)
  - [ModelConfiguration](#modelconfiguration)
- [Results](#results)
  - [OptimizationResult](#optimizationresult)
- [Utilities](#utilities)

---

## Models

### AbstractModel

Base class for all models in the framework.

```python
from optimizer_framework.models import AbstractModel
```

#### Class Definition

```python
class AbstractModel(ABC):
    """Abstract base class for all optimization models."""
    
    @abstractmethod
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Generate predictions from input data."""
        pass
    
    @abstractmethod
    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Calculate feature contributions."""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier."""
        pass
    
    @property
    @abstractmethod
    def required_dimensions(self) -> List[str]:
        """Return required data dimensions."""
        pass
```

#### Methods

##### predict(x: xr.Dataset) -> xr.DataArray

Generate predictions from input data.

**Parameters:**
- `x` (xr.Dataset): Input dataset containing budget allocations and other features

**Returns:**
- `xr.DataArray`: Predictions with appropriate dimensions and coordinates

**Example:**
```python
# Create input data
budget_data = xr.Dataset({
    'tv': xr.DataArray([100000], dims=['time']),
    'digital': xr.DataArray([200000], dims=['time']),
    'radio': xr.DataArray([50000], dims=['time'])
})

# Get predictions
predictions = model.predict(budget_data)
print(f"Predicted outcome: {predictions.values}")
```

##### contributions(x: xr.Dataset) -> xr.Dataset

Calculate individual feature contributions to the prediction.

**Parameters:**
- `x` (xr.Dataset): Input dataset

**Returns:**
- `xr.Dataset`: Dataset containing contribution values for each feature

**Example:**
```python
contributions = model.contributions(budget_data)
for var in contributions.data_vars:
    print(f"{var} contribution: {float(contributions[var])}")
```

---

### ModelWrapper

Convenience wrapper for integrating existing models.

```python
from optimizer_framework.models import ModelWrapper
```

#### Class Definition

```python
class ModelWrapper(AbstractModel):
    """Wrapper class for easy model integration."""
    
    def __init__(self, 
                 predict_func: Callable,
                 contribution_func: Optional[Callable] = None,
                 model_type: str = "custom",
                 required_dimensions: List[str] = None):
        """
        Initialize model wrapper.
        
        Args:
            predict_func: Function that generates predictions
            contribution_func: Optional function for contributions
            model_type: Model type identifier
            required_dimensions: List of required dimensions
        """
```

#### Example Usage

```python
# Wrap existing sklearn model
from sklearn.ensemble import RandomForestRegressor

# Train your model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Create wrapper
def predict_func(x: xr.Dataset) -> xr.DataArray:
    # Convert xarray to features
    features = prepare_features(x)
    predictions = rf_model.predict(features)
    return xr.DataArray(predictions, dims=['time'])

wrapped_model = ModelWrapper(
    predict_func=predict_func,
    model_type="random_forest",
    required_dimensions=['time', 'channel']
)
```

---

### DockerModelWrapper

Wrapper for models deployed as Docker containers.

```python
from optimizer_framework.models import DockerModelWrapper
```

#### Class Definition

```python
class DockerModelWrapper(AbstractModel):
    """Wrapper for dockerized models."""
    
    def __init__(self,
                 name: str,
                 version: str,
                 endpoint: str = None,
                 port: int = 8000,
                 timeout: int = 30):
        """
        Initialize Docker model wrapper.
        
        Args:
            name: Model name
            version: Model version
            endpoint: API endpoint (default: http://localhost:{port})
            port: Service port
            timeout: Request timeout in seconds
        """
```

#### Methods

##### health_check() -> bool

Check if the model service is healthy.

**Returns:**
- `bool`: True if service is healthy

##### get_schema() -> Dict

Get model input/output schema.

**Returns:**
- `Dict`: Schema definition

#### Example Usage

```python
# Connect to dockerized model
docker_model = DockerModelWrapper(
    name="revenue-predictor",
    version="2.0",
    port=8001
)

# Check health
if docker_model.health_check():
    print("Model service is healthy")

# Use like any other model
predictions = docker_model.predict(budget_data)
```

---

## Optimizers

### BaseOptimizer

Abstract base class for optimization algorithms.

```python
from optimizer_framework.optimizers import BaseOptimizer
```

#### Class Definition

```python
class BaseOptimizer(ABC):
    """Base class for all optimizers."""
    
    def __init__(self,
                 model: AbstractModel,
                 config: Dict[str, Any] = None):
        """
        Initialize optimizer.
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
        """
    
    @abstractmethod
    def optimize(self,
                 initial_budget: Dict[str, float],
                 constraints: Dict[str, Any]) -> OptimizationResult:
        """Run optimization."""
        pass
```

---

### ScipyOptimizer

Optimizer using SciPy optimization algorithms.

```python
from optimizer_framework.optimizers import ScipyOptimizer
```

#### Class Definition

```python
class ScipyOptimizer(BaseOptimizer):
    """SciPy-based optimizer for convex problems."""
    
    def __init__(self,
                 model: AbstractModel,
                 method: str = 'SLSQP',
                 config: Dict[str, Any] = None):
        """
        Initialize SciPy optimizer.
        
        Args:
            model: Model to optimize
            method: Optimization method ('SLSQP', 'trust-constr', 'L-BFGS-B')
            config: Additional configuration
        """
```

#### Configuration Options

```python
config = {
    'tol': 1e-6,           # Tolerance for convergence
    'maxiter': 1000,       # Maximum iterations
    'disp': True,          # Display convergence messages
    'adaptive': True,      # Use adaptive step sizes
    'finite_diff_rel_step': None  # Step size for finite differences
}
```

#### Example Usage

```python
# Create optimizer
scipy_opt = ScipyOptimizer(
    model=my_model,
    method='trust-constr',
    config={'maxiter': 2000}
)

# Define constraints
constraints = {
    'total_budget': 1_000_000,
    'bounds': {
        'tv': (50_000, 400_000),
        'digital': (100_000, 600_000),
        'radio': (20_000, 100_000)
    }
}

# Run optimization
result = scipy_opt.optimize(
    initial_budget={'tv': 200_000, 'digital': 300_000, 'radio': 50_000},
    constraints=constraints
)

print(f"Optimal allocation: {result.optimal_budget}")
print(f"Expected outcome: {result.optimal_value}")
```

---

### OptunaOptimizer

Optimizer using Optuna for black-box optimization.

```python
from optimizer_framework.optimizers import OptunaOptimizer
```

#### Class Definition

```python
class OptunaOptimizer(BaseOptimizer):
    """Optuna-based optimizer for black-box optimization."""
    
    def __init__(self,
                 model: AbstractModel,
                 config: Dict[str, Any] = None):
        """
        Initialize Optuna optimizer.
        
        Args:
            model: Model to optimize
            config: Optuna configuration
        """
```

#### Configuration Options

```python
config = {
    'n_trials': 1000,              # Number of trials
    'n_jobs': -1,                  # Parallel jobs (-1 for all cores)
    'sampler': 'TPE',              # Sampling algorithm
    'pruner': 'MedianPruner',      # Pruning algorithm
    'study_name': 'optimization',   # Study name
    'storage': None,               # Database URL for distributed optimization
    'load_if_exists': False,       # Load existing study
    'direction': 'maximize'        # 'maximize' or 'minimize'
}
```

#### Example Usage

```python
# Create optimizer with parallel execution
optuna_opt = OptunaOptimizer(
    model=my_model,
    config={
        'n_trials': 2000,
        'n_jobs': 8,
        'sampler': 'TPE',
        'pruner': 'HyperbandPruner'
    }
)

# Run optimization
result = optuna_opt.optimize(initial_budget, constraints)

# Access study object for analysis
study = optuna_opt.study
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")

# Visualize optimization history
from optuna.visualization import plot_optimization_history
fig = plot_optimization_history(study)
fig.show()
```

---

## Strategies

### BaseOptimizationStrategy

Base class for optimization strategies.

```python
from optimizer_framework.strategies import BaseOptimizationStrategy
```

#### Class Definition

```python
class BaseOptimizationStrategy(ABC):
    """Base class for optimization strategies."""
    
    @abstractmethod
    def optimize(self,
                 model: AbstractModel,
                 initial_budget: Dict[str, float],
                 constraints: Dict[str, Any]) -> OptimizationResult:
        """Execute optimization strategy."""
        pass
```

---

### MultiObjectiveStrategy

Strategy for optimizing multiple objectives.

```python
from optimizer_framework.strategies import MultiObjectiveStrategy
```

#### Class Definition

```python
class MultiObjectiveStrategy(BaseOptimizationStrategy):
    """Multi-objective optimization strategy."""
    
    def __init__(self,
                 objectives: Dict[str, Dict[str, Any]],
                 method: str = 'weighted_sum'):
        """
        Initialize multi-objective strategy.
        
        Args:
            objectives: Dictionary of objectives with weights
            method: Optimization method ('weighted_sum', 'pareto', 'lexicographic')
        """
```

#### Example Usage

```python
# Define multiple objectives
objectives = {
    'revenue': {
        'model': revenue_model,
        'weight': 0.6,
        'direction': 'maximize'
    },
    'brand_awareness': {
        'model': awareness_model,
        'weight': 0.3,
        'direction': 'maximize'
    },
    'cost_efficiency': {
        'model': efficiency_model,
        'weight': 0.1,
        'direction': 'maximize'
    }
}

# Create strategy
multi_strategy = MultiObjectiveStrategy(
    objectives=objectives,
    method='weighted_sum'
)

# Use with optimizer
optimizer = OptimizerFactory.create(
    optimizer_type='scipy',
    model=None,  # Strategy handles models
    strategy=multi_strategy
)

result = optimizer.optimize(initial_budget, constraints)
```

---

## Configuration

### ConfigurationManager

Manages configuration loading and validation.

```python
from optimizer_framework.config import ConfigurationManager
```

#### Class Definition

```python
class ConfigurationManager:
    """Configuration management system."""
    
    @staticmethod
    def load(path: Union[str, Path]) -> Config:
        """Load configuration from file."""
        pass
    
    @staticmethod
    def validate(config: Dict) -> bool:
        """Validate configuration."""
        pass
    
    @staticmethod
    def merge(base: Config, override: Config) -> Config:
        """Merge configurations."""
        pass
```

#### Example Usage

```python
# Load configuration
config = ConfigurationManager.load('config/optimizer.yaml')

# Validate custom config
custom_config = {
    'model': {'type': 'docker', 'name': 'my-model'},
    'optimizer': {'type': 'optuna', 'n_trials': 1000}
}
is_valid = ConfigurationManager.validate(custom_config)

# Merge configurations
env_config = ConfigurationManager.load('config/production.yaml')
final_config = ConfigurationManager.merge(config, env_config)
```

---

### ModelConfiguration

Configuration class for models.

```python
from optimizer_framework.config import ModelConfiguration
```

#### Class Definition

```python
class ModelConfiguration:
    """Model configuration specification."""
    
    def __init__(self,
                 model_name: str,
                 model_version: str,
                 model_type: str,
                 levers: Dict[str, LeverSpecification],
                 outputs: Dict[str, OutputSpecification],
                 constraints: Dict[str, Any] = None):
        """Initialize model configuration."""
```

---

## Results

### OptimizationResult

Container for optimization results.

```python
from optimizer_framework.core import OptimizationResult
```

#### Class Definition

```python
class OptimizationResult:
    """Container for optimization results."""
    
    optimal_budget: Dict[str, float]
    optimal_value: float
    predictions: Optional[xr.DataArray]
    contributions: Optional[xr.Dataset]
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        pass
    
    def plot_allocation(self, **kwargs) -> plt.Figure:
        """Plot budget allocation."""
        pass
    
    def generate_report(self, template: str = None) -> str:
        """Generate text report."""
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to file."""
        pass
```

#### Properties and Methods

##### improvement

Calculate improvement over initial budget.

```python
improvement = result.improvement  # Absolute improvement
improvement_pct = result.improvement_percentage  # Percentage improvement
```

##### to_dict() -> Dict

Convert result to dictionary.

```python
result_dict = result.to_dict()
print(json.dumps(result_dict, indent=2))
```

##### compare(other: OptimizationResult) -> pd.DataFrame

Compare with another result.

```python
baseline_result = optimizer.optimize(baseline_budget, constraints)
improved_result = optimizer.optimize(improved_budget, constraints)

comparison = improved_result.compare(baseline_result)
print(comparison)
```

---

## Utilities

### Factory Classes

#### OptimizerFactory

Factory for creating optimizers.

```python
from optimizer_framework import OptimizerFactory

# Create optimizer by type
optimizer = OptimizerFactory.create(
    optimizer_type='optuna',  # 'scipy', 'optuna', 'cvxpy'
    model=my_model,
    config={'n_trials': 1000}
)

# Create from configuration
optimizer = OptimizerFactory.from_config(config)

# List available optimizers
available = OptimizerFactory.list_available()
print(f"Available optimizers: {available}")
```

#### ModelFactory

Factory for creating models.

```python
from optimizer_framework import ModelFactory

# Create model by type
model = ModelFactory.create(
    model_type='sklearn',
    model_path='models/revenue_model.pkl',
    config={'features': ['tv', 'digital', 'radio']}
)

# Register custom model type
ModelFactory.register('custom_type', MyCustomModel)
```

### Data Utilities

#### BudgetConverter

Convert between different budget representations.

```python
from optimizer_framework.utils import BudgetConverter

# Convert dictionary to xarray
budget_dict = {'tv': 100000, 'digital': 200000}
budget_xr = BudgetConverter.dict_to_xarray(
    budget_dict,
    dims=['time', 'channel']
)

# Convert xarray to dataframe
budget_df = BudgetConverter.xarray_to_dataframe(budget_xr)
```

#### ConstraintValidator

Validate constraint specifications.

```python
from optimizer_framework.utils import ConstraintValidator

# Validate constraints
constraints = {
    'total_budget': 1_000_000,
    'bounds': {'tv': (0, 500_000)}
}

validator = ConstraintValidator()
is_valid, errors = validator.validate(constraints)

if not is_valid:
    print(f"Constraint errors: {errors}")
```

### Visualization Utilities

#### OptimizationVisualizer

Create standard visualizations.

```python
from optimizer_framework.viz import OptimizationVisualizer

viz = OptimizationVisualizer()

# Plot optimization history
fig = viz.plot_convergence(optimizer.history)

# Plot budget allocation
fig = viz.plot_allocation(result.optimal_budget)

# Plot ROI by channel
fig = viz.plot_roi_analysis(model, result)

# Create dashboard
dashboard = viz.create_dashboard(results_list)
dashboard.show()
```

---

## Error Handling

### Custom Exceptions

```python
from optimizer_framework.exceptions import (
    OptimizationError,
    ModelError,
    ConstraintViolationError,
    ConfigurationError
)

try:
    result = optimizer.optimize(budget, constraints)
except ConstraintViolationError as e:
    print(f"Constraint violation: {e.constraint_name}")
    print(f"Violation amount: {e.violation_amount}")
except OptimizationError as e:
    print(f"Optimization failed: {e.message}")
    print(f"Iteration: {e.iteration}")
```

---

## Best Practices

### 1. Model Integration

```python
# Good: Implement all required methods
class MyModel(AbstractModel):
    @property
    def model_type(self) -> str:
        return "custom_regression"
    
    @property
    def required_dimensions(self) -> List[str]:
        return ["time", "channel", "geography"]
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        # Validate input
        self._validate_input(x)
        # Make predictions
        return self._internal_predict(x)
    
    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        # Calculate contributions
        return self._calculate_contributions(x)
```

### 2. Error Handling

```python
# Good: Comprehensive error handling
def optimize_with_fallback(optimizer, budget, constraints):
    try:
        # Try primary optimizer
        result = optimizer.optimize(budget, constraints)
    except OptimizationError:
        # Fall back to simpler method
        fallback_optimizer = ScipyOptimizer(
            optimizer.model,
            method='SLSQP'
        )
        result = fallback_optimizer.optimize(budget, constraints)
    
    return result
```

### 3. Configuration Management

```python
# Good: Environment-specific configuration
import os

env = os.getenv('OPTIMIZER_ENV', 'development')
base_config = ConfigurationManager.load('config/base.yaml')
env_config = ConfigurationManager.load(f'config/{env}.yaml')
config = ConfigurationManager.merge(base_config, env_config)
```

---

## Version Information

To check the installed version and dependencies:

```python
import optimizer_framework

# Version
print(f"Version: {optimizer_framework.__version__}")

# Available features
print(f"Features: {optimizer_framework.list_features()}")

# Check dependencies
optimizer_framework.check_dependencies()
```