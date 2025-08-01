# Performance Tuning Guide

This guide provides comprehensive strategies for optimizing the performance of your Atlas deployments, from model integration to large-scale optimization runs.

## Table of Contents

1. [Performance Profiling](#performance-profiling)
2. [Model Optimization](#model-optimization)
3. [Optimizer Performance](#optimizer-performance)
4. [Data Handling](#data-handling)
5. [Parallel Processing](#parallel-processing)
6. [Caching Strategies](#caching-strategies)
7. [Memory Management](#memory-management)
8. [Infrastructure Optimization](#infrastructure-optimization)
9. [Monitoring and Benchmarking](#monitoring-and-benchmarking)

## Performance Profiling

### Identifying Bottlenecks

Before optimizing, profile your code to identify bottlenecks:

```python
import cProfile
import pstats
from optimizer_framework import OptimizationService

# Profile optimization run
profiler = cProfile.Profile()
profiler.enable()

# Run optimization
service = OptimizationService(model, optimizer)
result = service.optimize(budget, constraints)

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time-consuming functions
```

### Using Line Profiler

For detailed line-by-line profiling:

```python
# Install: pip install line_profiler

from line_profiler import LineProfiler
import optimizer_framework.models

lp = LineProfiler()
lp.add_function(model.predict)
lp.add_function(optimizer._evaluate_objective)

# Run with profiling
lp_wrapper = lp(optimizer.optimize)
result = lp_wrapper(budget, constraints)

lp.print_stats()
```

### Memory Profiling

```python
# Install: pip install memory_profiler

from memory_profiler import profile

@profile
def memory_intensive_optimization():
    model = LargeModel()
    optimizer = OptimizerFactory.create('scipy', model)
    return optimizer.optimize(budget, constraints)

# Run with: python -m memory_profiler your_script.py
```

## Model Optimization

### Optimize Model Predictions

#### 1. Vectorized Operations

```python
# Bad: Loop-based prediction
class SlowModel(AbstractModel):
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        results = []
        for i in range(len(x.time)):
            for j in range(len(x.channel)):
                value = self._compute_single(x.isel(time=i, channel=j))
                results.append(value)
        return xr.DataArray(results)

# Good: Vectorized operations
class FastModel(AbstractModel):
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        # Process entire array at once
        features = self._prepare_features(x)
        predictions = self.model.predict(features)
        return xr.DataArray(
            predictions.reshape(x.dims['time'], x.dims['channel']),
            dims=['time', 'channel'],
            coords=x.coords
        )
```

#### 2. Model Simplification

```python
# Use surrogate models for expensive computations
from optimizer_framework.models import SurrogateModel

class EfficientSurrogate(SurrogateModel):
    def __init__(self, complex_model, n_samples=1000):
        # Train simpler model on complex model outputs
        X_sample = self._generate_sample_inputs(n_samples)
        y_sample = complex_model.predict_batch(X_sample)
        
        # Fit fast approximation
        from sklearn.ensemble import RandomForestRegressor
        self.surrogate = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        self.surrogate.fit(X_sample, y_sample)
    
    def predict(self, x):
        # 10-100x faster than complex model
        return self.surrogate.predict(x)
```

#### 3. Model Caching

```python
from functools import lru_cache
import hashlib

class CachedModel(AbstractModel):
    def __init__(self, base_model, cache_size=128):
        self.base_model = base_model
        self.cache = {}
        self.cache_size = cache_size
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        # Create cache key from input
        cache_key = self._hash_input(x)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute and cache
        result = self.base_model.predict(x)
        
        # LRU eviction
        if len(self.cache) >= self.cache_size:
            oldest = min(self.cache.items(), key=lambda x: x[1]['time'])
            del self.cache[oldest[0]]
        
        self.cache[cache_key] = {'result': result, 'time': time.time()}
        return result
    
    def _hash_input(self, x: xr.Dataset) -> str:
        # Create deterministic hash of input
        data_bytes = x.to_netcdf()
        return hashlib.sha256(data_bytes).hexdigest()
```

### Batch Processing

```python
class BatchOptimizedModel(AbstractModel):
    def __init__(self, base_model, batch_size=32):
        self.base_model = base_model
        self.batch_size = batch_size
    
    def predict_batch(self, x_list: List[xr.Dataset]) -> List[xr.DataArray]:
        """Efficiently process multiple predictions."""
        results = []
        
        # Process in batches
        for i in range(0, len(x_list), self.batch_size):
            batch = x_list[i:i + self.batch_size]
            
            # Stack into single array for GPU processing
            stacked = xr.concat(batch, dim='batch')
            
            # Single model call
            batch_predictions = self.base_model.predict(stacked)
            
            # Split results
            for j in range(len(batch)):
                results.append(batch_predictions.isel(batch=j))
        
        return results
```

## Optimizer Performance

### Algorithm Selection

Choose the right optimizer for your problem:

| Problem Type | Recommended Optimizer | Typical Speed |
|--------------|----------------------|---------------|
| Convex, smooth | SciPy (L-BFGS-B) | Very Fast |
| Convex with constraints | CVXPY | Fast |
| Non-convex, differentiable | SciPy (trust-constr) | Moderate |
| Black-box, few variables | Optuna (TPE) | Moderate |
| Black-box, many variables | Optuna (CMA-ES) | Slow |
| Mixed-integer | OR-Tools | Varies |

### Optimizer Configuration

#### SciPy Optimization

```python
# Fast configuration for convex problems
fast_scipy_config = {
    'method': 'L-BFGS-B',
    'options': {
        'ftol': 1e-6,      # Looser tolerance for speed
        'gtol': 1e-5,      
        'maxiter': 100,    # Limit iterations
        'maxfun': 200,     # Limit function evaluations
        'iprint': -1       # Disable output
    }
}

# Robust configuration for difficult problems
robust_scipy_config = {
    'method': 'trust-constr',
    'options': {
        'xtol': 1e-8,
        'gtol': 1e-8,
        'maxiter': 1000,
        'verbose': 0,
        'initial_tr_radius': 1.0,
        'factorization_method': 'SVDFactorization'
    }
}
```

#### Optuna Optimization

```python
# Parallel Optuna configuration
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna_config = {
    'n_trials': 1000,
    'n_jobs': -1,  # Use all CPU cores
    'sampler': TPESampler(
        n_startup_trials=10,
        n_ei_candidates=24,
        multivariate=True,
        constant_liar=True  # Better parallelization
    ),
    'pruner': MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
}

# Create study with distributed optimization
study = optuna.create_study(
    storage='postgresql://user:pass@localhost/optuna',
    study_name='distributed_optimization',
    load_if_exists=True,
    direction='maximize'
)
```

### Early Stopping

Implement early stopping to save computation:

```python
class EarlyStoppingOptimizer(BaseOptimizer):
    def __init__(self, model, patience=10, min_delta=1e-4):
        super().__init__(model)
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('-inf')
        self.patience_counter = 0
    
    def optimize(self, initial_budget, constraints):
        for iteration in range(self.max_iterations):
            # Evaluate current solution
            current_value = self._evaluate_objective(current_solution)
            
            # Check for improvement
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at iteration {iteration}")
                break
        
        return self._create_result(best_solution, self.best_value)
```

## Data Handling

### Efficient Data Structures

#### Use Xarray Efficiently

```python
# Bad: Converting between formats repeatedly
def inefficient_processing(data_dict):
    df = pd.DataFrame(data_dict)
    array = df.to_numpy()
    xr_data = xr.DataArray(array)
    return xr_data

# Good: Work directly with xarray
def efficient_processing(data_dict):
    return xr.Dataset({
        k: xr.DataArray(v, dims=['time', 'channel'])
        for k, v in data_dict.items()
    })
```

#### Lazy Loading

```python
# For large datasets, use dask
import dask.array as da

class LazyModel(AbstractModel):
    def __init__(self, model_path):
        # Don't load data into memory yet
        self.data = xr.open_dataset(
            model_path,
            chunks={'time': 100, 'channel': 10}
        )
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        # Computation happens only when needed
        result = self.data.lazy_compute(x)
        return result.compute()  # Force computation
```

### Data Preprocessing

```python
class OptimizedPreprocessor:
    def __init__(self):
        # Precompute expensive operations
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False)
    
    @lru_cache(maxsize=128)
    def preprocess(self, data_hash):
        """Cache preprocessing results."""
        # Expensive preprocessing cached
        return self._actual_preprocess(data_hash)
    
    def transform_batch(self, data_list):
        """Batch preprocessing for efficiency."""
        # Stack all data
        stacked = np.vstack(data_list)
        
        # Single transformation
        transformed = self.scaler.transform(stacked)
        
        # Split back
        sizes = [len(d) for d in data_list]
        return np.split(transformed, np.cumsum(sizes)[:-1])
```

## Parallel Processing

### Multi-threaded Optimization

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class ParallelOptimizer:
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or mp.cpu_count()
    
    def optimize_parallel_scenarios(self, scenarios):
        """Run multiple optimization scenarios in parallel."""
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self._optimize_single, scenario): scenario
                for scenario in scenarios
            }
            
            # Collect results as they complete
            results = {}
            for future in concurrent.futures.as_completed(futures):
                scenario = futures[future]
                try:
                    result = future.result()
                    results[scenario['name']] = result
                except Exception as e:
                    logger.error(f"Scenario {scenario['name']} failed: {e}")
        
        return results
    
    def _optimize_single(self, scenario):
        """Optimize a single scenario."""
        optimizer = OptimizerFactory.create(
            scenario['optimizer_type'],
            scenario['model']
        )
        return optimizer.optimize(
            scenario['budget'],
            scenario['constraints']
        )
```

### GPU Acceleration

```python
# For models that support GPU
import torch
import cupy as cp

class GPUAcceleratedModel(AbstractModel):
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        # Convert to tensor
        tensor_input = torch.from_numpy(x.to_array().values).float()
        tensor_input = tensor_input.to(self.device)
        
        # GPU prediction
        with torch.no_grad():
            predictions = self.model(tensor_input)
        
        # Convert back
        return xr.DataArray(
            predictions.cpu().numpy(),
            dims=x.dims,
            coords=x.coords
        )
```

### Distributed Optimization

```python
# Using Ray for distributed optimization
import ray
from ray import tune

@ray.remote
class DistributedOptimizer:
    def __init__(self, model):
        self.model = model
        self.optimizer = OptimizerFactory.create('scipy', model)
    
    def optimize(self, budget, constraints):
        return self.optimizer.optimize(budget, constraints)

# Initialize Ray
ray.init(address='ray://head-node:10001')

# Create distributed optimizers
optimizers = [DistributedOptimizer.remote(model) for _ in range(10)]

# Run parallel optimizations
futures = [
    optimizer.optimize.remote(budget, constraints)
    for optimizer in optimizers
]

# Get results
results = ray.get(futures)
```

## Caching Strategies

### Multi-level Caching

```python
from functools import lru_cache
import redis
import pickle

class MultiLevelCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.memory_cache = {}
        self.redis_client = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
    
    def get(self, key):
        # L1: Memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # L2: Redis cache
        redis_value = self.redis_client.get(key)
        if redis_value:
            value = pickle.loads(redis_value)
            self.memory_cache[key] = value  # Promote to L1
            return value
        
        return None
    
    def set(self, key, value, ttl=3600):
        # Store in both levels
        self.memory_cache[key] = value
        self.redis_client.setex(
            key,
            ttl,
            pickle.dumps(value)
        )

class CachedOptimizer(BaseOptimizer):
    def __init__(self, model, cache):
        super().__init__(model)
        self.cache = cache
    
    def optimize(self, budget, constraints):
        # Create cache key
        cache_key = self._create_cache_key(budget, constraints)
        
        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Cache hit for optimization")
            return cached_result
        
        # Run optimization
        result = super().optimize(budget, constraints)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
```

### Smart Cache Invalidation

```python
class SmartCache:
    def __init__(self, tolerance=0.01):
        self.cache = {}
        self.tolerance = tolerance
    
    def get_similar(self, key, data):
        """Get cached result for similar input."""
        for cached_key, cached_data in self.cache.items():
            if self._is_similar(data, cached_data['input']):
                logger.info(f"Found similar cached result")
                return cached_data['result']
        return None
    
    def _is_similar(self, data1, data2):
        """Check if two inputs are similar enough."""
        if set(data1.keys()) != set(data2.keys()):
            return False
        
        for key in data1.keys():
            if abs(data1[key] - data2[key]) / data2[key] > self.tolerance:
                return False
        
        return True
```

## Memory Management

### Memory-Efficient Data Loading

```python
# Use generators for large datasets
def data_generator(file_path, chunk_size=1000):
    """Load data in chunks to save memory."""
    with pd.read_csv(file_path, chunksize=chunk_size) as reader:
        for chunk in reader:
            # Process chunk
            processed = preprocess_chunk(chunk)
            yield processed

# Memory-efficient model training
class MemoryEfficientModel:
    def fit_generator(self, data_gen, steps):
        for i, batch in enumerate(data_gen):
            if i >= steps:
                break
            self.partial_fit(batch)
```

### Garbage Collection

```python
import gc

class MemoryAwareOptimizer(BaseOptimizer):
    def optimize(self, budget, constraints):
        try:
            # Run optimization
            result = super().optimize(budget, constraints)
            
        finally:
            # Force garbage collection after optimization
            gc.collect()
            
            # Clear any caches
            if hasattr(self.model, 'clear_cache'):
                self.model.clear_cache()
        
        return result
```

### Memory Monitoring

```python
import psutil
import os

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.process = psutil.Process(os.getpid())
    
    def check_memory(self):
        """Check current memory usage."""
        mem_info = self.process.memory_info()
        return mem_info.rss
    
    def log_memory_usage(self, stage):
        """Log memory usage at different stages."""
        mem_mb = self.check_memory() / 1024 / 1024
        logger.info(f"Memory usage at {stage}: {mem_mb:.2f} MB")
    
    def ensure_memory_available(self):
        """Ensure enough memory is available."""
        if self.check_memory() > self.threshold_bytes:
            logger.warning("High memory usage detected, clearing caches")
            gc.collect()
            
            if self.check_memory() > self.threshold_bytes:
                raise MemoryError("Insufficient memory for optimization")
```

## Infrastructure Optimization

### Docker Optimization

```dockerfile
# Optimized Dockerfile for model serving
FROM python:3.11-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Final stage
FROM python:3.11-slim-bullseye

# Copy Python packages
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
WORKDIR /app
COPY . /app

# Use CPU optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Run with optimizations
CMD ["python", "-O", "-m", "optimizer_framework.server"]
```

### Kubernetes Scaling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: optimizer-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: optimizer
  template:
    metadata:
      labels:
        app: optimizer
    spec:
      containers:
      - name: optimizer
        image: optimizer:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: OPTIMIZER_WORKERS
          value: "4"
        - name: OPTIMIZER_CACHE_SIZE
          value: "1000"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: optimizer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: optimizer-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Benchmarking

### Performance Metrics

```python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class PerformanceMetrics:
    optimization_time: float
    model_evaluation_time: float
    constraint_evaluation_time: float
    iterations: int
    function_evaluations: int
    memory_peak_mb: float
    
class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    def monitor_optimization(self, optimizer, budget, constraints):
        """Monitor optimization performance."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Track detailed timings
        optimizer.add_callback('model_eval', self._track_model_time)
        optimizer.add_callback('constraint_eval', self._track_constraint_time)
        
        # Run optimization
        result = optimizer.optimize(budget, constraints)
        
        # Collect metrics
        metrics = PerformanceMetrics(
            optimization_time=time.time() - start_time,
            model_evaluation_time=self.model_time_total,
            constraint_evaluation_time=self.constraint_time_total,
            iterations=result.iterations,
            function_evaluations=result.func_evaluations,
            memory_peak_mb=(psutil.Process().memory_info().rss - start_memory) / 1024 / 1024
        )
        
        self.metrics.append(metrics)
        return result, metrics
    
    def generate_report(self):
        """Generate performance report."""
        df = pd.DataFrame(self.metrics)
        
        report = f"""
Performance Report
==================

Summary Statistics:
{df.describe()}

Performance Breakdown:
- Model Evaluation: {df['model_evaluation_time'].sum():.2f}s ({df['model_evaluation_time'].sum() / df['optimization_time'].sum() * 100:.1f}%)
- Constraint Evaluation: {df['constraint_evaluation_time'].sum():.2f}s ({df['constraint_evaluation_time'].sum() / df['optimization_time'].sum() * 100:.1f}%)
- Other: {(df['optimization_time'].sum() - df['model_evaluation_time'].sum() - df['constraint_evaluation_time'].sum()):.2f}s

Average Performance:
- Time per iteration: {df['optimization_time'].mean() / df['iterations'].mean():.4f}s
- Time per function evaluation: {df['optimization_time'].mean() / df['function_evaluations'].mean():.4f}s
- Memory usage: {df['memory_peak_mb'].mean():.2f} MB (peak)
        """
        
        return report
```

### Continuous Benchmarking

```python
# benchmark_suite.py
class BenchmarkSuite:
    def __init__(self):
        self.benchmarks = []
    
    def add_benchmark(self, name, model, optimizer_config, budget, constraints):
        """Add a benchmark scenario."""
        self.benchmarks.append({
            'name': name,
            'model': model,
            'optimizer_config': optimizer_config,
            'budget': budget,
            'constraints': constraints
        })
    
    def run_benchmarks(self, n_runs=5):
        """Run all benchmarks multiple times."""
        results = {}
        
        for benchmark in self.benchmarks:
            benchmark_results = []
            
            for run in range(n_runs):
                optimizer = OptimizerFactory.create(
                    **benchmark['optimizer_config'],
                    model=benchmark['model']
                )
                
                start = time.time()
                result = optimizer.optimize(
                    benchmark['budget'],
                    benchmark['constraints']
                )
                duration = time.time() - start
                
                benchmark_results.append({
                    'run': run,
                    'duration': duration,
                    'optimal_value': result.optimal_value,
                    'iterations': result.iterations
                })
            
            results[benchmark['name']] = pd.DataFrame(benchmark_results)
        
        return results
    
    def compare_optimizers(self, optimizers, scenario):
        """Compare different optimizers on same problem."""
        comparison = []
        
        for opt_name, opt_config in optimizers.items():
            optimizer = OptimizerFactory.create(**opt_config, model=scenario['model'])
            
            start = time.time()
            result = optimizer.optimize(scenario['budget'], scenario['constraints'])
            duration = time.time() - start
            
            comparison.append({
                'optimizer': opt_name,
                'time': duration,
                'value': result.optimal_value,
                'iterations': result.iterations
            })
        
        return pd.DataFrame(comparison)
```

## Best Practices Summary

### Quick Wins

1. **Enable parallel processing**: Set `n_jobs=-1` in Optuna
2. **Use caching**: Cache model predictions and optimization results
3. **Vectorize operations**: Replace loops with array operations
4. **Adjust tolerances**: Looser tolerances for faster convergence
5. **Batch processing**: Process multiple scenarios together

### Architecture Patterns

1. **Surrogate models**: Use fast approximations for expensive models
2. **Lazy evaluation**: Don't compute until necessary
3. **Pipeline optimization**: Minimize data transformations
4. **Resource pooling**: Reuse expensive objects
5. **Async processing**: Use async/await for I/O operations

### Monitoring Checklist

- [ ] Profile before optimizing
- [ ] Monitor memory usage
- [ ] Track optimization metrics
- [ ] Set up alerts for anomalies
- [ ] Regular benchmark runs
- [ ] Document performance changes

## Performance Troubleshooting

### Common Issues and Solutions

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Slow model predictions | Inefficient implementation | Vectorize operations, use GPU |
| High memory usage | Large intermediate arrays | Use chunking, clear caches |
| Poor parallel scaling | GIL or shared resources | Use ProcessPoolExecutor |
| Optimization doesn't converge | Poor initial guess | Use warm starts, adjust method |
| Erratic performance | Resource contention | Isolate processes, monitor system |

### Debug Performance Issues

```python
# Performance debugging toolkit
class PerformanceDebugger:
    @staticmethod
    def analyze_model(model, sample_data, n_runs=100):
        """Analyze model performance characteristics."""
        timings = []
        
        for _ in range(n_runs):
            start = time.perf_counter()
            model.predict(sample_data)
            timings.append(time.perf_counter() - start)
        
        return {
            'mean_time': np.mean(timings),
            'std_time': np.std(timings),
            'min_time': np.min(timings),
            'max_time': np.max(timings),
            'variance_ratio': np.std(timings) / np.mean(timings)
        }
```

Remember: Always measure before and after optimization to ensure your changes actually improve performance!