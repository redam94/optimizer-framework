# Model Integration Design Guide

## Overview

This guide provides comprehensive instructions for integrating new models into Atlas. Whether you're adding a machine learning model, statistical model, or external API, this guide will help you create robust, maintainable integrations that leverage the full power of the framework.

## Model Integration Principles

Before diving into implementation, understand these core principles:

1. **Separation of Concerns**: Models should focus on prediction, not data transformation
2. **Standardized Interfaces**: All models implement the same abstract interface
3. **Comprehensive Validation**: Input and output validation ensures reliability
4. **Performance Optimization**: Consider caching and parallel execution
5. **Error Handling**: Graceful degradation and informative error messages

## Integration Approaches

### Approach 1: Direct Python Integration

Best for models implemented in Python or easily callable from Python.

#### Step 1: Implement the AbstractModel Interface

```python
from optimizer_framework.models import AbstractModel
import xarray as xr
from typing import Dict, List, Optional

class MyRevenueModel(AbstractModel):
    """
    Example revenue prediction model integration.
    """
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize your model.
        
        Args:
            model_path: Path to model artifacts
            config: Optional configuration dictionary
        """
        super().__init__()
        self.model = self._load_model(model_path)
        self.config = config or {}
        self._validate_config()
    
    def _load_model(self, model_path: str):
        """Load model from disk."""
        # Example: Load scikit-learn model
        import joblib
        return joblib.load(model_path)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = ['channels', 'time_period']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "revenue_regression"
    
    @property
    def required_dimensions(self) -> List[str]:
        """Specify required data dimensions."""
        return ["time", "channel", "geography"]
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """
        Generate predictions from input data.
        
        Args:
            x: Input dataset with budget allocations
            
        Returns:
            Predictions as DataArray
        """
        # Validate input
        self._validate_input(x)
        
        # Transform xarray to model input format
        model_input = self._prepare_input(x)
        
        # Generate predictions
        predictions = self.model.predict(model_input)
        
        # Convert back to xarray
        return self._format_output(predictions, x)
    
    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        """
        Calculate channel contributions.
        
        Args:
            x: Input dataset
            
        Returns:
            Dataset with contribution information
        """
        # Use SHAP or other attribution method
        import shap
        
        explainer = shap.Explainer(self.model)
        model_input = self._prepare_input(x)
        shap_values = explainer(model_input)
        
        return self._format_contributions(shap_values, x)
    
    def _validate_input(self, x: xr.Dataset):
        """Validate input data structure."""
        for dim in self.required_dimensions:
            if dim not in x.dims:
                raise ValueError(f"Missing required dimension: {dim}")
        
        # Check for required variables
        for channel in self.config['channels']:
            if channel not in x.data_vars:
                raise ValueError(f"Missing channel data: {channel}")
    
    def _prepare_input(self, x: xr.Dataset) -> np.ndarray:
        """Transform xarray Dataset to model input format."""
        # Example: Stack data into 2D array
        features = []
        for channel in self.config['channels']:
            features.append(x[channel].values.flatten())
        
        return np.column_stack(features)
    
    def _format_output(self, predictions: np.ndarray, 
                      template: xr.Dataset) -> xr.DataArray:
        """Format predictions as xarray DataArray."""
        # Reshape predictions to match input dimensions
        dims = template.dims
        coords = {dim: template.coords[dim] for dim in dims}
        
        reshaped = predictions.reshape([len(coords[d]) for d in dims])
        
        return xr.DataArray(
            data=reshaped,
            dims=dims,
            coords=coords,
            name="revenue_prediction",
            attrs={"units": "USD", "model": self.model_type}
        )
```

#### Step 2: Create Model Configuration

```python
# model_config.py
from optimizer_framework.config import ModelConfiguration, LeverSpecification

class RevenueModelConfig(ModelConfiguration):
    """Configuration for revenue prediction model."""
    
    def __init__(self):
        super().__init__(
            model_name="Revenue Predictor",
            model_version="2.0",
            model_type="regression"
        )
        
        # Define optimization levers
        self.levers = {
            "tv": LeverSpecification(
                name="tv",
                lever_type="spend",
                baseline_value=100_000,
                min_value=0,
                max_value=500_000,
                units="USD",
                description="Television advertising spend"
            ),
            "digital": LeverSpecification(
                name="digital",
                lever_type="spend",
                baseline_value=150_000,
                min_value=10_000,
                max_value=1_000_000,
                units="USD",
                description="Digital marketing spend"
            ),
            "radio": LeverSpecification(
                name="radio",
                lever_type="spend",
                baseline_value=50_000,
                min_value=0,
                max_value=200_000,
                units="USD",
                description="Radio advertising spend"
            )
        }
        
        # Define data mapping
        self.data_mapping = {
            "tv": {
                "variables": ["tv_grps", "tv_reach"],
                "transformation": "logarithmic",
                "scaling": "min_max"
            },
            "digital": {
                "variables": ["digital_impressions", "digital_clicks"],
                "transformation": "none",
                "scaling": "standard"
            }
        }
        
        # Define output specification
        self.outputs = {
            "revenue": {
                "type": "continuous",
                "dimensions": ["time", "geography"],
                "aggregation": "sum",
                "units": "USD"
            }
        }
```

#### Step 3: Register and Test the Model

```python
# test_model_integration.py
import pytest
from optimizer_framework import ModelRegistry
from mymodels import MyRevenueModel, RevenueModelConfig

def test_model_integration():
    """Test model integration with framework."""
    
    # Initialize model
    model = MyRevenueModel(
        model_path="models/revenue_model.pkl",
        config={"channels": ["tv", "digital", "radio"], 
                "time_period": "weekly"}
    )
    
    # Create test data
    test_data = create_test_dataset()
    
    # Test prediction
    predictions = model.predict(test_data)
    assert predictions.dims == ("time", "channel", "geography")
    assert predictions.min() >= 0  # Revenue should be non-negative
    
    # Test contributions
    contributions = model.contributions(test_data)
    assert "tv" in contributions.data_vars
    
    # Register model
    registry = ModelRegistry()
    registry.register(
        model=model,
        config=RevenueModelConfig(),
        tags=["production", "revenue"]
    )
```

### Approach 2: Docker Container Integration

Best for models in different languages, complex dependencies, or third-party models.

#### Step 1: Create Model Service

```python
# model_service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import numpy as np

app = FastAPI(title="Revenue Model Service")

class PredictionRequest(BaseModel):
    budget: Dict[str, float]
    options: Dict = {}

class PredictionResponse(BaseModel):
    prediction: float
    metadata: Dict = {}

# Load model at startup
model = load_your_model()

@app.get("/")
def info():
    """Model information endpoint."""
    return {
        "name": "Revenue Model",
        "version": "2.0",
        "type": "regression",
        "supported_channels": ["tv", "digital", "radio"]
    }

@app.get("/schema")
def schema():
    """Model input schema."""
    return {
        "required_variables": ["tv", "digital", "radio"],
        "constraints": {
            "tv": {"min": 0, "max": 500000},
            "digital": {"min": 10000, "max": 1000000},
            "radio": {"min": 0, "max": 200000}
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate prediction."""
    try:
        # Validate input
        validate_budget(request.budget)
        
        # Transform to model input
        model_input = prepare_input(request.budget)
        
        # Generate prediction
        prediction = model.predict(model_input)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            metadata={"model_version": "2.0"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/contributions")
def contributions(request: PredictionRequest):
    """Calculate feature contributions."""
    # Implementation here
    pass
```

#### Step 2: Create Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ ./model/
COPY model_service/ ./model_service/

# Expose port
EXPOSE 8000

# Run service
CMD ["uvicorn", "model_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 3: Create Docker Wrapper

```python
# docker_wrapper.py
from optimizer_framework.models import DockerModelWrapper
import requests
import xarray as xr

class RevenueModelDocker(DockerModelWrapper):
    """Docker wrapper for revenue model."""
    
    def __init__(self, service_url: str = "http://revenue-model:8000"):
        super().__init__(
            name="Revenue Model",
            version="2.0",
            service_url=service_url
        )
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Call Docker service for prediction."""
        # Convert xarray to budget dict
        budget = self._dataset_to_budget(x)
        
        # Call service
        response = requests.post(
            f"{self.service_url}/predict",
            json={"budget": budget}
        )
        response.raise_for_status()
        
        # Convert response to xarray
        result = response.json()
        return self._create_prediction_array(
            result["prediction"], x
        )
```

### Approach 3: External API Integration

Best for cloud services, vendor APIs, or remote models.

#### Step 1: Create API Wrapper

```python
# api_wrapper.py
from optimizer_framework.models import AbstractModel
import requests
from typing import Dict
import xarray as xr

class ExternalAPIModel(AbstractModel):
    """Wrapper for external prediction API."""
    
    def __init__(self, api_key: str, endpoint: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.session = self._create_session()
    
    def _create_session(self):
        """Create authenticated session."""
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        return session
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Call external API for predictions."""
        # Prepare request
        request_data = self._prepare_api_request(x)
        
        # Make API call with retry logic
        response = self._call_api_with_retry(
            self.endpoint + "/predict",
            json=request_data
        )
        
        # Process response
        return self._process_api_response(response, x)
    
    def _call_api_with_retry(self, url: str, **kwargs):
        """Call API with exponential backoff retry."""
        import time
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, **kwargs)
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                
                time.sleep(retry_delay * (2 ** attempt))
```

## Best Practices

### 1. Input Validation

Always validate inputs thoroughly:

```python
def _validate_input(self, x: xr.Dataset):
    """Comprehensive input validation."""
    
    # Check dimensions
    required_dims = self.required_dimensions
    missing_dims = set(required_dims) - set(x.dims)
    if missing_dims:
        raise ValueError(f"Missing dimensions: {missing_dims}")
    
    # Check data types
    for var in x.data_vars:
        if not np.issubdtype(x[var].dtype, np.number):
            raise TypeError(f"Variable {var} must be numeric")
    
    # Check value ranges
    for var, bounds in self.variable_bounds.items():
        if var in x.data_vars:
            min_val, max_val = bounds
            actual_min = float(x[var].min())
            actual_max = float(x[var].max())
            
            if actual_min < min_val or actual_max > max_val:
                raise ValueError(
                    f"{var} values outside bounds [{min_val}, {max_val}]"
                )
```

### 2. Error Handling

Implement comprehensive error handling:

```python
class ModelError(Exception):
    """Base exception for model errors."""
    pass

class ModelPredictionError(ModelError):
    """Error during prediction."""
    pass

class ModelValidationError(ModelError):
    """Input validation error."""
    pass

def predict(self, x: xr.Dataset) -> xr.DataArray:
    """Predict with error handling."""
    try:
        self._validate_input(x)
        result = self._internal_predict(x)
        self._validate_output(result)
        return result
    
    except ValidationError as e:
        raise ModelValidationError(f"Invalid input: {e}")
    
    except Exception as e:
        self.logger.error(f"Prediction failed: {e}")
        raise ModelPredictionError(f"Prediction failed: {e}")
```

### 3. Performance Optimization

Implement caching and batch processing:

```python
from functools import lru_cache
import hashlib

class CachedModel(AbstractModel):
    """Model with prediction caching."""
    
    def __init__(self, base_model, cache_size: int = 128):
        self.base_model = base_model
        self.cache_size = cache_size
        self._cache = {}
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Predict with caching."""
        # Create cache key
        cache_key = self._create_cache_key(x)
        
        # Check cache
        if cache_key in self._cache:
            self.logger.info("Cache hit")
            return self._cache[cache_key]
        
        # Generate prediction
        result = self.base_model.predict(x)
        
        # Update cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[cache_key] = result
        return result
    
    def _create_cache_key(self, x: xr.Dataset) -> str:
        """Create unique cache key for dataset."""
        # Hash the data values
        data_bytes = x.to_netcdf()
        return hashlib.sha256(data_bytes).hexdigest()
```

### 4. Monitoring and Logging

Add comprehensive monitoring:

```python
import logging
from datetime import datetime
import time

class MonitoredModel(AbstractModel):
    """Model with performance monitoring."""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "prediction_count": 0,
            "total_time": 0,
            "errors": 0
        }
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Predict with monitoring."""
        start_time = time.time()
        
        try:
            # Log input characteristics
            self.logger.info(
                f"Prediction request - shape: {x.dims}, "
                f"variables: {list(x.data_vars)}"
            )
            
            # Make prediction
            result = self.base_model.predict(x)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["prediction_count"] += 1
            self.metrics["total_time"] += elapsed
            
            # Log success
            self.logger.info(
                f"Prediction successful - time: {elapsed:.2f}s"
            )
            
            return result
        
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Prediction failed: {e}")
            raise
```

### 5. Testing Strategy

Implement comprehensive tests:

```python
# test_model.py
import pytest
import numpy as np
from mymodel import MyRevenueModel

class TestRevenueModel:
    """Test suite for revenue model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return MyRevenueModel("test_model.pkl")
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return create_sample_dataset(
            channels=["tv", "digital"],
            time_periods=52,
            geographies=["US", "EU"]
        )
    
    def test_prediction_shape(self, model, sample_data):
        """Test prediction output shape."""
        result = model.predict(sample_data)
        assert result.shape == sample_data["tv"].shape
    
    def test_prediction_bounds(self, model, sample_data):
        """Test prediction value bounds."""
        result = model.predict(sample_data)
        assert result.min() >= 0  # Revenue non-negative
        assert result.max() <= 1e9  # Reasonable upper bound
    
    def test_zero_budget(self, model, sample_data):
        """Test zero budget scenario."""
        # Set all budgets to zero
        zero_data = sample_data.copy()
        for var in zero_data.data_vars:
            zero_data[var] = 0
        
        result = model.predict(zero_data)
        assert result.sum() == 0  # No spend = no revenue
    
    @pytest.mark.parametrize("channel", ["tv", "digital"])
    def test_channel_contribution(self, model, sample_data, channel):
        """Test individual channel contributions."""
        # Isolate single channel
        single_channel = sample_data.copy()
        for var in single_channel.data_vars:
            if var != channel:
                single_channel[var] = 0
        
        result = model.predict(single_channel)
        assert result.sum() > 0  # Channel has positive impact
```

## Model Lifecycle Management

### Version Control

```python
class VersionedModel(AbstractModel):
    """Model with version tracking."""
    
    def __init__(self, model_path: str, version: str):
        self.version = version
        self.model_path = model_path
        self.created_at = datetime.now()
        self.metadata = self._load_metadata()
    
    def get_version_info(self) -> Dict:
        """Get model version information."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_path": self.model_path,
            "performance_metrics": self.metadata.get("metrics", {}),
            "training_data": self.metadata.get("training_info", {})
        }
```

### Model Registry Integration

```python
# Register model with framework
from optimizer_framework import ModelRegistry

registry = ModelRegistry()

# Register model
model_id = registry.register(
    model=MyRevenueModel("model.pkl"),
    version="2.0.0",
    tags=["production", "revenue", "mmm"],
    metadata={
        "training_date": "2024-01-15",
        "performance": {"rmse": 0.05, "mape": 0.03}
    }
)

# Use registered model
model = registry.get_model(model_id)
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Dimension Mismatch**
   - Ensure input data has all required dimensions
   - Check coordinate alignment
   - Verify dimension ordering

2. **Performance Issues**
   - Implement caching for expensive operations
   - Use batch processing for multiple scenarios
   - Consider model simplification

3. **Integration Failures**
   - Verify API endpoints and authentication
   - Check network connectivity
   - Review error logs

4. **Validation Errors**
   - Double-check input data ranges
   - Verify data types
   - Ensure all required variables present

## Next Steps

After integrating your model:

1. Write comprehensive tests
2. Document model assumptions and limitations
3. Set up monitoring and alerting
4. Create usage examples
5. Submit for code review

For more information, see:
- [Optimization Strategy Guide](optimization_strategies.md)
- [Testing Best Practices](../contributing.md#testing-guidelines)
- [API Reference](../api/index.md#models)