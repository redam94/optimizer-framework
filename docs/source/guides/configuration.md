# Configuration Guide

This guide covers all aspects of configuring the Optimizer Framework, from basic settings to advanced customization options.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Configuration File Formats](#configuration-file-formats)
3. [Environment Variables](#environment-variables)
4. [Model Configuration](#model-configuration)
5. [Optimizer Configuration](#optimizer-configuration)
6. [Constraint Configuration](#constraint-configuration)
7. [Logging Configuration](#logging-configuration)
8. [API Configuration](#api-configuration)
9. [Performance Tuning](#performance-tuning)
10. [Best Practices](#best-practices)

## Configuration Overview

The Optimizer Framework uses a hierarchical configuration system:

```
1. Default Configuration (built-in)
2. Configuration Files (YAML/JSON/TOML)
3. Environment Variables
4. Command Line Arguments
5. Runtime Configuration
```

Each level overrides the previous, allowing flexible configuration management.

### Configuration Loading Order

```python
from optimizer_framework.config import ConfigurationManager

# Load configuration with precedence
config = ConfigurationManager.load_config(
    default_config="config/defaults.yaml",
    user_config="config/production.yaml",
    env_prefix="OPTIMIZER_",
    cli_args=sys.argv[1:]
)
```

## Configuration File Formats

### YAML Configuration (Recommended)

```yaml
# config/optimizer.yaml
version: "1.0"

# Application settings
app:
  name: "Optimizer Framework"
  environment: "production"
  debug: false
  log_level: "INFO"

# Model configuration
models:
  default_timeout: 300
  cache_enabled: true
  cache_ttl: 3600
  registry:
    type: "local"
    path: "/app/models"
  
  # Model-specific settings
  revenue_model:
    type: "docker"
    image: "revenue-model:latest"
    port: 8001
    health_check_interval: 30
    timeout: 600
    
  awareness_model:
    type: "python"
    module: "models.awareness"
    class: "AwarenessModel"
    params:
      saturation_alpha: 2.5
      carryover_rate: 0.3

# Optimizer configuration
optimizer:
  default_type: "optuna"
  max_iterations: 1000
  convergence_tolerance: 1e-6
  
  scipy:
    method: "SLSQP"
    options:
      ftol: 1e-6
      maxiter: 1000
      disp: false
  
  optuna:
    n_trials: 2000
    n_jobs: -1
    sampler:
      type: "TPE"
      n_startup_trials: 10
      n_ei_candidates: 24
    pruner:
      type: "MedianPruner"
      n_startup_trials: 5

# Database configuration
database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 0
  pool_pre_ping: true
  echo: false

# Cache configuration
cache:
  type: "redis"
  redis:
    host: "${REDIS_HOST:localhost}"
    port: ${REDIS_PORT:6379}
    db: 0
    password: "${REDIS_PASSWORD:}"
    socket_timeout: 5
    decode_responses: true
  
  memory:
    max_size: 1000
    ttl: 3600

# API configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: ${API_WORKERS:4}
  cors:
    enabled: true
    origins:
      - "https://app.example.com"
      - "https://localhost:3000"
    allow_credentials: true
    allow_methods: ["GET", "POST", "PUT", "DELETE"]
    allow_headers: ["*"]
  
  rate_limiting:
    enabled: true
    default_limit: "100/minute"
    endpoints:
      "/api/optimize": "10/minute"
      "/api/models": "1000/hour"
  
  authentication:
    enabled: true
    type: "jwt"
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    token_expiration: 3600

# Monitoring configuration
monitoring:
  metrics:
    enabled: true
    endpoint: "/metrics"
    include_histogram: true
  
  tracing:
    enabled: false
    jaeger:
      agent_host: "localhost"
      agent_port: 6831
      service_name: "optimizer-framework"
  
  health_check:
    endpoint: "/health"
    detailed_endpoint: "/health/detailed"
    checks:
      - database
      - cache
      - models

# Feature flags
features:
  async_optimization: true
  model_versioning: true
  auto_scaling: false
  experimental:
    neural_optimizer: false
    quantum_annealing: false
```

### JSON Configuration

```json
{
  "version": "1.0",
  "app": {
    "name": "Optimizer Framework",
    "environment": "production",
    "debug": false,
    "log_level": "INFO"
  },
  "models": {
    "default_timeout": 300,
    "cache_enabled": true,
    "registry": {
      "type": "local",
      "path": "/app/models"
    }
  },
  "optimizer": {
    "default_type": "scipy",
    "scipy": {
      "method": "SLSQP",
      "options": {
        "ftol": 1e-6,
        "maxiter": 1000
      }
    }
  }
}
```

### TOML Configuration

```toml
# config/optimizer.toml
version = "1.0"

[app]
name = "Optimizer Framework"
environment = "production"
debug = false
log_level = "INFO"

[models]
default_timeout = 300
cache_enabled = true

[models.registry]
type = "local"
path = "/app/models"

[optimizer]
default_type = "scipy"

[optimizer.scipy]
method = "SLSQP"

[optimizer.scipy.options]
ftol = 1e-6
maxiter = 1000
```

## Environment Variables

### Standard Environment Variables

```bash
# Application
export OPTIMIZER_ENV=production
export OPTIMIZER_DEBUG=false
export OPTIMIZER_LOG_LEVEL=INFO

# Database
export DATABASE_URL=postgresql://user:pass@localhost:5432/optimizer
export DATABASE_POOL_SIZE=20

# Redis Cache
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=secret

# API
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Security
export JWT_SECRET_KEY=your-secret-key
export CORS_ORIGINS=https://app.example.com,https://localhost:3000

# AWS (if using)
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_REGION=us-east-1

# Feature Flags
export FEATURE_ASYNC_OPTIMIZATION=true
export FEATURE_MODEL_VERSIONING=true
```

### Loading Environment Variables

```python
from optimizer_framework.config import EnvConfig

# Load with prefix
env_config = EnvConfig(prefix="OPTIMIZER_")

# Access values with type conversion
debug = env_config.get_bool("DEBUG", default=False)
port = env_config.get_int("API_PORT", default=8000)
origins = env_config.get_list("CORS_ORIGINS", default=["*"])

# Load from .env file
from dotenv import load_dotenv
load_dotenv()
```

## Model Configuration

### Model Registry Configuration

```yaml
# models/registry.yaml
models:
  - name: "revenue_model"
    version: "2.0.0"
    type: "sklearn"
    path: "models/revenue_model_v2.pkl"
    features:
      - tv_spend
      - digital_spend
      - radio_spend
    metadata:
      trained_date: "2024-01-15"
      performance:
        rmse: 0.05
        mape: 0.03
    
  - name: "awareness_model"
    version: "1.5.0"
    type: "tensorflow"
    path: "models/awareness_model"
    serving_config:
      batch_size: 32
      input_shape: [None, 10]
    
  - name: "attribution_model"
    version: "3.1.0"
    type: "api"
    endpoint: "https://api.attribution.com/v1/predict"
    auth:
      type: "bearer"
      token: "${ATTRIBUTION_API_TOKEN}"
    timeout: 30
    retry:
      max_attempts: 3
      backoff: "exponential"
```

### Model-Specific Configuration

```python
# config/models/revenue_model.py
MODEL_CONFIG = {
    "name": "Revenue Model",
    "version": "2.0.0",
    "type": "regression",
    
    # Model parameters
    "parameters": {
        "saturation": {
            "tv": {"alpha": 2.5, "gamma": 0.8},
            "digital": {"alpha": 2.0, "gamma": 0.9},
            "radio": {"alpha": 2.3, "gamma": 0.7}
        },
        "carryover": {
            "tv": 0.3,
            "digital": 0.1,
            "radio": 0.2
        },
        "base_sales": 500000,
        "trend": 0.02,
        "seasonality": {
            "amplitude": 50000,
            "frequency": 52  # Weekly
        }
    },
    
    # Data requirements
    "input_schema": {
        "required_features": ["tv", "digital", "radio"],
        "optional_features": ["competitor_spend", "macro_index"],
        "time_granularity": "weekly",
        "min_history": 52  # weeks
    },
    
    # Performance settings
    "performance": {
        "cache_predictions": True,
        "batch_size": 1000,
        "use_gpu": False,
        "parallel_workers": 4
    }
}
```

## Optimizer Configuration

### Optimizer Profiles

```yaml
# config/optimizer_profiles.yaml
profiles:
  fast:
    type: "scipy"
    method: "L-BFGS-B"
    options:
      ftol: 1e-4
      maxiter: 100
      maxfun: 150
    
  accurate:
    type: "scipy"
    method: "trust-constr"
    options:
      xtol: 1e-8
      gtol: 1e-8
      maxiter: 1000
      initial_trust_radius: 1.0
    
  exploration:
    type: "optuna"
    n_trials: 5000
    sampler:
      type: "TPE"
      multivariate: true
      constant_liar: true
    pruner:
      type: "HyperbandPruner"
      min_resource: 10
      reduction_factor: 3
    
  production:
    type: "hybrid"
    stages:
      - profile: "exploration"
        n_trials: 100
      - profile: "accurate"
        warm_start: true
```

### Dynamic Optimizer Selection

```python
# config/optimizer_selector.py
def select_optimizer_config(problem_characteristics):
    """Select optimizer configuration based on problem."""
    
    n_variables = problem_characteristics['n_variables']
    constraint_type = problem_characteristics['constraint_type']
    objective_smoothness = problem_characteristics['smoothness']
    
    if n_variables < 10 and objective_smoothness == 'smooth':
        return load_profile('fast')
    elif constraint_type == 'nonlinear':
        return load_profile('accurate')
    elif n_variables > 100:
        return load_profile('exploration')
    else:
        return load_profile('production')
```

## Constraint Configuration

### Constraint Templates

```yaml
# config/constraints.yaml
constraint_templates:
  standard_marketing:
    budget_constraints:
      total:
        min: 100000
        max: 10000000
      
      channel_bounds:
        tv:
          min: 0
          max: 5000000
        digital:
          min: 10000
          max: 8000000
        radio:
          min: 0
          max: 1000000
    
    business_rules:
      - name: "digital_minimum"
        type: "percentage"
        channel: "digital"
        min_percentage: 0.3
        
      - name: "traditional_maximum"
        type: "percentage"
        channels: ["tv", "radio", "print"]
        max_percentage: 0.5
        
      - name: "channel_ratio"
        type: "ratio"
        numerator: "tv"
        denominator: "digital"
        min_ratio: 0.5
        max_ratio: 2.0
    
    custom_constraints:
      - name: "market_share"
        function: "constraints.market_share_limit"
        params:
          max_share: 0.3
          market_size: 10000000
```

### Dynamic Constraints

```python
# config/dynamic_constraints.py
from datetime import datetime

def get_seasonal_constraints(base_constraints, current_date):
    """Adjust constraints based on seasonality."""
    constraints = base_constraints.copy()
    
    # Q4 holiday season
    if current_date.month in [10, 11, 12]:
        # Increase budgets
        constraints['budget_constraints']['total']['max'] *= 1.5
        constraints['business_rules'].append({
            'name': 'holiday_tv_minimum',
            'type': 'percentage',
            'channel': 'tv',
            'min_percentage': 0.25
        })
    
    # Summer season
    elif current_date.month in [6, 7, 8]:
        # Focus on digital
        constraints['business_rules'].append({
            'name': 'summer_digital_focus',
            'type': 'percentage',
            'channel': 'digital',
            'min_percentage': 0.5
        })
    
    return constraints
```

## Logging Configuration

### Structured Logging

```yaml
# config/logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'
  
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/optimizer.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: json
    filename: logs/errors.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  optimizer_framework:
    level: INFO
    handlers: [console, file]
    propagate: false
  
  optimizer_framework.models:
    level: DEBUG
    handlers: [file]
  
  optimizer_framework.api:
    level: INFO
    handlers: [console, file]

root:
  level: INFO
  handlers: [console]
```

### Log Aggregation

```python
# config/log_aggregation.py
import logging
from logging.handlers import SysLogHandler

def setup_centralized_logging(app_config):
    """Configure centralized logging."""
    
    if app_config.get('logging.centralized.enabled'):
        # Syslog handler
        syslog = SysLogHandler(
            address=(
                app_config['logging.centralized.host'],
                app_config['logging.centralized.port']
            )
        )
        syslog.setFormatter(
            logging.Formatter('optimizer-framework: %(message)s')
        )
        
        # Add to root logger
        logging.getLogger().addHandler(syslog)
        
    # Elasticsearch handler
    if app_config.get('logging.elasticsearch.enabled'):
        from elasticsearch import Elasticsearch
        from pythonjsonlogger import jsonlogger
        
        es_handler = ElasticsearchHandler(
            hosts=[app_config['logging.elasticsearch.host']],
            index_name='optimizer-logs'
        )
        es_handler.setFormatter(jsonlogger.JsonFormatter())
        
        logging.getLogger('optimizer_framework').addHandler(es_handler)
```

## API Configuration

### Rate Limiting Configuration

```python
# config/rate_limiting.py
from slowapi import Limiter
from slowapi.util import get_remote_address

RATE_LIMIT_CONFIG = {
    "default": "100/minute",
    "endpoints": {
        "/api/optimize": {
            "limit": "10/minute",
            "key_func": get_remote_address,
            "error_message": "Optimization rate limit exceeded"
        },
        "/api/models/predict": {
            "limit": "1000/hour",
            "key_func": lambda req: f"{get_remote_address(req)}:{req.headers.get('X-API-Key')}"
        },
        "/api/admin/*": {
            "limit": "30/minute",
            "key_func": lambda req: req.headers.get('X-API-Key', 'anonymous')
        }
    },
    "storage": {
        "type": "redis",
        "redis_url": "redis://localhost:6379/1"
    }
}
```

### CORS Configuration

```python
# config/cors.py
from fastapi.middleware.cors import CORSMiddleware

CORS_CONFIG = {
    "allow_origins": [
        "https://app.example.com",
        "https://staging.example.com",
        "http://localhost:3000"  # Development
    ],
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": [
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID"
    ],
    "expose_headers": [
        "X-Rate-Limit-Limit",
        "X-Rate-Limit-Remaining",
        "X-Rate-Limit-Reset"
    ],
    "max_age": 86400  # 24 hours
}

def configure_cors(app):
    """Configure CORS for the application."""
    if app.config.get('api.cors.enabled'):
        app.add_middleware(
            CORSMiddleware,
            **CORS_CONFIG
        )
```

## Performance Tuning

### Cache Configuration

```yaml
# config/cache.yaml
cache:
  # Model prediction cache
  predictions:
    enabled: true
    backend: "redis"
    ttl: 3600  # 1 hour
    max_size: 10000
    eviction_policy: "lru"
    key_prefix: "pred:"
    
  # Optimization results cache
  optimizations:
    enabled: true
    backend: "redis"
    ttl: 86400  # 24 hours
    max_size: 1000
    key_prefix: "opt:"
    
  # API response cache
  api_responses:
    enabled: true
    backend: "memory"
    ttl: 300  # 5 minutes
    max_size: 5000
    
  # Configuration
  redis:
    connection_pool:
      max_connections: 50
      max_idle_time: 300
      retry_on_timeout: true
```

### Worker Configuration

```yaml
# config/workers.yaml
workers:
  # API workers
  api:
    count: ${WORKERS:4}
    worker_class: "uvicorn.workers.UvicornWorker"
    timeout: 300
    graceful_timeout: 30
    max_requests: 1000
    max_requests_jitter: 50
    
  # Background workers
  background:
    count: 2
    queues:
      - optimization
      - model_training
      - reporting
    concurrency: 10
    
  # Optimization workers
  optimization:
    pool_size: ${CPU_COUNT:4}
    process_based: true
    memory_limit: "2GB"
    timeout: 3600
```

## Best Practices

### Configuration Management

1. **Use Environment Variables for Secrets**
   ```python
   # Good
   database_url = os.environ['DATABASE_URL']
   
   # Bad
   database_url = "postgresql://user:password@localhost/db"
   ```

2. **Validate Configuration on Startup**
   ```python
   from pydantic import BaseSettings, validator
   
   class AppConfig(BaseSettings):
       database_url: str
       redis_url: str
       jwt_secret: str
       
       @validator('database_url')
       def validate_db_url(cls, v):
           if not v.startswith(('postgresql://', 'mysql://')):
               raise ValueError('Invalid database URL')
           return v
       
       class Config:
           env_prefix = 'OPTIMIZER_'
   ```

3. **Use Configuration Profiles**
   ```python
   # config/profiles.py
   PROFILES = {
       'development': {
           'debug': True,
           'log_level': 'DEBUG',
           'cache_enabled': False
       },
       'staging': {
           'debug': False,
           'log_level': 'INFO',
           'cache_enabled': True
       },
       'production': {
           'debug': False,
           'log_level': 'WARNING',
           'cache_enabled': True
       }
   }
   ```

4. **Implement Configuration Hot-Reload**
   ```python
   from watchdog.observers import Observer
   from watchdog.events import FileSystemEventHandler
   
   class ConfigReloader(FileSystemEventHandler):
       def on_modified(self, event):
           if event.src_path.endswith('.yaml'):
               logger.info(f"Reloading configuration: {event.src_path}")
               ConfigurationManager.reload()
   ```

5. **Document All Configuration Options**
   ```yaml
   # Each configuration option should have:
   setting_name:
     value: "default_value"
     description: "What this setting controls"
     type: "string"
     required: false
     env_var: "OPTIMIZER_SETTING_NAME"
     example: "example_value"
   ```

### Security Considerations

1. **Never commit secrets to version control**
2. **Use secret management services in production**
3. **Rotate secrets regularly**
4. **Implement least privilege access**
5. **Audit configuration changes**

### Configuration Testing

```python
# tests/test_config.py
import pytest
from optimizer_framework.config import ConfigurationManager

def test_load_config():
    """Test configuration loading."""
    config = ConfigurationManager.load_config('tests/fixtures/test_config.yaml')
    assert config['app']['name'] == 'Test Optimizer'
    assert config['database']['pool_size'] == 10

def test_env_override():
    """Test environment variable override."""
    os.environ['OPTIMIZER_DATABASE_POOL_SIZE'] = '20'
    config = ConfigurationManager.load_config()
    assert config['database']['pool_size'] == 20

def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        ConfigurationManager.load_config('tests/fixtures/invalid_config.yaml')
```

Remember: Good configuration management is key to maintaining a flexible, secure, and maintainable system!