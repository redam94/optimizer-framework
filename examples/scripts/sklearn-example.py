"""
Example: Using Scikit-learn Models with Atlas Optimization Framework

This example demonstrates how to:
1. Load/create sklearn models
2. Wrap them for Atlas compatibility
3. Run optimization
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Import Atlas components (adjust imports based on your setup)
#from atlas import OptimizerFactory
from atlas.models.wrappers.sklearn_wrapper import SklearnModelWrapper
from atlas.factories.sklearn_factory import SklearnModelFactory, ModelConfigBuilder


# ============================================================================
# Example 1: Basic Usage - Loading an Existing Model
# ============================================================================

def example_load_existing_model():
    """Load a pre-trained sklearn model and optimize with Atlas."""
    
    print("Example 1: Loading Existing Model")
    print("-" * 50)
    
    # Create wrapper for existing model
    model_wrapper = SklearnModelWrapper(
        model_path='models/revenue_model.pkl',
        feature_names=['tv_spend', 'digital_spend', 'radio_spend'],
        target_name='revenue',
        scaler_path='models/scaler.pkl'  # Optional
    )
    
    # Create optimizer
    # optimizer = OptimizerFactory.create(
    #     optimizer_type='scipy',
    #     model=model_wrapper
    # )
    
    # Define budget constraints
    constraints = {
        'total_budget': 1_000_000,
        'bounds': {
            'tv_spend': (100_000, 500_000),
            'digital_spend': (200_000, 600_000),
            'radio_spend': (50_000, 300_000)
        }
    }
    
    # Create input data for optimization
    # Atlas expects xarray Dataset
    initial_budget = xr.Dataset({
        'tv_spend': xr.DataArray([300_000]),
        'digital_spend': xr.DataArray([400_000]),
        'radio_spend': xr.DataArray([300_000])
    })
    
    # # Run optimization
    # result = optimizer.optimize(initial_budget, constraints)
    
    # print(f"Optimal Budget Allocation:")
    # print(f"  TV: ${result.optimal_budget['tv_spend']:,.0f}")
    # print(f"  Digital: ${result.optimal_budget['digital_spend']:,.0f}")
    # print(f"  Radio: ${result.optimal_budget['radio_spend']:,.0f}")
    # print(f"Expected Revenue: ${result.optimal_value:,.0f}")
    

# ============================================================================
# Example 2: Training and Using a New Model
# ============================================================================

def example_train_new_model():
    """Train a new sklearn model and use it with Atlas."""
    
    print("\nExample 2: Training New Model")
    print("-" * 50)
    
    # Generate synthetic marketing data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: marketing spend by channel
    tv_spend = np.random.uniform(10_000, 100_000, n_samples)
    digital_spend = np.random.uniform(20_000, 150_000, n_samples)
    radio_spend = np.random.uniform(5_000, 50_000, n_samples)
    
    # Target: revenue with some non-linear relationships
    revenue = (
        0.8 * np.sqrt(tv_spend) * 100 +
        1.2 * np.log1p(digital_spend) * 1000 +
        0.5 * radio_spend +
        np.random.normal(0, 10_000, n_samples)
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'tv_spend': tv_spend,
        'digital_spend': digital_spend,
        'radio_spend': radio_spend,
        'revenue': revenue
    })
    
    # Prepare features and target
    feature_names = ['tv_spend', 'digital_spend', 'radio_spend']
    X = data[feature_names]
    y = data['revenue']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Model R² - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    # Save model and scaler
    joblib.dump(model, 'revenue_model.pkl')
    joblib.dump(scaler, 'revenue_scaler.pkl')
    
    # Create wrapper using factory
    model_wrapper = SklearnModelFactory.create(
        model_instance=model,
        feature_names=feature_names,
        target_name='revenue',
        scaler=scaler
    )
    
    # # Optimize
    # optimizer = OptimizerFactory.create('scipy', model=model_wrapper)
    
    # Create optimization input
    current_budget = xr.Dataset({
        'tv_spend': xr.DataArray([50_000]),
        'digital_spend': xr.DataArray([80_000]),
        'radio_spend': xr.DataArray([20_000])
    })
    
    # constraints = {
    #     'total_budget': 150_000,
    #     'bounds': {
    #         'tv_spend': (10_000, 100_000),
    #         'digital_spend': (20_000, 100_000),
    #         'radio_spend': (5_000, 50_000)
    #     }
    # }
    
    # result = optimizer.optimize(current_budget, constraints)
    
    print(f"\nOptimization Results:")
    print(f"Current Revenue: ${model_wrapper.predict(current_budget).item():,.0f}")
    # print(f"Optimized Revenue: ${result.optimal_value:,.0f}")
    # print(f"Improvement: {result.improvement_percentage:.1f}%")
    

# ============================================================================
# Example 3: Using Model Factory with Configuration
# ============================================================================

def example_factory_with_config():
    """Use model factory with configuration builder."""
    
    print("\nExample 3: Factory with Configuration")
    print("-" * 50)
    
    # Build configuration
    config = (ModelConfigBuilder()
        .model_type('random_forest')
        .features(['tv', 'digital', 'radio', 'social'])
        .target('conversions')
        .model_params(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5
        )
        .dimensions(time_dim='week', channel_dim='channel')
        .contribution_method('feature_importance')
        
    )
    
    # Save configuration
    config.save('model_config.yaml')
    
    # Create model from config
    # (This would create a new untrained model - typically you'd load a trained one)
    model_wrapper = SklearnModelFactory.from_config('model_config.yaml')
    
    print("Model configuration created and saved")
    

# ============================================================================
# Example 4: Multi-Channel Optimization with Time Series
# ============================================================================

def example_time_series_optimization():
    """Optimize budget allocation across time periods."""
    
    print("\nExample 4: Time Series Optimization")
    print("-" * 50)
    
    # Load model
    model_wrapper = SklearnModelWrapper(
        model_path='models/time_series_model.pkl',
        feature_names=['tv', 'digital', 'radio'],
        time_dim='week',
        channel_dim='channel'
    )
    
    # Create time series budget data
    weeks = pd.date_range('2024-01-01', periods=4, freq='W')
    channels = ['tv', 'digital', 'radio']
    
    # Current budget allocation
    current_budget = xr.Dataset({
        channel: xr.DataArray(
            np.random.uniform(20_000, 50_000, len(weeks)),
            dims=['week'],
            coords={'week': weeks}
        ) for channel in channels
    })
    
    # Optimize for each week
    optimizer = OptimizerFactory.create('scipy', model=model_wrapper)
    
    weekly_results = []
    for week in weeks:
        week_data = current_budget.sel(week=week)
        
        constraints = {
            'total_budget': 120_000,
            'bounds': {
                'tv': (10_000, 60_000),
                'digital': (20_000, 80_000),
                'radio': (5_000, 40_000)
            }
        }
        
        result = optimizer.optimize(week_data, constraints)
        weekly_results.append({
            'week': week,
            'optimal_budget': result.optimal_budget,
            'expected_value': result.optimal_value
        })
    
    # Display results
    for res in weekly_results:
        print(f"\nWeek {res['week'].strftime('%Y-%m-%d')}:")
        for channel, amount in res['optimal_budget'].items():
            print(f"  {channel}: ${amount:,.0f}")
        print(f"  Expected outcome: ${res['expected_value']:,.0f}")


# ============================================================================
# Example 5: Model Comparison
# ============================================================================

def example_model_comparison():
    """Compare different sklearn models for optimization."""
    
    print("\nExample 5: Model Comparison")
    print("-" * 50)
    
    # Test data
    test_budget = xr.Dataset({
        'feature1': xr.DataArray([100_000]),
        'feature2': xr.DataArray([150_000]),
        'feature3': xr.DataArray([80_000])
    })
    
    constraints = {
        'total_budget': 330_000,
        'bounds': {
            'feature1': (50_000, 150_000),
            'feature2': (100_000, 200_000),
            'feature3': (50_000, 100_000)
        }
    }
    
    # Compare different model types
    model_types = ['linear', 'ridge', 'random_forest']
    results = []
    
    for model_type in model_types:
        try:
            # Create model wrapper
            # (In practice, these would be pre-trained models)
            model_wrapper = SklearnModelFactory.create(
                model_type=model_type,
                feature_names=['feature1', 'feature2', 'feature3'],
                model_params={'random_state': 42} if model_type == 'random_forest' else {}
            )
            
            # Create optimizer
            optimizer = OptimizerFactory.create('scipy', model=model_wrapper)
            
            # Optimize
            result = optimizer.optimize(test_budget, constraints)
            
            results.append({
                'model': model_type,
                'optimal_value': result.optimal_value,
                'budget': result.optimal_budget
            })
            
        except Exception as e:
            print(f"Error with {model_type}: {str(e)}")
    
    # Display comparison
    print("\nModel Comparison Results:")
    for res in results:
        print(f"\n{res['model'].title()} Model:")
        print(f"  Optimal value: ${res['optimal_value']:,.0f}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run examples
    try:
        example_load_existing_model()
    except FileNotFoundError:
        print("Skipping Example 1: Model files not found")
    
    example_train_new_model()
    example_factory_with_config()
    
    # try:
    #     example_time_series_optimization()
    # except FileNotFoundError:
    #     print("Skipping Example 4: Time series model not found")
    
    # example_model_comparison()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("="*50)