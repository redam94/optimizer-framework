# Marketing Mix Modeling (MMM) Example

This comprehensive example demonstrates how to use Atlas for Marketing Mix Modeling, a common use case for budget optimization across marketing channels.

## Overview

Marketing Mix Modeling helps businesses understand:
- How different marketing channels contribute to sales
- Optimal budget allocation across channels
- Diminishing returns and saturation effects
- Synergies between channels
- Time-lagged effects of marketing spend

## Complete MMM Implementation

### Step 1: Setup and Data Preparation

```python
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from atlas import (
    ModelWrapper, 
    OptimizerFactory,
    ConfigurationManager
)
from atlas.strategies import MediaMixOptimizationStrategy
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

### Step 2: Load and Prepare Marketing Data

```python
# Load historical marketing data
# In practice, this would come from your data warehouse
def generate_sample_mmm_data():
    """Generate realistic marketing mix data for demonstration."""
    np.random.seed(42)
    
    # Time periods (weekly data for 2 years)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='W')
    n_periods = len(dates)
    
    # Marketing channels
    channels = ['tv', 'digital_search', 'digital_display', 'social', 'radio', 'print']
    
    # Generate spend data with realistic patterns
    data = pd.DataFrame({'date': dates})
    
    # TV: High spend with seasonality
    data['tv_spend'] = (
        100000 + 50000 * np.sin(np.arange(n_periods) * 2 * np.pi / 52) +
        np.random.normal(0, 10000, n_periods)
    ).clip(min=0)
    
    # Digital Search: Growing trend
    data['digital_search_spend'] = (
        50000 + 1000 * np.arange(n_periods) +
        np.random.normal(0, 5000, n_periods)
    ).clip(min=0)
    
    # Digital Display: Stable with campaigns
    data['digital_display_spend'] = (
        30000 + 20000 * (np.random.random(n_periods) > 0.7) +
        np.random.normal(0, 3000, n_periods)
    ).clip(min=0)
    
    # Social: Growing with high volatility
    data['social_spend'] = (
        20000 + 500 * np.arange(n_periods) +
        np.random.normal(0, 8000, n_periods)
    ).clip(min=0)
    
    # Radio: Declining trend
    data['radio_spend'] = (
        40000 - 300 * np.arange(n_periods) +
        np.random.normal(0, 3000, n_periods)
    ).clip(min=10000)
    
    # Print: Seasonal (high in Q4)
    data['print_spend'] = (
        15000 + 10000 * ((data['date'].dt.quarter == 4).astype(int)) +
        np.random.normal(0, 2000, n_periods)
    ).clip(min=0)
    
    # Generate impressions (with diminishing returns)
    for channel in channels:
        spend_col = f'{channel}_spend'
        data[f'{channel}_impressions'] = (
            1000 * np.sqrt(data[spend_col]) +
            np.random.normal(0, 100, n_periods)
        ).clip(min=0)
    
    # Generate sales with channel contributions
    base_sales = 500000
    channel_effects = {
        'tv': 0.3,
        'digital_search': 0.5,
        'digital_display': 0.2,
        'social': 0.15,
        'radio': 0.1,
        'print': 0.05
    }
    
    # Calculate sales with carryover effects
    data['sales'] = base_sales
    
    for channel, effect in channel_effects.items():
        spend_col = f'{channel}_spend'
        # Add immediate effect
        data['sales'] += effect * np.sqrt(data[spend_col])
        # Add carryover effect (30% of previous week)
        carryover = 0.3 * effect * np.sqrt(data[spend_col].shift(1).fillna(0))
        data['sales'] += carryover
    
    # Add seasonality and noise
    data['sales'] += 50000 * np.sin(np.arange(n_periods) * 2 * np.pi / 52)
    data['sales'] += np.random.normal(0, 20000, n_periods)
    
    # Add external factors
    data['competitor_spend'] = np.random.uniform(200000, 400000, n_periods)
    data['macro_index'] = 100 + np.random.normal(0, 5, n_periods)
    
    return data

# Load data
marketing_data = generate_sample_mmm_data()
print(marketing_data.head())
print(f"\nData shape: {marketing_data.shape}")
print(f"Date range: {marketing_data['date'].min()} to {marketing_data['date'].max()}")
```

### Step 3: Create Marketing Mix Model

```python
class MarketingMixModel(ModelWrapper):
    """
    Marketing Mix Model with saturation curves and carryover effects.
    """
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 carryover_rates: dict = None,
                 saturation_params: dict = None):
        """
        Initialize MMM model.
        
        Args:
            historical_data: Historical marketing and sales data
            carryover_rates: Adstock/carryover rates by channel
            saturation_params: Saturation curve parameters by channel
        """
        super().__init__()
        self.historical_data = historical_data
        self.channels = ['tv', 'digital_search', 'digital_display', 
                        'social', 'radio', 'print']
        
        # Default carryover rates (percentage of effect carrying to next period)
        self.carryover_rates = carryover_rates or {
            'tv': 0.3,
            'digital_search': 0.1,
            'digital_display': 0.15,
            'social': 0.05,
            'radio': 0.2,
            'print': 0.25
        }
        
        # Default saturation parameters (Hill transformation)
        self.saturation_params = saturation_params or {
            'tv': {'alpha': 2.5, 'gamma': 0.8},
            'digital_search': {'alpha': 2.0, 'gamma': 0.9},
            'digital_display': {'alpha': 2.2, 'gamma': 0.85},
            'social': {'alpha': 1.8, 'gamma': 0.95},
            'radio': {'alpha': 2.3, 'gamma': 0.7},
            'print': {'alpha': 2.0, 'gamma': 0.6}
        }
        
        # Fit response curves
        self._fit_response_curves()
    
    def _fit_response_curves(self):
        """Fit response curves from historical data."""
        self.response_curves = {}
        
        for channel in self.channels:
            spend = self.historical_data[f'{channel}_spend'].values
            # Simplified: use sqrt transformation
            # In practice, you'd fit actual response curves
            self.response_curves[channel] = {
                'coefficient': np.corrcoef(
                    np.sqrt(spend), 
                    self.historical_data['sales']
                )[0, 1] * 1000
            }
    
    def _apply_saturation(self, spend: float, channel: str) -> float:
        """Apply Hill saturation transformation."""
        params = self.saturation_params[channel]
        alpha = params['alpha']
        gamma = params['gamma']
        
        # Hill transformation: spend^alpha / (spend^alpha + gamma^alpha)
        return (spend ** alpha) / (spend ** alpha + gamma ** alpha)
    
    def _apply_carryover(self, spend_series: np.ndarray, channel: str) -> np.ndarray:
        """Apply adstock/carryover transformation."""
        carryover = self.carryover_rates[channel]
        result = np.zeros_like(spend_series)
        
        for t in range(len(spend_series)):
            # Current period effect
            result[t] = spend_series[t]
            # Add carryover from previous periods
            for lag in range(1, min(t + 1, 13)):  # Max 13 weeks carryover
                result[t] += spend_series[t - lag] * (carryover ** lag)
        
        return result
    
    def predict(self, budget_allocation: xr.Dataset) -> xr.DataArray:
        """
        Predict sales given budget allocation.
        
        Args:
            budget_allocation: Dataset with budget by channel and time
            
        Returns:
            Predicted sales
        """
        # Extract dimensions
        time_periods = budget_allocation.coords['time']
        n_periods = len(time_periods)
        
        # Initialize predictions
        predictions = np.zeros(n_periods)
        base_sales = 500000  # Base sales without marketing
        
        # Process each channel
        for channel in self.channels:
            if channel in budget_allocation.data_vars:
                # Get spend series
                spend = budget_allocation[channel].values
                
                # Apply saturation
                saturated_spend = np.array([
                    self._apply_saturation(s, channel) for s in spend
                ])
                
                # Apply carryover
                effective_spend = self._apply_carryover(saturated_spend, channel)
                
                # Calculate contribution
                coef = self.response_curves[channel]['coefficient']
                contribution = coef * effective_spend
                
                predictions += contribution
        
        # Add base sales and seasonality
        week_of_year = pd.to_datetime(time_periods.values).isocalendar().week
        seasonality = 50000 * np.sin(2 * np.pi * week_of_year / 52)
        predictions += base_sales + seasonality
        
        # Return as xarray
        return xr.DataArray(
            predictions,
            coords={'time': time_periods},
            dims=['time'],
            name='predicted_sales'
        )
    
    def get_channel_contributions(self, budget_allocation: xr.Dataset) -> xr.Dataset:
        """Calculate individual channel contributions."""
        contributions = {}
        
        for channel in self.channels:
            if channel in budget_allocation.data_vars:
                # Create single-channel budget
                single_channel = budget_allocation.copy()
                for ch in self.channels:
                    if ch != channel:
                        single_channel[ch] = 0
                
                # Get contribution
                contribution = self.predict(single_channel)
                contributions[channel] = contribution - 500000  # Remove base
        
        return xr.Dataset(contributions)
    
    def calculate_roi(self, budget_allocation: xr.Dataset) -> dict:
        """Calculate ROI by channel."""
        contributions = self.get_channel_contributions(budget_allocation)
        roi = {}
        
        for channel in self.channels:
            if channel in budget_allocation.data_vars:
                total_spend = float(budget_allocation[channel].sum())
                total_contribution = float(contributions[channel].sum())
                
                if total_spend > 0:
                    roi[channel] = total_contribution / total_spend
                else:
                    roi[channel] = 0
        
        return roi
```

### Step 4: Set Up Optimization

```python
# Create model instance
mmm_model = MarketingMixModel(marketing_data)

# Define optimization constraints
def create_mmm_constraints(total_budget: float = 1_000_000):
    """Create realistic MMM optimization constraints."""
    return {
        'total_budget': total_budget,
        'channel_bounds': {
            'tv': (50_000, 400_000),           # TV needs minimum for effectiveness
            'digital_search': (100_000, 500_000), # High-performing channel
            'digital_display': (20_000, 200_000),
            'social': (30_000, 300_000),
            'radio': (10_000, 100_000),         # Declining channel
            'print': (5_000, 50_000)            # Minimal investment
        },
        'business_rules': {
            # Digital should be at least 40% of budget
            'min_digital_share': 0.4,
            # No single channel over 40%
            'max_channel_share': 0.4,
            # Traditional media (TV + Radio + Print) max 50%
            'max_traditional_share': 0.5
        },
        'ratio_constraints': {
            # Search should be at least 50% of display
            ('digital_search', 'digital_display'): (0.5, None),
            # Social shouldn't exceed search
            ('social', 'digital_search'): (None, 1.0)
        }
    }

# Create optimization strategy
optimization_strategy = MediaMixOptimizationStrategy(
    reach_frequency_targets={'reach': 0.75, 'frequency': 3.0},
    seasonality_factors={'Q1': 0.8, 'Q2': 1.0, 'Q3': 0.9, 'Q4': 1.3},
    competitive_response=True
)

# Create optimizer
optimizer = OptimizerFactory.create(
    optimizer_type='optuna',
    model=mmm_model,
    strategy=optimization_strategy,
    config={
        'n_trials': 1000,
        'n_jobs': -1,
        'sampler': 'TPE',
        'pruner': 'MedianPruner'
    }
)
```

### Step 5: Run Optimization for Different Scenarios

```python
# Scenario 1: Current period optimization
current_budget = {
    'tv': 150_000,
    'digital_search': 200_000,
    'digital_display': 80_000,
    'social': 100_000,
    'radio': 50_000,
    'print': 20_000
}

print("Current Budget Allocation:")
for channel, amount in current_budget.items():
    print(f"  {channel}: ${amount:,.0f}")
print(f"  Total: ${sum(current_budget.values()):,.0f}")

# Run optimization
constraints = create_mmm_constraints(total_budget=1_000_000)
optimal_result = optimizer.optimize(
    initial_budget=current_budget,
    constraints=constraints,
    time_periods=['2024-W01']  # Next week
)

print("\nOptimal Budget Allocation:")
for channel, amount in optimal_result.optimal_budget.items():
    print(f"  {channel}: ${amount:,.0f}")
print(f"  Total: ${sum(optimal_result.optimal_budget.values()):,.0f}")
print(f"\nExpected Sales Increase: ${optimal_result.improvement:,.0f}")

# Scenario 2: Multi-period optimization (Quarter)
quarterly_periods = pd.date_range('2024-01-01', '2024-03-31', freq='W')
quarterly_budget = 12_000_000  # $12M for Q1

quarterly_result = optimizer.optimize_multiperiod(
    periods=quarterly_periods,
    total_budget=quarterly_budget,
    constraints=constraints,
    objective='maximize_sales'
)

# Visualize quarterly allocation
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Quarterly Budget Optimization Results', fontsize=16)

# Plot 1: Budget allocation over time
ax1 = axes[0, 0]
allocation_df = quarterly_result.to_dataframe()
allocation_df.plot(kind='area', ax=ax1)
ax1.set_title('Weekly Budget Allocation by Channel')
ax1.set_xlabel('Week')
ax1.set_ylabel('Budget ($)')
ax1.legend(loc='best')

# Plot 2: Channel mix comparison
ax2 = axes[0, 1]
current_mix = pd.Series(current_budget)
optimal_mix = pd.Series(optimal_result.optimal_budget)
comparison_df = pd.DataFrame({
    'Current': current_mix,
    'Optimal': optimal_mix
})
comparison_df.plot(kind='bar', ax=ax2)
ax2.set_title('Current vs Optimal Channel Mix')
ax2.set_ylabel('Budget ($)')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

# Plot 3: ROI by channel
ax3 = axes[1, 0]
current_roi = mmm_model.calculate_roi(
    xr.Dataset({ch: xr.DataArray([amt]) for ch, amt in current_budget.items()})
)
optimal_roi = mmm_model.calculate_roi(
    xr.Dataset({ch: xr.DataArray([amt]) for ch, amt in optimal_result.optimal_budget.items()})
)
roi_comparison = pd.DataFrame({
    'Current ROI': current_roi,
    'Optimal ROI': optimal_roi
})
roi_comparison.plot(kind='bar', ax=ax3)
ax3.set_title('ROI Comparison by Channel')
ax3.set_ylabel('ROI (Sales/Spend)')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

# Plot 4: Saturation curves
ax4 = axes[1, 1]
spend_range = np.linspace(0, 500_000, 100)
for channel in ['tv', 'digital_search', 'social']:
    response = [mmm_model._apply_saturation(s, channel) * 
                mmm_model.response_curves[channel]['coefficient'] 
                for s in spend_range]
    ax4.plot(spend_range, response, label=channel)
ax4.set_title('Channel Response Curves')
ax4.set_xlabel('Spend ($)')
ax4.set_ylabel('Response')
ax4.legend()

plt.tight_layout()
plt.show()
```

### Step 6: Advanced Analysis

```python
# Marginal ROI Analysis
def calculate_marginal_roi(model, current_budget, increment=10_000):
    """Calculate marginal ROI for each channel."""
    marginal_roi = {}
    
    for channel in model.channels:
        # Current performance
        current_pred = model.predict(
            xr.Dataset({ch: xr.DataArray([current_budget.get(ch, 0)]) 
                       for ch in model.channels})
        )
        
        # Performance with increment
        incremented_budget = current_budget.copy()
        incremented_budget[channel] = incremented_budget.get(channel, 0) + increment
        
        incremented_pred = model.predict(
            xr.Dataset({ch: xr.DataArray([incremented_budget.get(ch, 0)]) 
                       for ch in model.channels})
        )
        
        # Marginal ROI
        marginal_sales = float(incremented_pred - current_pred)
        marginal_roi[channel] = marginal_sales / increment
    
    return marginal_roi

marginal_roi = calculate_marginal_roi(mmm_model, current_budget)
print("\nMarginal ROI Analysis (per $10k increase):")
for channel, roi in sorted(marginal_roi.items(), key=lambda x: x[1], reverse=True):
    print(f"  {channel}: ${roi:.2f} per dollar")

# Budget Optimization with Different Objectives
objectives = {
    'maximize_sales': {'weight': 1.0, 'target': 'sales'},
    'maximize_efficiency': {'weight': 1.0, 'target': 'roi'},
    'balanced': {'weight': 0.7, 'target': 'sales', 
                 'weight_roi': 0.3, 'target_roi': 'efficiency'}
}

results_by_objective = {}
for obj_name, obj_config in objectives.items():
    result = optimizer.optimize(
        initial_budget=current_budget,
        constraints=constraints,
        objective=obj_config
    )
    results_by_objective[obj_name] = result

# Compare results
print("\nOptimization Results by Objective:")
comparison_data = []
for obj_name, result in results_by_objective.items():
    total_spend = sum(result.optimal_budget.values())
    expected_sales = result.optimal_value
    overall_roi = expected_sales / total_spend if total_spend > 0 else 0
    
    comparison_data.append({
        'Objective': obj_name,
        'Total Spend': total_spend,
        'Expected Sales': expected_sales,
        'Overall ROI': overall_roi
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Scenario Analysis
scenarios = {
    'recession': {
        'total_budget': 700_000,
        'constraints': {
            'max_traditional_share': 0.3,  # Focus on measurable channels
            'min_digital_share': 0.6
        }
    },
    'growth': {
        'total_budget': 1_500_000,
        'constraints': {
            'min_awareness_channels': 0.4,  # TV + Radio for awareness
            'allow_experimental': True
        }
    },
    'competitive_response': {
        'total_budget': 1_000_000,
        'constraints': {
            'min_search_share': 0.3,  # Defend search position
            'max_channel_volatility': 0.2  # Limit big changes
        }
    }
}

scenario_results = {}
for scenario_name, scenario_config in scenarios.items():
    constraints = create_mmm_constraints(
        total_budget=scenario_config['total_budget']
    )
    constraints.update(scenario_config.get('constraints', {}))
    
    result = optimizer.optimize(
        initial_budget=current_budget,
        constraints=constraints
    )
    scenario_results[scenario_name] = result

# Visualize scenario comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Budget allocation by scenario
scenario_budgets = pd.DataFrame({
    scenario: result.optimal_budget 
    for scenario, result in scenario_results.items()
})
scenario_budgets.plot(kind='bar', ax=ax1)
ax1.set_title('Budget Allocation by Scenario')
ax1.set_xlabel('Channel')
ax1.set_ylabel('Budget ($)')
ax1.legend(title='Scenario')

# Expected outcomes by scenario
outcomes = pd.DataFrame({
    'Expected Sales': {
        scenario: result.optimal_value 
        for scenario, result in scenario_results.items()
    },
    'Total Budget': {
        scenario: sum(result.optimal_budget.values())
        for scenario, result in scenario_results.items()
    }
})
outcomes.plot(kind='bar', ax=ax2)
ax2.set_title('Expected Outcomes by Scenario')
ax2.set_xlabel('Scenario')
ax2.set_ylabel('Value ($)')

plt.tight_layout()
plt.show()
```

### Step 7: Generate Executive Report

```python
class MMM_ReportGenerator:
    """Generate executive reports for MMM optimization results."""
    
    def __init__(self, model, optimization_results, current_budget):
        self.model = model
        self.results = optimization_results
        self.current_budget = current_budget
    
    def generate_executive_summary(self):
        """Generate executive summary of optimization results."""
        optimal_budget = self.results.optimal_budget
        
        # Calculate key metrics
        current_total = sum(self.current_budget.values())
        optimal_total = sum(optimal_budget.values())
        
        # Predict performance
        current_sales = float(self.model.predict(
            xr.Dataset({ch: xr.DataArray([self.current_budget.get(ch, 0)]) 
                       for ch in self.model.channels})
        ))
        optimal_sales = float(self.model.predict(
            xr.Dataset({ch: xr.DataArray([optimal_budget.get(ch, 0)]) 
                       for ch in self.model.channels})
        ))
        
        sales_increase = optimal_sales - current_sales
        sales_increase_pct = (sales_increase / current_sales) * 100
        
        # ROI comparison
        current_roi = current_sales / current_total if current_total > 0 else 0
        optimal_roi = optimal_sales / optimal_total if optimal_total > 0 else 0
        roi_improvement = optimal_roi - current_roi
        
        # Channel shifts
        channel_changes = {}
        for channel in self.model.channels:
            current = self.current_budget.get(channel, 0)
            optimal = optimal_budget.get(channel, 0)
            change_amt = optimal - current
            change_pct = (change_amt / current * 100) if current > 0 else 0
            channel_changes[channel] = {
                'current': current,
                'optimal': optimal,
                'change_amt': change_amt,
                'change_pct': change_pct
            }
        
        # Generate report
        report = f"""
# Marketing Mix Optimization - Executive Summary

## Key Findings

**Expected Sales Improvement**: ${sales_increase:,.0f} (+{sales_increase_pct:.1f}%)
**ROI Improvement**: {roi_improvement:.2f} ({current_roi:.2f} → {optimal_roi:.2f})
**Budget Maintained**: ${optimal_total:,.0f}

## Recommended Actions

### Immediate Changes (High Impact)
"""
        
        # Sort by absolute change
        sorted_changes = sorted(
            channel_changes.items(), 
            key=lambda x: abs(x[1]['change_amt']), 
            reverse=True
        )
        
        for channel, change in sorted_changes[:3]:
            if change['change_amt'] > 0:
                action = "INCREASE"
                emoji = "📈"
            else:
                action = "DECREASE"
                emoji = "📉"
            
            report += f"""
{emoji} **{action} {channel.replace('_', ' ').title()}**
   - Current: ${change['current']:,.0f}
   - Recommended: ${change['optimal']:,.0f}
   - Change: ${change['change_amt']:,.0f} ({change['change_pct']:+.1f}%)
"""
        
        # Add strategic insights
        report += """
## Strategic Insights

### Channel Performance
"""
        marginal_roi = calculate_marginal_roi(self.model, self.current_budget)
        top_performers = sorted(
            marginal_roi.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        report += "**Highest Marginal ROI Channels**:\n"
        for channel, roi in top_performers:
            report += f"- {channel.replace('_', ' ').title()}: ${roi:.2f} per dollar\n"
        
        # Add risks and recommendations
        report += """

### Risk Considerations

1. **Implementation Timeline**: Changes should be phased over 2-3 weeks to monitor impact
2. **Market Conditions**: Results assume stable competitive environment
3. **Measurement**: Set up proper tracking before implementation

### Next Steps

1. Review and approve budget reallocation
2. Set up A/B tests for major changes
3. Implement tracking for ROI validation
4. Schedule follow-up optimization in 4 weeks

---
*Report generated: {date}*
        """.format(date=datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        return report

# Generate report
report_gen = MMM_ReportGenerator(mmm_model, optimal_result, current_budget)
executive_summary = report_gen.generate_executive_summary()
print(executive_summary)

# Save report
with open('mmm_optimization_report', 'w') as f:
    f.write(executive_summary)
```

### Step 8: Continuous Optimization

```python
class MMM_ContinuousOptimizer:
    """Continuous optimization with performance tracking."""
    
    def __init__(self, model, optimizer, tracking_config):
        self.model = model
        self.optimizer = optimizer
        self.tracking_config = tracking_config
        self.optimization_history = []
    
    def run_weekly_optimization(self, current_performance, market_conditions):
        """Run weekly optimization with latest data."""
        # Update model with latest performance
        self.model.update_with_actuals(current_performance)
        
        # Adjust constraints based on market
        constraints = self._adjust_constraints(market_conditions)
        
        # Run optimization
        result = self.optimizer.optimize(
            initial_budget=current_performance['budget'],
            constraints=constraints
        )
        
        # Track results
        self.optimization_history.append({
            'date': datetime.now(),
            'result': result,
            'market_conditions': market_conditions
        })
        
        return result
    
    def _adjust_constraints(self, market_conditions):
        """Dynamically adjust constraints based on market."""
        base_constraints = create_mmm_constraints()
        
        # Adjust based on competition
        if market_conditions.get('competitor_activity') == 'high':
            base_constraints['business_rules']['min_search_share'] = 0.35
        
        # Adjust for seasonality
        if market_conditions.get('season') == 'holiday':
            base_constraints['channel_bounds']['tv'] = (100_000, 500_000)
        
        return base_constraints
    
    def generate_performance_dashboard(self):
        """Generate performance tracking dashboard."""
        if not self.optimization_history:
            return "No optimization history available"
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MMM Optimization Performance Dashboard', fontsize=16)
        
        # Extract historical data
        dates = [h['date'] for h in self.optimization_history]
        predicted_sales = [h['result'].optimal_value for h in self.optimization_history]
        
        # Plot 1: Predicted sales over time
        axes[0, 0].plot(dates, predicted_sales, marker='o')
        axes[0, 0].set_title('Predicted Sales Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Predicted Sales ($)')
        
        # Plot 2: Budget allocation evolution
        channels = self.model.channels
        allocation_history = pd.DataFrame([
            h['result'].optimal_budget for h in self.optimization_history
        ], index=dates)
        
        allocation_history.plot(ax=axes[0, 1])
        axes[0, 1].set_title('Budget Allocation Evolution')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Budget ($)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: ROI trends
        roi_history = pd.DataFrame([
            self.model.calculate_roi(
                xr.Dataset({ch: xr.DataArray([h['result'].optimal_budget.get(ch, 0)]) 
                           for ch in channels})
            ) for h in self.optimization_history
        ], index=dates)
        
        roi_history.plot(ax=axes[1, 0])
        axes[1, 0].set_title('Channel ROI Trends')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('ROI')
        
        # Plot 4: Market conditions impact
        if len(self.optimization_history) > 1:
            conditions = [h['market_conditions'].get('competitor_activity', 'normal') 
                         for h in self.optimization_history]
            condition_impact = pd.Series(predicted_sales, index=dates)
            
            for condition in set(conditions):
                mask = [c == condition for c in conditions]
                axes[1, 1].scatter(
                    [d for d, m in zip(dates, mask) if m],
                    [s for s, m in zip(predicted_sales, mask) if m],
                    label=f'Competition: {condition}',
                    alpha=0.7
                )
            
            axes[1, 1].set_title('Sales vs Market Conditions')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Predicted Sales ($)')
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig

# Example usage
continuous_optimizer = MMM_ContinuousOptimizer(
    model=mmm_model,
    optimizer=optimizer,
    tracking_config={
        'update_frequency': 'weekly',
        'performance_threshold': 0.95,
        'reoptimize_trigger': 'auto'
    }
)
```

## Key Takeaways

This comprehensive example demonstrates:

1. **Data Preparation**: Loading and structuring marketing data for optimization
2. **Model Creation**: Building a realistic MMM with saturation and carryover effects
3. **Constraint Definition**: Setting up business-realistic constraints
4. **Optimization Execution**: Running single and multi-period optimizations
5. **Results Analysis**: Calculating ROI, marginal returns, and scenario analysis
6. **Visualization**: Creating insightful charts for decision-making
7. **Reporting**: Generating executive-friendly summaries
8. **Continuous Optimization**: Setting up ongoing optimization processes

## Next Steps

To adapt this example for your use case:

1. Replace sample data with your actual marketing data
2. Customize the model with your specific response curves
3. Adjust constraints based on your business rules
4. Add additional channels or modify existing ones
5. Integrate with your data pipeline for automation

For more advanced features, explore:
- Multi-Objective Optimization (coming soon)
- Real-Time Optimization(coming soon)
- [Custom Model Integration](../guides/model_integration.md)