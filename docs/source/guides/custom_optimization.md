# Custom Optimization Strategy Guide

## Overview

This guide demonstrates how to create a custom optimization strategy in Atlas by solving a real-world challenge: optimizing inventory levels across a distribution network spanning 10 US states. We'll build a strategy that balances inventory holding costs against stockout risks while considering transportation constraints, demand uncertainty, and service level requirements.

## Business Problem - US Distribution Network Inventory Optimization

### The Challenge

A retail company operates:
- **10 Distribution Centers (DCs)** across the US
- **500+ Retail Stores** served by these DCs
- **5,000+ SKUs** with varying demand patterns
- **$50M+ in inventory** at any given time

Current pain points:
- High inventory carrying costs (20% of inventory value annually)
- Frequent stockouts in high-demand regions (5-8% lost sales)
- Inefficient inter-DC transfers
- Poor demand forecast accuracy (±30% error)

### Business Objectives

1. **Minimize Total Cost**: Inventory holding + transportation + stockout costs
2. **Maintain Service Levels**: 98% product availability across all stores
3. **Optimize Network Flow**: Efficient routing from DCs to stores
4. **Balance Risk**: Account for demand uncertainty and supply disruptions

## Strategy Design

### Core Components

Our custom strategy will integrate:
1. **Multi-Echelon Inventory Model**: Optimize across DC and store levels
2. **Demand Forecasting**: Incorporate uncertainty and seasonality
3. **Network Flow Optimization**: Minimize transportation costs
4. **Safety Stock Calculation**: Buffer against demand variability
5. **Dynamic Rebalancing**: Move inventory based on regional demand

### Mathematical Framework

The strategy optimizes:
- **Decision Variables**: Inventory levels at each DC, reorder points, transfer quantities
- **Objective Function**: Total cost (holding + transport + stockout)
- **Constraints**: DC capacity, service levels, lead times, budget

## Implementation Guide

### Step 1: Define the Strategy Interface

```python
from atlas.strategies import BaseStrategy
from atlas.optimization import OptimizationResult
import numpy as np
import pandas as pd

class DistributionNetworkStrategy(BaseStrategy):
    """
    Custom strategy for multi-echelon inventory optimization
    across a US distribution network.
    """
    
    def __init__(self, config):
        """
        Initialize strategy with network configuration.
        
        Args:
            config: Dictionary containing:
                - dc_locations: List of distribution center locations
                - store_assignments: Mapping of stores to DCs
                - product_catalog: SKU information
                - cost_parameters: Holding, transport, stockout costs
                - service_targets: Required service levels
        """
        super().__init__(config)
        self.dc_locations = config['dc_locations']
        self.store_assignments = config['store_assignments']
        self.products = config['product_catalog']
        self.costs = config['cost_parameters']
        self.service_targets = config['service_targets']
        
        # Initialize network graph
        self._build_network_graph()
        
        # Set up demand forecasting
        self._initialize_demand_models()
```

### Step 2: Implement Core Optimization Logic

```python
    def optimize(self, current_state, constraints, objectives):
        """
        Execute the inventory optimization strategy.
        
        Args:
            current_state: Current inventory levels and positions
            constraints: Business constraints (capacity, budget, etc.)
            objectives: Optimization objectives and weights
            
        Returns:
            OptimizationResult with recommended actions
        """
        # Step 1: Forecast demand by region and product
        demand_forecast = self._forecast_demand(
            horizon=self.config['planning_horizon'],
            confidence_level=0.95
        )
        
        # Step 2: Calculate optimal base stock levels
        base_stock_levels = self._optimize_base_stock(
            demand_forecast=demand_forecast,
            service_targets=self.service_targets,
            constraints=constraints
        )
        
        # Step 3: Determine reorder points and quantities
        reorder_plan = self._calculate_reorder_points(
            base_stock=base_stock_levels,
            lead_times=self.config['lead_times'],
            demand_variability=demand_forecast['std_dev']
        )
        
        # Step 4: Optimize inter-DC transfers
        transfer_plan = self._optimize_transfers(
            current_inventory=current_state['inventory_levels'],
            target_levels=base_stock_levels,
            transport_costs=self.costs['transportation']
        )
        
        # Step 5: Generate deployment schedule
        deployment = self._create_deployment_schedule(
            reorder_plan=reorder_plan,
            transfer_plan=transfer_plan,
            constraints=constraints
        )
        
        # Step 6: Calculate expected outcomes
        outcomes = self._simulate_outcomes(
            deployment=deployment,
            demand_scenarios=self._generate_demand_scenarios()
        )
        
        return OptimizationResult(
            actions=deployment,
            expected_cost=outcomes['total_cost'],
            service_level=outcomes['service_level'],
            risk_metrics=outcomes['risk_analysis']
        )
```

### Step 3: Implement Demand Forecasting

```python
    def _forecast_demand(self, horizon, confidence_level):
        """
        Forecast demand by DC, product, and time period.
        """
        forecasts = {}
        
        for dc in self.dc_locations:
            dc_forecast = {}
            
            # Get stores served by this DC
            served_stores = self.store_assignments[dc]
            
            for product in self.products:
                # Aggregate historical demand from served stores
                historical_demand = self._get_historical_demand(
                    stores=served_stores,
                    product=product,
                    periods=52  # 52 weeks of history
                )
                
                # Apply time series model with seasonality
                model = self._build_demand_model(
                    historical_demand,
                    seasonality=product.get('seasonal_pattern', 'none')
                )
                
                # Generate forecast with prediction intervals
                point_forecast = model.predict(horizon)
                prediction_interval = model.prediction_interval(
                    confidence_level=confidence_level
                )
                
                dc_forecast[product['sku']] = {
                    'mean': point_forecast,
                    'lower': prediction_interval[0],
                    'upper': prediction_interval[1],
                    'std_dev': self._calculate_demand_std(historical_demand)
                }
            
            forecasts[dc] = dc_forecast
        
        return forecasts
```

### Step 4: Optimize Base Stock Levels

```python
    def _optimize_base_stock(self, demand_forecast, service_targets, constraints):
        """
        Calculate optimal inventory levels for each DC-product combination.
        """
        from scipy.optimize import minimize
        
        # Decision variables: base stock level for each DC-product
        n_vars = len(self.dc_locations) * len(self.products)
        
        def objective_function(x):
            """Total cost: holding + expected stockout cost"""
            inventory_levels = x.reshape(len(self.dc_locations), len(self.products))
            
            total_cost = 0
            for i, dc in enumerate(self.dc_locations):
                for j, product in enumerate(self.products):
                    level = inventory_levels[i, j]
                    
                    # Holding cost
                    holding_cost = (
                        level * 
                        product['unit_cost'] * 
                        self.costs['holding_rate']
                    )
                    
                    # Expected stockout cost
                    demand_dist = demand_forecast[dc][product['sku']]
                    stockout_prob = self._calculate_stockout_probability(
                        inventory_level=level,
                        demand_mean=demand_dist['mean'],
                        demand_std=demand_dist['std_dev']
                    )
                    
                    stockout_cost = (
                        stockout_prob * 
                        demand_dist['mean'] * 
                        product['unit_cost'] * 
                        self.costs['stockout_penalty']
                    )
                    
                    total_cost += holding_cost + stockout_cost
            
            return total_cost
        
        # Constraints
        constraint_list = []
        
        # Service level constraints
        for i, dc in enumerate(self.dc_locations):
            for j, product in enumerate(self.products):
                def service_constraint(x, i=i, j=j):
                    level = x.reshape(len(self.dc_locations), len(self.products))[i, j]
                    demand_dist = demand_forecast[self.dc_locations[i]][self.products[j]['sku']]
                    
                    service_level = 1 - self._calculate_stockout_probability(
                        inventory_level=level,
                        demand_mean=demand_dist['mean'],
                        demand_std=demand_dist['std_dev']
                    )
                    
                    return service_level - service_targets.get(
                        self.products[j]['category'], 
                        0.95
                    )
                
                constraint_list.append({
                    'type': 'ineq',
                    'fun': service_constraint
                })
        
        # Capacity constraints
        for i, dc in enumerate(self.dc_locations):
            def capacity_constraint(x, i=i):
                dc_inventory = x.reshape(len(self.dc_locations), len(self.products))[i, :]
                total_volume = sum(
                    dc_inventory[j] * self.products[j]['unit_volume']
                    for j in range(len(self.products))
                )
                return constraints['dc_capacity'][self.dc_locations[i]] - total_volume
            
            constraint_list.append({
                'type': 'ineq',
                'fun': capacity_constraint
            })
        
        # Budget constraint
        def budget_constraint(x):
            total_inventory_value = sum(
                x[i] * self.products[i % len(self.products)]['unit_cost']
                for i in range(n_vars)
            )
            return constraints['inventory_budget'] - total_inventory_value
        
        constraint_list.append({
            'type': 'ineq',
            'fun': budget_constraint
        })
        
        # Solve optimization
        initial_guess = self._generate_initial_solution(demand_forecast)
        
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            constraints=constraint_list,
            options={'maxiter': 1000}
        )
        
        return result.x.reshape(len(self.dc_locations), len(self.products))
```

### Step 5: Optimize Network Transfers

```python
    def _optimize_transfers(self, current_inventory, target_levels, transport_costs):
        """
        Determine optimal inter-DC transfers to rebalance inventory.
        """
        from scipy.optimize import linprog
        
        n_dcs = len(self.dc_locations)
        n_products = len(self.products)
        
        # Decision variables: transfer[from_dc, to_dc, product]
        n_vars = n_dcs * n_dcs * n_products
        
        # Objective: minimize transportation cost
        c = []
        for from_dc in range(n_dcs):
            for to_dc in range(n_dcs):
                for product in range(n_products):
                    if from_dc == to_dc:
                        c.append(0)  # No cost for not transferring
                    else:
                        # Cost based on distance and product weight
                        distance = self._get_distance(
                            self.dc_locations[from_dc],
                            self.dc_locations[to_dc]
                        )
                        weight = self.products[product]['weight']
                        cost_per_mile = transport_costs['per_pound_mile']
                        c.append(distance * weight * cost_per_mile)
        
        # Constraints
        A_eq = []
        b_eq = []
        
        # Conservation constraints: net flow equals target - current
        for dc in range(n_dcs):
            for product in range(n_products):
                constraint = [0] * n_vars
                
                # Outflows from this DC
                for to_dc in range(n_dcs):
                    if to_dc != dc:
                        idx = from_dc * n_dcs * n_products + to_dc * n_products + product
                        constraint[idx] = -1
                
                # Inflows to this DC
                for from_dc in range(n_dcs):
                    if from_dc != dc:
                        idx = from_dc * n_dcs * n_products + dc * n_products + product
                        constraint[idx] = 1
                
                A_eq.append(constraint)
                b_eq.append(
                    target_levels[dc, product] - current_inventory[dc, product]
                )
        
        # Bounds: non-negative transfers, limited by current inventory
        bounds = []
        for from_dc in range(n_dcs):
            for to_dc in range(n_dcs):
                for product in range(n_products):
                    if from_dc == to_dc:
                        bounds.append((0, 0))
                    else:
                        max_transfer = current_inventory[from_dc, product]
                        bounds.append((0, max_transfer))
        
        # Solve
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        # Parse results into transfer plan
        transfer_plan = []
        if result.success:
            transfers = result.x.reshape(n_dcs, n_dcs, n_products)
            
            for from_dc in range(n_dcs):
                for to_dc in range(n_dcs):
                    for product in range(n_products):
                        quantity = transfers[from_dc, to_dc, product]
                        if quantity > 0.1:  # Threshold to avoid tiny transfers
                            transfer_plan.append({
                                'from': self.dc_locations[from_dc],
                                'to': self.dc_locations[to_dc],
                                'product': self.products[product]['sku'],
                                'quantity': int(quantity),
                                'cost': quantity * c[from_dc * n_dcs * n_products + 
                                                   to_dc * n_products + product]
                            })
        
        return transfer_plan
```

### Step 6: Handle Uncertainty and Risk

```python
    def _simulate_outcomes(self, deployment, demand_scenarios):
        """
        Monte Carlo simulation to assess strategy performance under uncertainty.
        """
        n_simulations = 1000
        results = {
            'service_levels': [],
            'total_costs': [],
            'stockout_events': []
        }
        
        for sim in range(n_simulations):
            # Sample demand scenario
            demand_scenario = self._sample_demand_scenario()
            
            # Simulate inventory dynamics
            inventory_trace = self._simulate_inventory_dynamics(
                initial_state=deployment['initial_inventory'],
                reorder_plan=deployment['reorder_plan'],
                transfer_plan=deployment['transfer_plan'],
                demand=demand_scenario,
                lead_times=self._sample_lead_times()
            )
            
            # Calculate metrics
            service_level = self._calculate_service_level(inventory_trace)
            total_cost = self._calculate_total_cost(inventory_trace)
            stockouts = self._identify_stockout_events(inventory_trace)
            
            results['service_levels'].append(service_level)
            results['total_costs'].append(total_cost)
            results['stockout_events'].extend(stockouts)
        
        # Aggregate results
        return {
            'service_level': {
                'mean': np.mean(results['service_levels']),
                'p95': np.percentile(results['service_levels'], 5),
                'p99': np.percentile(results['service_levels'], 1)
            },
            'total_cost': {
                'mean': np.mean(results['total_costs']),
                'std': np.std(results['total_costs']),
                'p95': np.percentile(results['total_costs'], 95)
            },
            'risk_analysis': {
                'stockout_risk': len(results['stockout_events']) / n_simulations,
                'value_at_risk': np.percentile(results['total_costs'], 95),
                'conditional_value_at_risk': np.mean([
                    c for c in results['total_costs'] 
                    if c > np.percentile(results['total_costs'], 95)
                ])
            }
        }
```

### Step 7: Create Performance Monitoring

```python
    def monitor_performance(self, actual_data, predictions):
        """
        Track strategy performance and adapt parameters.
        """
        metrics = {
            'forecast_accuracy': self._calculate_forecast_accuracy(
                actual_data['demand'],
                predictions['demand_forecast']
            ),
            'service_achievement': self._calculate_service_achievement(
                actual_data['stockouts'],
                self.service_targets
            ),
            'cost_variance': self._calculate_cost_variance(
                actual_data['costs'],
                predictions['expected_cost']
            ),
            'network_efficiency': self._calculate_network_efficiency(
                actual_data['transfers'],
                actual_data['transportation_costs']
            )
        }
        
        # Adaptive learning
        if metrics['forecast_accuracy'] < 0.8:
            self._retrain_demand_models(actual_data)
        
        if metrics['service_achievement'] < 0.95:
            self._adjust_safety_stock_parameters()
        
        return metrics
```

## Integration with Atlas

### Step 1: Register the Strategy

```python
from atlas import StrategyRegistry

# Register the custom strategy
StrategyRegistry.register(
    name='distribution_network_optimization',
    strategy_class=DistributionNetworkStrategy,
    description='Multi-echelon inventory optimization for US distribution',
    version='1.0.0'
)
```

### Step 2: Configure the Strategy

```python
# Define network configuration
config = {
    'dc_locations': [
        {'id': 'DC_CA', 'state': 'CA', 'lat': 34.0522, 'lon': -118.2437},
        {'id': 'DC_TX', 'state': 'TX', 'lat': 32.7767, 'lon': -96.7970},
        {'id': 'DC_FL', 'state': 'FL', 'lat': 28.5383, 'lon': -81.3792},
        {'id': 'DC_NY', 'state': 'NY', 'lat': 40.7128, 'lon': -74.0060},
        {'id': 'DC_IL', 'state': 'IL', 'lat': 41.8781, 'lon': -87.6298},
        {'id': 'DC_WA', 'state': 'WA', 'lat': 47.6062, 'lon': -122.3321},
        {'id': 'DC_GA', 'state': 'GA', 'lat': 33.7490, 'lon': -84.3880},
        {'id': 'DC_OH', 'state': 'OH', 'lat': 39.9612, 'lon': -82.9988},
        {'id': 'DC_CO', 'state': 'CO', 'lat': 39.7392, 'lon': -104.9903},
        {'id': 'DC_NC', 'state': 'NC', 'lat': 35.2271, 'lon': -80.8431}
    ],
    'store_assignments': load_store_assignments(),
    'product_catalog': load_product_catalog(),
    'cost_parameters': {
        'holding_rate': 0.20,  # 20% annual holding cost
        'stockout_penalty': 2.5,  # 2.5x unit cost for stockouts
        'transportation': {
            'per_pound_mile': 0.15,
            'fixed_cost': 250
        }
    },
    'service_targets': {
        'high_priority': 0.99,
        'standard': 0.95,
        'low_priority': 0.90
    },
    'planning_horizon': 52,  # weeks
    'lead_times': {
        'supplier_to_dc': 14,  # days
        'dc_to_dc': 3,        # days
        'dc_to_store': 1      # days
    }
}

# Create strategy instance
strategy = DistributionNetworkStrategy(config)
```

### Step 3: Execute Optimization

```python
from atlas import OptimizationEngine

# Create optimization engine with the custom strategy
engine = OptimizationEngine(strategy=strategy)

# Define current state
current_state = {
    'inventory_levels': load_current_inventory(),
    'in_transit': load_in_transit_inventory(),
    'open_orders': load_open_orders()
}

# Define constraints
constraints = {
    'inventory_budget': 50_000_000,  # $50M total inventory
    'dc_capacity': {
        'DC_CA': 100_000,  # cubic feet
        'DC_TX': 150_000,
        'DC_FL': 80_000,
        'DC_NY': 120_000,
        'DC_IL': 130_000,
        'DC_WA': 90_000,
        'DC_GA': 110_000,
        'DC_OH': 100_000,
        'DC_CO': 85_000,
        'DC_NC': 95_000
    },
    'max_transfers_per_week': 50,
    'min_shipment_size': 1000  # pounds
}

# Define objectives
objectives = {
    'minimize_total_cost': 0.6,
    'maximize_service_level': 0.3,
    'minimize_transfers': 0.1
}

# Run optimization
result = engine.optimize(
    current_state=current_state,
    constraints=constraints,
    objectives=objectives
)

# Review results
print(f"Expected total cost: ${result.expected_cost:,.2f}")
print(f"Expected service level: {result.service_level:.1%}")
print(f"Number of transfers: {len(result.actions['transfers'])}")
print(f"Risk metrics: {result.risk_metrics}")
```

## Testing and Validation

### Unit Testing

```python
import unittest
from atlas.testing import StrategyTestCase

class TestDistributionNetworkStrategy(StrategyTestCase):
    def setUp(self):
        self.strategy = DistributionNetworkStrategy(test_config)
    
    def test_demand_forecasting(self):
        """Test demand forecasting accuracy"""
        historical_data = generate_test_demand_data()
        forecast = self.strategy._forecast_demand(
            horizon=4,
            confidence_level=0.95
        )
        
        # Check forecast structure
        self.assertEqual(len(forecast), 10)  # 10 DCs
        
        # Check forecast reasonableness
        for dc in forecast:
            for product in forecast[dc]:
                self.assertGreater(forecast[dc][product]['mean'], 0)
                self.assertLess(
                    forecast[dc][product]['lower'],
                    forecast[dc][product]['upper']
                )
    
    def test_base_stock_optimization(self):
        """Test base stock calculation"""
        demand_forecast = generate_test_forecast()
        base_stock = self.strategy._optimize_base_stock(
            demand_forecast=demand_forecast,
            service_targets={'standard': 0.95},
            constraints={'inventory_budget': 1_000_000}
        )
        
        # Check all products have base stock
        self.assertEqual(base_stock.shape[0], 10)  # DCs
        self.assertGreater(base_stock.min(), 0)
    
    def test_transfer_optimization(self):
        """Test inter-DC transfer logic"""
        current = np.random.randint(0, 1000, (10, 100))
        target = np.random.randint(0, 1000, (10, 100))
        
        transfers = self.strategy._optimize_transfers(
            current_inventory=current,
            target_levels=target,
            transport_costs={'per_pound_mile': 0.15}
        )
        
        # Verify transfer feasibility
        for transfer in transfers:
            from_idx = self._get_dc_index(transfer['from'])
            product_idx = self._get_product_index(transfer['product'])
            self.assertLessEqual(
                transfer['quantity'],
                current[from_idx, product_idx]
            )
```

### Integration Testing

```python
def test_end_to_end_optimization():
    """Test complete optimization workflow"""
    # Setup test scenario
    test_scenario = create_test_scenario(
        n_dcs=10,
        n_products=100,
        n_stores=500,
        demand_pattern='seasonal'
    )
    
    # Run optimization
    result = run_optimization_test(
        scenario=test_scenario,
        strategy=DistributionNetworkStrategy,
        iterations=10
    )
    
    # Validate results
    assert result.service_level['mean'] >= 0.95
    assert result.total_cost['mean'] <= test_scenario.budget
    assert all(transfer['quantity'] > 0 for transfer in result.transfers)
```

### Performance Testing

```python
def benchmark_strategy_performance():
    """Benchmark strategy performance at scale"""
    benchmarks = []
    
    for scale in [100, 500, 1000, 5000]:  # Number of SKUs
        start_time = time.time()
        
        config = generate_scaled_config(n_products=scale)
        strategy = DistributionNetworkStrategy(config)
        
        result = strategy.optimize(
            current_state=generate_scaled_state(scale),
            constraints=standard_constraints,
            objectives=standard_objectives
        )
        
        execution_time = time.time() - start_time
        
        benchmarks.append({
            'scale': scale,
            'execution_time': execution_time,
            'memory_usage': get_memory_usage(),
            'solution_quality': result.expected_cost
        })
    
    return benchmarks
```

## Deployment Considerations

### Configuration Management

```yaml
# config/distribution_strategy.yaml
strategy:
  name: distribution_network_optimization
  version: 1.0.0
  
network:
  dc_locations: !include dc_locations.yaml
  store_mappings: !include store_mappings.yaml
  
parameters:
  planning_horizon: 52
  reorder_frequency: weekly
  safety_stock_method: dynamic
  
costs:
  holding_rate: 0.20
  stockout_penalty: 2.5
  transportation:
    ltl_rate: 0.15  # per pound-mile
    ftl_rate: 2.50  # per mile
    
optimization:
  solver: SLSQP
  max_iterations: 1000
  tolerance: 1e-6
  
monitoring:
  metrics_frequency: daily
  alert_thresholds:
    service_level: 0.93
    forecast_accuracy: 0.75
```

### Monitoring and Alerts

```python
from atlas.monitoring import StrategyMonitor

monitor = StrategyMonitor(
    strategy='distribution_network_optimization',
    metrics=[
        'service_level',
        'total_cost',
        'forecast_accuracy',
        'transfer_efficiency'
    ],
    alert_channels=['email', 'slack']
)

# Set up alerts
monitor.add_alert(
    condition='service_level < 0.95',
    severity='high',
    message='Service level below target'
)

monitor.add_alert(
    condition='total_cost > budget * 1.1',
    severity='critical',
    message='Cost overrun exceeding 10%'
)
```

### Continuous Improvement

```python
from atlas.learning import StrategyLearner

learner = StrategyLearner(strategy)

# Learn from historical performance
learner.train(
    historical_data=load_historical_results(),
    performance_metrics=['cost', 'service_level'],
    parameter_ranges={
        'safety_stock_multiplier': (1.0, 3.0),
        'reorder_threshold': (0.1, 0.5),
        'transfer_threshold': (100, 1000)
    }
)

# Apply learned parameters
optimized_params = learner.get_optimal_parameters()
strategy.update_parameters(optimized_params)
```

## Results and Benefits

### Expected Outcomes

Based on typical implementations, this strategy delivers:

1. **Cost Reduction**: 15-25% reduction in total inventory costs
2. **Service Improvement**: 2-5% increase in product availability
3. **Network Efficiency**: 30% reduction in inter-DC transfers
4. **Inventory Optimization**: 20% reduction in working capital

### Key Success Factors

1. **Data Quality**: Accurate demand history and lead times
2. **Model Calibration**: Regular updating of forecast models
3. **Change Management**: Gradual rollout with monitoring
4. **Stakeholder Buy-in**: Clear communication of benefits

## Conclusion

This custom optimization strategy demonstrates how Atlas enables sophisticated, domain-specific optimization while maintaining clean abstractions and reusable patterns. The distribution network optimization strategy can be adapted for various industries and scaled from small regional networks to global supply chains.

Key takeaways:
1. Atlas's flexibility allows encoding complex business logic
2. Custom strategies can leverage multiple optimization techniques
3. Uncertainty and risk can be explicitly modeled
4. Performance scales well with proper design
5. Continuous learning improves results over time

For more information on building custom strategies, see the [Atlas Strategy Development Guide](../guides/strategy_development.md).