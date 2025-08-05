# atlas_complete_example.py
"""
Complete example showing how to use Atlas base classes and visualizers
for a marketing budget optimization scenario
"""

from curses.ascii import alt
import json
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import altair

# Import our base classes and visualizers
from atlas.core.baseclasses import (
    BaseOptimizer,
    Budget,
    CompositeStrategy,
    Constraint,
    ConstraintType,
    GradientBasedOptimizer,
    OptimizationResult,
    OptimizationStrategy,
    RiskAdjustedStrategy,
    ROIMaximizationStrategy,
)
from atlas.core.visualizer import (
    # AllocationVisualizer,
    # ConvergenceVisualizer,
    ExportableVisualizationManager,
    InteractiveOptimizationDashboard,
    WhatIfAnalysisVisualizer,
)

# ==================== Custom Model Implementation ====================


class MarketingMixModel:
    """Simple marketing mix model with saturation and carryover effects"""

    def __init__(
        self,
        base_effectiveness: Dict[str, float],
        saturation_params: Dict[str, float],
        carryover_rates: Dict[str, float],
    ):
        self.base_effectiveness = base_effectiveness
        self.saturation_params = saturation_params
        self.carryover_rates = carryover_rates
        self.previous_spend = {k: 0 for k in base_effectiveness.keys()}

    def predict(self, budget_array) -> float:
        """Predict revenue based on budget allocation"""
        total_revenue = 0

        # Handle both xarray and dict inputs
        if hasattr(budget_array, "to_dict"):
            allocations = budget_array.to_dict()["data"]
            channels = budget_array.coords["channel"].values
            budget_dict = dict(zip(channels, allocations))
        else:
            budget_dict = budget_array

        for channel, amount in budget_dict.items():
            if channel in self.base_effectiveness:
                # Apply carryover effect
                effective_spend = (
                    amount + self.carryover_rates[channel] * self.previous_spend[channel]
                )

                # Apply saturation curve (Hill transformation)
                alpha = self.saturation_params[channel]
                saturated_effect = effective_spend / (effective_spend + alpha)

                # Calculate revenue contribution
                channel_revenue = (
                    self.base_effectiveness[channel] * saturated_effect * effective_spend
                )
                total_revenue += channel_revenue

                # Update previous spend for next prediction
                self.previous_spend[channel] = amount

        return total_revenue


# ==================== Custom Optimization Strategies ====================


class SeasonalStrategy(OptimizationStrategy):
    """Strategy that accounts for seasonal patterns"""

    def __init__(
        self, seasonal_multipliers: Dict[str, float], current_season: str, name: str = None
    ):
        super().__init__(name or "Seasonal Strategy")
        self.seasonal_multipliers = seasonal_multipliers
        self.current_season = current_season

    def calculate_objective(self, allocation: Budget, model, **kwargs) -> float:
        """Calculate objective with seasonal adjustments"""
        base_prediction = model.predict(allocation.to_xarray())

        # Apply seasonal multiplier
        seasonal_factor = self.seasonal_multipliers.get(self.current_season, 1.0)

        return base_prediction * seasonal_factor


class IncrementalityStrategy(OptimizationStrategy):
    """Strategy focused on incremental returns"""

    def __init__(
        self,
        baseline_spend: Dict[str, float],
        incrementality_curves: Dict[str, callable],
        name: str = None,
    ):
        super().__init__(name or "Incrementality Strategy")
        self.baseline_spend = baseline_spend
        self.incrementality_curves = incrementality_curves

    def calculate_objective(self, allocation: Budget, model, **kwargs) -> float:
        """Calculate incremental value above baseline"""
        total_incremental = 0

        for channel, amount in allocation.allocations.items():
            baseline = self.baseline_spend.get(channel, 0)
            incremental_spend = max(0, amount - baseline)

            # Apply incrementality curve
            if channel in self.incrementality_curves:
                incremental_factor = self.incrementality_curves[channel](incremental_spend)
            else:
                incremental_factor = 1.0

            # Calculate incremental contribution
            channel_budget = Budget({channel: amount})
            channel_value = model.predict(channel_budget.to_xarray())
            incremental_value = channel_value * incremental_factor

            total_incremental += incremental_value

        return total_incremental


# ==================== Advanced Optimizer Implementation ====================


class MultiStageOptimizer(BaseOptimizer):
    """Optimizer that runs multiple stages with different strategies"""

    def __init__(self, strategies: List[OptimizationStrategy], stage_configs: List[Dict[str, Any]]):
        # Initialize with first strategy
        super().__init__(strategies[0], stage_configs[0])
        self.strategies = strategies
        self.stage_configs = stage_configs
        self.stage_results = []

    def optimize(
        self, initial_budget: Budget, constraints: List[Constraint], model
    ) -> OptimizationResult:
        """Run multi-stage optimization"""

        current_budget = initial_budget.copy()
        all_convergence_history = []
        total_iterations = 0
        start_time = datetime.now()

        # Run each stage
        for stage_idx, (strategy, config) in enumerate(zip(self.strategies, self.stage_configs)):
            self.logger.info(f"Running stage {stage_idx + 1}: {strategy.name}")

            # Create stage optimizer
            stage_optimizer = GradientBasedOptimizer(strategy, config)

            # Run stage optimization
            stage_result = stage_optimizer.optimize(current_budget, constraints, model)

            # Store stage results
            self.stage_results.append(
                {"stage": stage_idx + 1, "strategy": strategy.name, "result": stage_result}
            )

            # Update for next stage
            current_budget = stage_result.optimal_budget
            all_convergence_history.extend(stage_result.convergence_history)
            total_iterations += stage_result.iterations

        time_elapsed = (datetime.now() - start_time).total_seconds()

        # Return final result
        return OptimizationResult(
            optimal_budget=current_budget,
            optimal_value=self.stage_results[-1]["result"].optimal_value,
            iterations=total_iterations,
            success=True,
            convergence_history=all_convergence_history,
            time_elapsed=time_elapsed,
            metadata={
                "optimizer": "MultiStageOptimizer",
                "stages": len(self.strategies),
                "stage_results": self.stage_results,
            },
        )


# ==================== Complete Optimization Workflow ====================


def run_marketing_optimization_example():
    """Run a complete marketing budget optimization example"""

    print("=" * 60)
    print("Atlas Marketing Budget Optimization Example")
    print("=" * 60)

    # 1. Setup model
    print("\n1. Setting up marketing mix model...")

    model = MarketingMixModel(
        base_effectiveness={
            "TV": 2.5,
            "Digital": 3.2,
            "Radio": 1.8,
            "Print": 1.2,
            "Social": 2.8,
            "Outdoor": 1.5,
        },
        saturation_params={
            "TV": 500000,
            "Digital": 300000,
            "Radio": 150000,
            "Print": 100000,
            "Social": 200000,
            "Outdoor": 100000,
        },
        carryover_rates={
            "TV": 0.3,
            "Digital": 0.1,
            "Radio": 0.2,
            "Print": 0.4,
            "Social": 0.15,
            "Outdoor": 0.35,
        },
    )

    # 2. Define constraints
    print("\n2. Setting up constraints...")

    constraints = [
        Constraint(
            name="total_budget",
            type=ConstraintType.EQUALITY,
            function=lambda b: b.total(),
            value=1000000,  # $1M total budget
        ),
        Constraint(
            name="channel_bounds",
            type=ConstraintType.BOUNDS,
            bounds={
                "TV": (50000, 400000),
                "Digital": (100000, 500000),
                "Radio": (20000, 150000),
                "Print": (10000, 100000),
                "Social": (50000, 300000),
                "Outdoor": (10000, 100000),
            },
        ),
        Constraint(
            name="digital_minimum",
            type=ConstraintType.INEQUALITY,
            function=lambda b: b.allocations.get("Digital", 0) / b.total(),
            value=0.25,  # At least 25% to digital
        ),
    ]

    # 3. Create optimization strategies
    print("\n3. Creating optimization strategies...")

    # ROI Maximization
    roi_strategy = ROIMaximizationStrategy(saturation_params=model.saturation_params)

    # Risk Adjusted
    risk_strategy = RiskAdjustedStrategy(risk_aversion=0.3)

    # Seasonal
    seasonal_strategy = SeasonalStrategy(
        seasonal_multipliers={"Q1": 0.9, "Q2": 1.0, "Q3": 0.8, "Q4": 1.3}, current_season="Q4"
    )

    # Composite strategy
    composite_strategy = CompositeStrategy(
        [(roi_strategy, 0.5), (risk_strategy, 0.3), (seasonal_strategy, 0.2)]
    )

    # 4. Run optimizations
    print("\n4. Running optimizations...")

    initial_budget = Budget(
        {
            "TV": 200000,
            "Digital": 300000,
            "Radio": 100000,
            "Print": 50000,
            "Social": 250000,
            "Outdoor": 100000,
        }
    )

    # Single strategy optimization
    print("\n   - ROI Maximization...")
    roi_optimizer = GradientBasedOptimizer(
        roi_strategy, config={"learning_rate": 0.01, "max_iterations": 200}
    )
    roi_result = roi_optimizer.optimize(initial_budget, constraints, model)

    # Composite strategy optimization
    print("   - Composite Strategy...")
    composite_optimizer = GradientBasedOptimizer(
        composite_strategy, config={"learning_rate": 0.01, "max_iterations": 200}
    )
    composite_result = composite_optimizer.optimize(initial_budget, constraints, model)

    # Multi-stage optimization
    print("   - Multi-Stage Optimization...")
    multi_optimizer = MultiStageOptimizer(
        strategies=[roi_strategy, risk_strategy, seasonal_strategy],
        stage_configs=[
            {"learning_rate": 0.02, "max_iterations": 100},
            {"learning_rate": 0.01, "max_iterations": 50},
            {"learning_rate": 0.005, "max_iterations": 50},
        ],
    )
    multi_result = multi_optimizer.optimize(initial_budget, constraints, model)

    # 5. Create visualizations
    print("\n5. Creating visualizations...")

    # Convergence visualization
    # conv_viz = ConvergenceVisualizer()
    # convergence_chart = conv_viz.create_convergence_chart(
    #     [roi_result, composite_result, multi_result]
    # )

    # Allocation comparison
    # alloc_viz = AllocationVisualizer()
    # allocation_chart = alloc_viz.create_allocation_chart(
    #     budgets=[
    #         initial_budget,
    #         roi_result.optimal_budget,
    #         composite_result.optimal_budget,
    #         multi_result.optimal_budget,
    #     ],
    #     labels=["Initial", "ROI Max", "Composite", "Multi-Stage"],
    # )

    # What-if analysis
    whatif_viz = WhatIfAnalysisVisualizer()
    sensitivity_fig = whatif_viz.create_sensitivity_analysis(
        model,
        roi_result.optimal_budget,
        ["TV", "Digital", "Social"],
        variation_range=(0.7, 1.3),
        steps=15,
    )

    # Interactive dashboard
    dashboard = InteractiveOptimizationDashboard()
    scenario_chart = dashboard.create_scenario_comparison_tool(
        roi_result,
        [
            {
                "name": "Conservative",
                "budget": composite_result.optimal_budget,
                "objective": composite_result.optimal_value,
            },
            {
                "name": "Aggressive",
                "budget": multi_result.optimal_budget,
                "objective": multi_result.optimal_value,
            },
        ],
    )

    # 6. Export for UI
    print("\n6. Exporting visualizations for UI integration...")

    export_manager = ExportableVisualizationManager()
    # export_manager.register_visualization("convergence", convergence_chart)
    # export_manager.register_visualization("allocations", allocation_chart)
    export_manager.register_visualization("sensitivity", sensitivity_fig)
    print(isinstance(scenario_chart, altair.HConcatChart | altair.VConcatChart | altair.Chart))
    export_manager.register_visualization("scenarios", scenario_chart)

    # Create API response
    api_response = export_manager.create_visualization_api_response()

    # Save to file
    with open("optimization_results.json", "w") as f:
        json.dump(api_response, f, indent=2)

    # 7. Print results summary
    print("\n7. Optimization Results Summary")
    print("=" * 60)

    print(f"\nInitial Budget Total: ${initial_budget.total():,.2f}")
    print(f"Initial Objective Value: ${model.predict(initial_budget.to_xarray()):,.2f}")
    print(f"Baseline No Budget: ${model.predict(initial_budget.to_xarray()*0):,.2f}")

    print(f"\nROI Maximization Results:")
    print(f"  - Total Budget: ${roi_result.optimal_budget.to_xarray().sum(...).item():,.2f}")
    print(f"  - Optimal Value: ${roi_result.optimal_value:,.2f}")
    print(f"  - Iterations: {roi_result.iterations}")
    print(f"  - Time: {roi_result.time_elapsed:.2f}s")

    print(f"\nComposite Strategy Results:")
    print(f"  - Optimal Value: ${composite_result.optimal_value:,.2f}")
    print(f"  - Iterations: {composite_result.iterations}")
    print(f"  - Time: {composite_result.time_elapsed:.2f}s")

    print(f"\nMulti-Stage Results:")
    print(f"  - Optimal Value: ${multi_result.optimal_value:,.2f}")
    print(f"  - Total Iterations: {multi_result.iterations}")
    print(f"  - Time: {multi_result.time_elapsed:.2f}s")

    print("\n\nOptimal Budget Allocations (ROI Max):")
    print("-" * 40)
    for channel, amount in roi_result.optimal_budget.allocations.items():
        percentage = amount / roi_result.optimal_budget.total() * 100
        print(f"{channel:10} ${amount:>10,.0f} ({percentage:>5.1f}%)")

    print("\n✅ Optimization complete! Results saved to optimization_results.json")

    return {
        "model": model,
        "results": {"roi": roi_result, "composite": composite_result, "multi_stage": multi_result},
        "visualizations": api_response,
        "export_manager": export_manager,
    }


# ==================== FastAPI Integration Example ====================

FASTAPI_INTEGRATION = """
# fastapi_atlas_integration.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio

app = FastAPI(title="Atlas Optimization API")

# Request/Response models
class BudgetRequest(BaseModel):
    allocations: Dict[str, float]
    
class OptimizationRequest(BaseModel):
    initial_budget: BudgetRequest
    constraints: List[Dict[str, Any]]
    strategy: str = "roi_maximization"
    config: Optional[Dict[str, Any]] = None

class OptimizationResponse(BaseModel):
    optimal_budget: Dict[str, float]
    optimal_value: float
    iterations: int
    success: bool
    visualizations: Optional[List[Dict[str, Any]]] = None

# Optimization endpoint
@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_budget(request: OptimizationRequest):
    try:
        # Convert request to Budget object
        initial_budget = Budget(request.initial_budget.allocations)
        
        # Create model and strategy
        model = create_model()  # Your model creation logic
        strategy = create_strategy(request.strategy)
        
        # Create optimizer
        optimizer = GradientBasedOptimizer(
            strategy, 
            request.config or {'max_iterations': 100}
        )
        
        # Run optimization (in background for long-running tasks)
        result = await asyncio.to_thread(
            optimizer.optimize,
            initial_budget,
            parse_constraints(request.constraints),
            model
        )
        
        # Create visualizations
        viz_manager = ExportableVisualizationManager()
        # ... create and register visualizations
        
        return OptimizationResponse(
            optimal_budget=result.optimal_budget.allocations,
            optimal_value=result.optimal_value,
            iterations=result.iterations,
            success=result.success,
            visualizations=viz_manager.create_visualization_api_response()['visualizations']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Visualization endpoints
@app.get("/visualizations/{optimization_id}")
async def get_visualizations(optimization_id: str):
    # Retrieve stored visualizations
    pass

# What-if analysis endpoint
@app.post("/whatif")
async def what_if_analysis(
    base_budget: BudgetRequest,
    scenarios: List[Dict[str, Any]]
):
    # Run what-if analysis
    pass

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Atlas Optimization API"}
"""


if __name__ == "__main__":
    # Run the complete example
    results = run_marketing_optimization_example()

    # print("\n\n" + "=" * 60)
    # print("FastAPI Integration Example")
    # print("=" * 60)
    # print("\nSee FASTAPI_INTEGRATION variable for FastAPI integration code")
    # print("\nTo run the API:")
    # print("1. Save the FastAPI code to 'fastapi_atlas_integration.py'")
    # print("2. Run: uvicorn fastapi_atlas_integration:app --reload")
    # print("3. Visit: http://localhost:8000/docs for API documentation")
