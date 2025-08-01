"""Core interfaces and abstract base classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Union

import xarray as xr
from pydantic import BaseModel


class ModelProtocol(Protocol):
    """Protocol defining the interface for all models."""
    
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """Generate predictions from input data."""
        ...
    
    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Calculate feature contributions."""
        ...


class AbstractModel(ABC):
    """Abstract base class for all models in the framework."""
    
    @abstractmethod
    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """
        Generate predictions from input data.
        
        Args:
            x: Input dataset with dimensions matching model requirements
            
        Returns:
            Predictions as DataArray with appropriate dimensions
        """
        pass
    
    @abstractmethod
    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        """
        Calculate feature contributions for the given input.
        
        Args:
            x: Input dataset
            
        Returns:
            Dataset containing contribution information
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the type identifier for this model."""
        pass
    
    @property
    @abstractmethod
    def required_dimensions(self) -> list[str]:
        """Return list of required dimensions for input data."""
        pass


class OptimizerProtocol(Protocol):
    """Protocol defining the interface for optimizers."""
    
    def optimize(
        self,
        model: ModelProtocol,
        constraints: Dict[str, Any],
        config: Dict[str, Any]
    ) -> "OptimizationResult":
        """Run optimization."""
        ...


class OptimizationResult(BaseModel):
    """Result container for optimization runs."""
    
    optimal_budget: Dict[str, float]
    optimal_value: float
    predictions: Optional[xr.DataArray] = None
    contributions: Optional[xr.Dataset] = None
    metadata: Dict[str, Any] = {}
    convergence_info: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True
