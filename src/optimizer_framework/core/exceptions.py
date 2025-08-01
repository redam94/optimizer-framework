"""Custom exceptions for the optimizer framework."""


class OptimizerFrameworkError(Exception):
    """Base exception for all framework errors."""
    pass


class ConfigurationError(OptimizerFrameworkError):
    """Raised when configuration is invalid or missing."""
    pass


class ModelError(OptimizerFrameworkError):
    """Raised when model operations fail."""
    pass


class OptimizationError(OptimizerFrameworkError):
    """Raised when optimization fails."""
    pass


class DataValidationError(OptimizerFrameworkError):
    """Raised when data validation fails."""
    pass


class ConstraintViolationError(OptimizationError):
    """Raised when constraints cannot be satisfied."""
    pass
