"""Custom exceptions for the optimizer framework."""


class OptimizerFrameworkError(Exception):
    """Base exception for all framework errors."""

    pass  # pragma: no cover


class ConfigurationError(OptimizerFrameworkError):
    """Raised when configuration is invalid or missing."""

    pass  # pragma: no cover


class ModelError(OptimizerFrameworkError):
    """Raised when model operations fail."""

    pass  # pragma: no cover


class OptimizationError(OptimizerFrameworkError):
    """Raised when optimization fails."""

    pass  # pragma: no cover


class DataValidationError(OptimizerFrameworkError):
    """Raised when data validation fails."""

    pass  # pragma: no cover


class ConstraintViolationError(OptimizationError):
    """Raised when constraints cannot be satisfied."""

    pass  # pragma: no cover
