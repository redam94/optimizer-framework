"""Custom exceptions for the optimizer framework."""


class OptimizerFrameworkError(Exception):  # pragma: no cover
    """Base exception for all framework errors."""

    pass  # pragma: no cover


class ConfigurationError(OptimizerFrameworkError):  # pragma: no cover
    """Raised when configuration is invalid or missing."""

    pass  # pragma: no cover


class ModelError(OptimizerFrameworkError):  # pragma: no cover
    """Raised when model operations fail."""

    pass  # pragma: no cover


class OptimizationError(OptimizerFrameworkError):  # pragma: no cover
    """Raised when optimization fails."""

    pass  # pragma: no cover


class DataValidationError(OptimizerFrameworkError):  # pragma: no cover
    """Raised when data validation fails."""

    pass  # pragma: no cover


class ConstraintViolationError(OptimizationError):  # pragma: no cover
    """Raised when constraints cannot be satisfied."""

    pass  # pragma: no cover
