"""
Scikit-learn Model Factory for Atlas Optimization Framework

This module provides a factory pattern for creating and managing
scikit-learn models within the Atlas framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # type: ignore
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # type: ignore
from sklearn.svm import SVR  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from xgboost import XGBRegressor

from atlas.models.wrappers.sklearn_wrapper import SklearnModelWrapper


class SklearnModelFactory:
    """
    Factory class for creating scikit-learn model wrappers for Atlas.

    Supports various regression models and handles configuration management.
    """

    # Registry of supported model types
    MODEL_REGISTRY = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "decision_tree": DecisionTreeRegressor,
        "svr": SVR,
        "xgboost": XGBRegressor,
        "lightgbm": LGBMRegressor,
    }

    @classmethod
    def create(
        cls,
        model_type: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None,
        model_instance: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        scaler_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SklearnModelWrapper:
        """
        Create a scikit-learn model wrapper.

        Args:
            model_type: Type of model to create (if creating new)
            model_path: Path to saved model file
            model_instance: Pre-instantiated model
            feature_names: List of feature names
            target_name: Name of target variable
            scaler_path: Path to saved scaler
            config: Additional configuration
            **kwargs: Additional arguments for SklearnModelWrapper

        Returns:
            SklearnModelWrapper instance
        """
        # Merge config with kwargs
        if config:
            kwargs.update(config)

        # Determine how to create/load the model
        if model_instance is not None:
            # Use provided model instance
            return SklearnModelWrapper(
                model=model_instance,
                feature_names=feature_names,
                target_name=target_name,
                scaler_path=scaler_path,
                **kwargs,
            )
        elif model_path is not None:
            # Load from file
            return SklearnModelWrapper(
                model_path=model_path,
                feature_names=feature_names,
                target_name=target_name,
                scaler_path=scaler_path,
                **kwargs,
            )
        elif model_type is not None:
            # Create new model
            model_class = cls.MODEL_REGISTRY.get(model_type.lower())
            if model_class is None:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available types: {list(cls.MODEL_REGISTRY.keys())}"
                )

            # Extract model parameters from kwargs
            model_params = kwargs.pop("model_params", {})
            model = model_class(**model_params)

            return SklearnModelWrapper(
                model=model,
                feature_names=feature_names,
                target_name=target_name,
                scaler_path=scaler_path,
                model_type=f"sklearn_{model_type}",
                **kwargs,
            )
        else:
            raise ValueError("Must provide either 'model_type', 'model_path', or 'model_instance'")

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> SklearnModelWrapper:
        """
        Create model wrapper from configuration file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            SklearnModelWrapper instance
        """
        config_path = Path(config_path)

        # Load configuration
        if config_path.suffix in [".yml", ".yaml"]:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Create model from config
        return cls.create(**config)

    @classmethod
    def register_model(cls, name: str, model_class: Type) -> None:
        """
        Register a custom model type.

        Args:
            name: Name for the model type
            model_class: Model class (must have sklearn-compatible interface)
        """
        cls.MODEL_REGISTRY[name.lower()] = model_class

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls.MODEL_REGISTRY.keys())


class ModelConfigBuilder:
    """
    Builder class for creating model configurations.

    Helps create properly formatted configuration dictionaries
    for the model factory.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}

    def model_type(self, model_type: str) -> "ModelConfigBuilder":
        """Set model type."""
        self.config["model_type"] = model_type
        return self

    def model_path(self, path: Union[str, Path]) -> "ModelConfigBuilder":
        """Set model path."""
        self.config["model_path"] = str(path)
        return self

    def features(self, feature_names: List[str]) -> "ModelConfigBuilder":
        """Set feature names."""
        self.config["feature_names"] = feature_names
        return self

    def target(self, target_name: str) -> "ModelConfigBuilder":
        """Set target name."""
        self.config["target_name"] = target_name
        return self

    def scaler(self, scaler_path: Union[str, Path]) -> "ModelConfigBuilder":
        """Set scaler path."""
        self.config["scaler_path"] = str(scaler_path)
        return self

    def model_params(self, **params: Any) -> "ModelConfigBuilder":
        """Set model parameters."""
        self.config["model_params"] = params
        return self

    def dimensions(
        self, time_dim: str = "time", channel_dim: str = "channel"
    ) -> "ModelConfigBuilder":
        """Set dimension names."""
        self.config["time_dim"] = time_dim
        self.config["channel_dim"] = channel_dim
        return self

    def contribution_method(self, method: str) -> "ModelConfigBuilder":
        """Set contribution calculation method."""
        self.config["contribution_method"] = method
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return configuration dictionary."""
        return self.config.copy()

    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file."""
        path = Path(path)

        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function for quick model creation
def create_sklearn_model(
    model_type: Optional[str] = None, model_path: Optional[Union[str, Path]] = None, **kwargs: Any
) -> SklearnModelWrapper:
    """
    Convenience function to create sklearn model wrapper.

    Args:
        model_type: Type of model ('linear', 'random_forest', etc.)
        model_path: Path to saved model
        **kwargs: Additional arguments

    Returns:
        SklearnModelWrapper instance
    """
    return SklearnModelFactory.create(model_type=model_type, model_path=model_path, **kwargs)
