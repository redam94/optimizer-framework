"""
Scikit-learn Model Wrapper for Atlas Optimization Framework

This module provides a wrapper to integrate scikit-learn regression models
with the Atlas optimization framework.
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr

# Assuming these are from the Atlas framework based on the docs
from atlas.core.interfaces import AbstractModel


class SklearnModelWrapper(AbstractModel):
    """
    Wrapper for scikit-learn regression models to work with Atlas optimization.

    This wrapper handles:
    - Loading sklearn models from files
    - Converting between xarray and numpy formats
    - Providing predictions and feature contributions
    - Supporting various sklearn regression models
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[Union[str, Path]] = None,
        feature_names: Optional[List[str]] = None,
        target_name: str = "target",
        model_type: str = "sklearn_regression",
        scaler: Optional[Any] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        feature_order: Optional[List[str]] = None,
        time_dim: str = "time",
        channel_dim: str = "channel",
        contribution_method: str = "auto",
    ):
        """
        Initialize the sklearn model wrapper.

        Args:
            model: Pre-loaded sklearn model instance
            model_path: Path to saved model file (.pkl or .joblib)
            feature_names: List of feature names the model expects
            target_name: Name of the target variable
            model_type: Type identifier for the model
            scaler: Pre-loaded scaler instance (e.g., StandardScaler)
            scaler_path: Path to saved scaler file
            feature_order: Specific order of features (if different from feature_names)
            time_dim: Name of time dimension in xarray datasets
            channel_dim: Name of channel dimension in xarray datasets
            contribution_method: Method for calculating contributions
                               ('auto', 'coef', 'feature_importance', 'shap')
        """
        super().__init__()

        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Either 'model' or 'model_path' must be provided")

        # Load scaler if provided
        if scaler is not None:
            self.scaler = scaler
        elif scaler_path is not None:
            self.scaler = self._load_model(scaler_path)
        else:
            self.scaler = None

        # Set attributes
        self.feature_names = feature_names or self._infer_feature_names()
        self.feature_order = feature_order or self.feature_names
        self.target_name = target_name
        self._model_type = model_type
        self.time_dim = time_dim
        self.channel_dim = channel_dim
        self.contribution_method = contribution_method

        # Validate model
        self._validate_model()

    @property
    def model_type(self) -> str:
        """Return the type identifier for this model."""
        return self._model_type

    @property
    def required_dimensions(self) -> List[str]:
        """Return list of required dimensions for input data."""
        return [self.time_dim, self.channel_dim] if self.channel_dim else [self.time_dim]

    def _load_model(self, path: Union[str, Path]) -> Any:
        """Load model from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            if path.suffix == ".joblib":
                return joblib.load(path)
            else:  # Assume pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {str(e)}")

    def _infer_feature_names(self) -> List[str]:
        """Try to infer feature names from the model."""
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, "n_features_in_"):
            return [f"feature_{i}" for i in range(self.model.n_features_in_)]
        else:
            warnings.warn("Could not infer feature names from model. Using default names.")
            return []

    def _validate_model(self) -> None:
        """Validate that the model has required methods."""
        if not hasattr(self.model, "predict"):
            raise ValueError("Model must have a 'predict' method")

        # Check if it's a regression model
        if hasattr(self.model, "_estimator_type") and self.model._estimator_type != "regressor":
            warnings.warn(f"Model type is '{self.model._estimator_type}', expected 'regressor'")

    def _xarray_to_features(self, x: xr.Dataset) -> np.ndarray:
        """
        Convert xarray Dataset to numpy array of features.

        Args:
            x: Input xarray Dataset

        Returns:
            Feature array with shape (n_samples, n_features)
        """
        # Extract features in the correct order
        feature_arrays = []

        for feature in self.feature_order:
            if feature in x.data_vars:
                # Get the data and flatten if needed
                data = x[feature].values
                if data.ndim > 1:
                    data = data.flatten()
                feature_arrays.append(data)
            elif feature in x.coords:
                # Handle coordinate features
                data = x.coords[feature].values
                if data.ndim > 1:
                    data = data.flatten()
                feature_arrays.append(data)
            else:
                raise KeyError(f"Feature '{feature}' not found in dataset")

        # Stack features
        if feature_arrays:
            features = np.column_stack(feature_arrays)
        else:
            # If no specific features, try to use all data variables
            features = np.column_stack([x[var].values.flatten() for var in x.data_vars])

        return features

    def predict(self, x: xr.Dataset) -> xr.DataArray:
        """
        Generate predictions from input data.

        Args:
            x: Input xarray Dataset containing features

        Returns:
            xarray DataArray with predictions
        """
        # Convert to features
        features = self._xarray_to_features(x)

        # Apply scaling if available
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Make predictions
        predictions = self.model.predict(features)

        # Convert back to xarray
        # Determine output dimensions based on input
        dims = []
        coords = {}

        # Preserve dimensions from input
        if self.time_dim in x.dims:
            dims.append(self.time_dim)
            coords[self.time_dim] = x.coords[self.time_dim]

        if self.channel_dim in x.dims and len(predictions) == len(x[self.channel_dim]):
            dims.append(self.channel_dim)
            coords[self.channel_dim] = x.coords[self.channel_dim]

        # Create output DataArray
        if dims:
            # Reshape predictions to match dimensions
            shape = [len(coords[dim]) for dim in dims]
            predictions = predictions.reshape(shape)

            result = xr.DataArray(predictions, dims=dims, coords=coords, name=self.target_name)
        else:
            # Single prediction
            result = xr.DataArray(predictions, name=self.target_name)

        return result

    def contributions(self, x: xr.Dataset) -> xr.Dataset:
        """
        Calculate feature contributions to predictions.

        Args:
            x: Input xarray Dataset

        Returns:
            xarray Dataset with contribution values for each feature
        """
        method = self._determine_contribution_method()

        if method == "coef":
            return self._linear_contributions(x)
        elif method == "feature_importance":
            return self._tree_contributions(x)
        elif method == "permutation":
            return self._permutation_contributions(x)
        else:
            # Default: return equal contributions
            return self._equal_contributions(x)

    def _determine_contribution_method(self) -> str:
        """Determine the best method for calculating contributions."""
        if self.contribution_method != "auto":
            return self.contribution_method

        # Auto-detect based on model type
        model_class = self.model.__class__.__name__

        if hasattr(self.model, "coef_"):
            return "coef"
        elif hasattr(self.model, "feature_importances_"):
            return "feature_importance"
        else:
            return "equal"

    def _linear_contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Calculate contributions for linear models using coefficients."""
        features = self._xarray_to_features(x)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Get coefficients
        coef = self.model.coef_
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        # Calculate contributions
        contributions = features * coef

        # Add intercept contribution if available
        if hasattr(self.model, "intercept_"):
            intercept_contrib = np.full((features.shape[0], 1), self.model.intercept_)
            contributions = np.hstack([contributions, intercept_contrib])
            feature_names = self.feature_order + ["intercept"]
        else:
            feature_names = self.feature_order

        # Convert to xarray Dataset
        result = xr.Dataset()
        for i, feature in enumerate(feature_names):
            result[f"{feature}_contribution"] = xr.DataArray(
                contributions[:, i],
                dims=[self.time_dim] if self.time_dim in x.dims else ["sample"],
                name=f"{feature}_contribution",
            )

        return result

    def _tree_contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Calculate contributions for tree-based models using feature importance."""
        features = self._xarray_to_features(x)
        predictions = self.predict(x).values.flatten()

        # Get feature importances
        importances = self.model.feature_importances_

        # Normalize importances
        importances = importances / importances.sum()

        # Distribute prediction value according to importance
        contributions = predictions[:, np.newaxis] * importances[np.newaxis, :]

        # Convert to xarray Dataset
        result = xr.Dataset()
        for i, feature in enumerate(self.feature_order):
            result[f"{feature}_contribution"] = xr.DataArray(
                contributions[:, i],
                dims=[self.time_dim] if self.time_dim in x.dims else ["sample"],
                name=f"{feature}_contribution",
            )

        return result

    def _permutation_contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Calculate contributions using permutation importance."""
        # This is a simplified version - you might want to use
        # sklearn.inspection.permutation_importance for better results
        features = self._xarray_to_features(x)
        base_pred = self.predict(x).values.flatten()

        contributions = {}

        for i, feature in enumerate(self.feature_order):
            # Permute feature
            features_permuted = features.copy()
            np.random.shuffle(features_permuted[:, i])

            # Predict with permuted feature
            if self.scaler is not None:
                features_permuted_scaled = self.scaler.transform(features_permuted)
                perm_pred = self.model.predict(features_permuted_scaled)
            else:
                perm_pred = self.model.predict(features_permuted)

            # Contribution is the difference
            contributions[f"{feature}_contribution"] = base_pred - perm_pred

        # Convert to xarray Dataset
        result = xr.Dataset()
        for feature, contrib in contributions.items():
            result[feature] = xr.DataArray(
                contrib,
                dims=[self.time_dim] if self.time_dim in x.dims else ["sample"],
                name=feature,
            )

        return result

    def _equal_contributions(self, x: xr.Dataset) -> xr.Dataset:
        """Distribute predictions equally among features."""
        predictions = self.predict(x).values.flatten()
        n_features = len(self.feature_order)

        # Equal contribution per feature
        contrib_per_feature = predictions / n_features

        # Convert to xarray Dataset
        result = xr.Dataset()
        for feature in self.feature_order:
            result[f"{feature}_contribution"] = xr.DataArray(
                contrib_per_feature,
                dims=[self.time_dim] if self.time_dim in x.dims else ["sample"],
                name=f"{feature}_contribution",
            )

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores if available."""
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_order, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            # Use absolute coefficients as importance
            coef = np.abs(self.model.coef_)
            if coef.ndim > 1:
                coef = coef.mean(axis=0)
            return dict(zip(self.feature_order, coef))
        else:
            # Return equal importance
            n_features = len(self.feature_order)
            return {f: 1.0 / n_features for f in self.feature_order}

    def validate_input(self, x: xr.Dataset) -> bool:
        """Validate that input dataset has required features."""
        missing_features = []
        for feature in self.feature_order:
            if feature not in x.data_vars:
                missing_features.append(feature)

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        return True
