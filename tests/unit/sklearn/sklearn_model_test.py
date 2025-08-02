from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from atlas.factories.sklearn_factory import ModelConfigBuilder, SklearnModelFactory
from atlas.models.wrappers.sklearn_wrapper import SklearnModelWrapper


def make_model():
    # Generate synthetic marketing data
    np.random.seed(42)
    n_samples = 1000

    # Features: marketing spend by channel
    tv_spend = np.random.uniform(10_000, 100_000, n_samples)
    digital_spend = np.random.uniform(20_000, 150_000, n_samples)
    radio_spend = np.random.uniform(5_000, 50_000, n_samples)

    # Target: revenue with some non-linear relationships
    revenue = (
        0.8 * np.sqrt(tv_spend) * 100
        + 1.2 * np.log1p(digital_spend) * 1000
        + 0.5 * radio_spend
        + np.random.normal(0, 10_000, n_samples)
    )

    # Create DataFrame
    data = pd.DataFrame(
        {
            "tv_spend": tv_spend,
            "digital_spend": digital_spend,
            "radio_spend": radio_spend,
            "revenue": revenue,
        }
    )

    # Prepare features and target
    feature_names = ["tv_spend", "digital_spend", "radio_spend"]
    X = data[feature_names]
    y = data["revenue"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Model R² - Train: {train_score:.3f}, Test: {test_score:.3f}")

    return {"model": model, "scaler": scaler, "feature_names": feature_names}


def test_sklearn_model():
    # Create synthetic model
    model_data = make_model()

    # Wrap the model using SklearnModelWrapper
    wrapper = SklearnModelWrapper(
        model=model_data["model"],
        target_name="revenue",
        scaler=model_data["scaler"],
        feature_names=model_data["feature_names"],
    )

    # Create a sample input dataset
    sample_input = xr.Dataset(
        {
            "tv_spend": ("time", [50000, 60000]),
            "digital_spend": ("time", [80000, 90000]),
            "radio_spend": ("time", [20000, 25000]),
        }
    )

    # Generate predictions
    predictions = wrapper.predict(sample_input)

    # Check predictions shape
    assert predictions.shape == (2,)

    # Check contributions
    contributions = wrapper.contributions(sample_input)

    assert isinstance(contributions, xr.Dataset)
    assert contributions.sizes["time"] == 2

    # Check Prediction matches expected output
    expected_predictions = model_data["model"].predict(
        model_data["scaler"].transform(sample_input.to_dataframe().values)
    )
    np.testing.assert_allclose(predictions.values, expected_predictions, rtol=1e-5)


def test_sklearn_model_from_artifact():
    # Create synthetic model
    model_data = make_model()

    # Save the model and scaler
    dirname = Path(__file__).parent
    model_path = dirname / "test_model.joblib"
    scaler_path = dirname / "test_scaler.joblib"
    joblib.dump(model_data["model"], model_path)
    joblib.dump(model_data["scaler"], scaler_path)

    # Create SklearnModelWrapper from factory
    wrapper = SklearnModelFactory.create(
        model_type="RandomForestRegressor",
        model_path=model_path,
        scaler_path=scaler_path,
        feature_names=model_data["feature_names"],
        target_name="revenue",
    )

    # Generate predictions using the wrapper
    sample_input = xr.Dataset(
        {
            "tv_spend": ("samples", [50000, 60000]),
            "digital_spend": ("samples", [80000, 90000]),
            "radio_spend": ("samples", [20000, 25000]),
        }
    )

    predictions = wrapper.predict(sample_input)

    # Check predictions shape
    assert predictions.shape == (2,)

    # Clean up
    model_path.unlink()
    scaler_path.unlink()
