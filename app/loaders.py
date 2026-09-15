"""
loaders.py
----------
Middleware layer between the dashboard and the rest of the team's
components (models, preprocessing, SHAP).

Idea: each function here first tries to load the real artifact from the
team (once received). If it isn't available yet, it returns a "mock"
(placeholder) version so you can keep working on and testing the
dashboard without waiting on anyone.

Once you receive the real files, update the paths under PATHS and
gradually remove the mock logic (or leave it as a safe fallback, which
is actually the better option).
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Expected paths (update these once you receive the real files from each Role)
# ---------------------------------------------------------------------------
PATHS = {
    "classical_model": "models/classical_model.pkl",
    "mlp_model": "models/mlp_model.h5",
    "autoencoder_model": "models/autoencoder.h5",
    "feature_columns": "models/feature_columns.json",
    "clean_data": "data/ml_feature_ready.csv",
}

# Default features (match the AI4I2020 dataset) — used for mock mode only
DEFAULT_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "type_L",
    "type_M",
    "type_H",
]


def get_feature_columns() -> list:
    """Returns feature names. Mock if the real file from Role 2 isn't available yet."""
    if os.path.exists(PATHS["feature_columns"]):
        with open(PATHS["feature_columns"], "r") as f:
            return json.load(f)
    return DEFAULT_FEATURES


def load_classical_model():
    """Loads the classical ML model (Role 3). Mock: a simple placeholder model."""
    if os.path.exists(PATHS["classical_model"]):
        return joblib.load(PATHS["classical_model"]), False  # (model, is_mock)
    return _MockClassifier(), True


def load_dl_models():
    """Loads the MLP and Autoencoder models (Role 4). Mock if not received yet."""
    mlp_ready = os.path.exists(PATHS["mlp_model"])
    ae_ready = os.path.exists(PATHS["autoencoder_model"])

    if mlp_ready and ae_ready:
        # Enable these two lines once you receive the real models (requires tensorflow/keras)
        # from tensorflow.keras.models import load_model
        # return load_model(PATHS["mlp_model"]), load_model(PATHS["autoencoder_model"]), False
        pass

    return _MockClassifier(), _MockAutoencoder(), True


def explain_prediction(model, input_row: pd.DataFrame) -> dict:
    """
    Calls the explain function from Role 5.
    Mock: returns the top 3 features in simple arbitrary order (display only).
    """
    try:
        from src.xai import explain  # the real function from Role 5, once available
        return explain(model, input_row)
    except ImportError:
        cols = list(input_row.columns)
        top = cols[: min(3, len(cols))]
        return {
            "top_features": [(c, float(input_row[c].iloc[0])) for c in top],
            "note": "\u26A0\uFE0F Real SHAP explanations not received yet — this is a mock preview.",
        }


def load_clean_data() -> pd.DataFrame:
    """Loads clean data (Role 1). Mock: generates small random data for display."""
    if os.path.exists(PATHS["clean_data"]):
        return pd.read_csv(PATHS["clean_data"])

    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "machine_id": [f"M-{i:03d}" for i in range(n)],
            "air_temperature": rng.normal(300, 2, n),
            "process_temperature": rng.normal(310, 2, n),
            "rotational_speed": rng.normal(1500, 100, n),
            "torque": rng.normal(40, 8, n),
            "tool_wear": rng.integers(0, 250, n),
            "machine_failure": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
        }
    )


# ---------------------------------------------------------------------------
# Simple mock components — used only until the real files are received
# ---------------------------------------------------------------------------
class _MockClassifier:
    """Mock model: returns a plausible failure probability based on tool_wear and torque."""

    def predict_proba(self, X: pd.DataFrame):
        wear = X.get("tool_wear", pd.Series([50] * len(X))).to_numpy()
        torque = X.get("torque", pd.Series([40] * len(X))).to_numpy()
        risk = np.clip((wear / 250) * 0.6 + (torque / 80) * 0.4, 0, 1)
        return np.column_stack([1 - risk, risk])

    def predict(self, X: pd.DataFrame):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


class _MockAutoencoder:
    """Mock autoencoder: returns a simple random reconstruction error."""

    def reconstruction_error(self, X: pd.DataFrame):
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, size=len(X))
