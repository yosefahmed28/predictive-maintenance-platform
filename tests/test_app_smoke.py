"""
Smoke tests for the middleware layer (app/loaders.py), whether it ends
up using real artifacts or falls back to mocks.

Run:
    pytest tests/test_app_smoke.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pandas as pd
from loaders import (
    get_raw_input_columns,
    load_classical_model,
    load_clean_data,
    score_raw_dataframe,
    explain_prediction,
)


def _sample_raw_row():
    return pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }])


def test_raw_input_columns_not_empty():
    cols = get_raw_input_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_classical_model_loads():
    model, is_mock = load_classical_model()
    assert model is not None
    assert isinstance(is_mock, bool)


def test_clean_data_loads_with_expected_columns():
    df = load_clean_data()
    assert len(df) > 0
    assert "machine_id" in df.columns


def test_score_raw_dataframe_returns_probability():
    result = score_raw_dataframe(_sample_raw_row())
    assert "failure_probability" in result.columns
    assert "predicted_failure" in result.columns
    assert 0.0 <= result["failure_probability"].iloc[0] <= 1.0


def test_explain_prediction_returns_top_features():
    explanation = explain_prediction(_sample_raw_row())
    assert "top_features" in explanation
    assert len(explanation["top_features"]) > 0
