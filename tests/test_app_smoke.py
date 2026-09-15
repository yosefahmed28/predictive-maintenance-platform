"""
Simple smoke tests to confirm the middleware layer (loaders) works
correctly, whether using mock or real components.

Run:
    pytest tests/test_app_smoke.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pandas as pd
from loaders import (
    get_feature_columns,
    load_classical_model,
    load_clean_data,
    explain_prediction,
)


def test_feature_columns_not_empty():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_classical_model_predicts():
    model, _ = load_classical_model()
    df = load_clean_data()
    cols = [c for c in get_feature_columns() if c in df.columns]
    assert len(cols) > 0, "No feature columns match the data"

    proba = model.predict_proba(df[cols])
    assert proba.shape[0] == len(df)
    assert proba.shape[1] == 2
    assert ((proba >= 0) & (proba <= 1)).all()


def test_clean_data_has_machine_id():
    df = load_clean_data()
    assert "machine_id" in df.columns
    assert len(df) > 0


def test_explain_returns_expected_keys():
    model, _ = load_classical_model()
    df = load_clean_data()
    cols = [c for c in get_feature_columns() if c in df.columns]
    row = df[cols].iloc[[0]]

    result = explain_prediction(model, row)
    assert "top_features" in result
    assert isinstance(result["top_features"], list)
