"""
loaders.py
----------
Middleware layer between the Streamlit dashboard (app.py) and the
trained artifacts received from the rest of the team:

    Role 1/2 -> data/ml_feature_ready.csv          (clean, feature-ready data)
    Role 2   -> src/features.py                    (engineer_features, TypeGroupedZScorer)
    Role 3   -> models/random_forest_tuned_final.pkl, models/preprocessor.pkl,
                models/zscorer.pkl, src/model_utils.py (predict_new_sample)
    Role 4   -> models/tabular_mlp_tuned.keras,
                models/autoencoder_anomaly_detector_tuned.keras
    Role 5   -> src/xai_utils.py (SHAP feature alignment)

All real artifacts have now been received (see Handoff_to_DL_MLOPS.md at
the repo root). Every loader still falls back to a small mock if a
specific file is genuinely missing, so the app degrades gracefully
instead of crashing outright if one artifact is briefly out of sync.

IMPORTANT — decision threshold: Handoff_to_DL_MLOPS.md Section 2 states
the 0.5 classical-model threshold is a **placeholder**, not final. The
real deployment threshold should be set jointly with Role 5 once the
cost-based alerting simulation has real cost estimates. Don't treat
CLASSICAL_THRESHOLD below as settled — it's exposed as a sidebar slider
in app.py precisely so this can be explored, not hard-coded.
"""

import os
import numpy as np
import pandas as pd
import joblib

from src.features import engineer_features

# ---------------------------------------------------------------------------
# Paths (relative to repo root — Streamlit is run from there)
# ---------------------------------------------------------------------------
PATHS = {
    "classical_model": "models/random_forest_tuned_final.pkl",
    "preprocessor": "models/preprocessor.pkl",
    "zscorer": "models/zscorer.pkl",
    "mlp_model": "models/tabular_mlp_tuned.keras",
    "autoencoder_model": "models/autoencoder_anomaly_detector_tuned.keras",
    "clean_data": "data/ml_feature_ready.csv",
}

# Raw input columns predict_new_sample() / engineer_features() expect.
RAW_INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Autoencoder anomaly threshold tuned by Role 4
# (see autoencoder_tuned_dashboard.png — weighted reconstruction MSE).
AUTOENCODER_THRESHOLD = 0.6541

# Placeholder only — see module docstring above.
DEFAULT_CLASSICAL_THRESHOLD = 0.5


def get_raw_input_columns() -> list:
    """Raw columns needed to score one machine (manual form or CSV upload)."""
    return RAW_INPUT_COLUMNS


# ---------------------------------------------------------------------------
# Model / artifact loading
# ---------------------------------------------------------------------------
def load_classical_model():
    """Loads the tuned Random Forest (Role 3). Returns (model, is_mock)."""
    if os.path.exists(PATHS["classical_model"]):
        return joblib.load(PATHS["classical_model"]), False
    return _MockClassifier(), True


def load_preprocessing():
    """Loads the exact fitted preprocessor + zscorer from training time
    (Role 2/3). These must never be refit here — only .transform()."""
    preprocessor = joblib.load(PATHS["preprocessor"]) if os.path.exists(PATHS["preprocessor"]) else None
    zscorer = joblib.load(PATHS["zscorer"]) if os.path.exists(PATHS["zscorer"]) else None
    return preprocessor, zscorer


def load_dl_models():
    """Loads the MLP classifier and Autoencoder (Role 4).
    Returns (mlp_model, autoencoder_model, is_mock)."""
    mlp_ready = os.path.exists(PATHS["mlp_model"])
    ae_ready = os.path.exists(PATHS["autoencoder_model"])

    if mlp_ready and ae_ready:
        from tensorflow.keras.models import load_model
        return load_model(PATHS["mlp_model"]), load_model(PATHS["autoencoder_model"]), False

    return _MockClassifier(), _MockAutoencoder(), True


def load_clean_data() -> pd.DataFrame:
    """Loads Role 1/2's clean, feature-ready dataset for the overview tab."""
    if os.path.exists(PATHS["clean_data"]):
        return pd.read_csv(PATHS["clean_data"])

    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "machine_id": [f"M-{i:03d}" for i in range(n)],
        "air_temperature_k": rng.normal(300, 2, n),
        "process_temperature_k": rng.normal(310, 2, n),
        "rotational_speed_rpm": rng.normal(1500, 100, n),
        "torque_nm": rng.normal(40, 8, n),
        "tool_wear_min": rng.integers(0, 250, n),
        "machine_failure": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
    })


def clean_df_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the clean/feature-ready dataframe (data/ml_feature_ready.csv
    column names) back into the RAW column names/units predict_new_sample()
    expects, so the dashboard can score the whole clean dataset without
    asking anyone to re-enter raw sensor values by hand.
    """
    type_col = np.select(
        [df.get("type_M", False) == True, df.get("type_L", False) == True],
        ["M", "L"],
        default="H",
    )
    return pd.DataFrame({
        "Type": type_col,
        "Air temperature [K]": df["air_temperature_k"],
        "Process temperature [K]": df["process_temperature_k"],
        "Rotational speed [rpm]": df["rotational_speed_rpm"],
        "Torque [Nm]": df["torque_nm"],
        "Tool wear [min]": df["tool_wear_min"],
    })


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_raw_dataframe(raw_df: pd.DataFrame, threshold: float = DEFAULT_CLASSICAL_THRESHOLD) -> pd.DataFrame:
    """
    Scores a RAW dataframe (original AI4I column names) with the tuned
    Random Forest, reusing Role 3's predict_new_sample() so the exact
    same feature engineering + scaling is applied as at training time.
    """
    model, is_mock = load_classical_model()
    preprocessor, zscorer = load_preprocessing()

    if is_mock or preprocessor is None or zscorer is None:
        proba = _MockClassifier().predict_proba(raw_df)[:, 1]
        return pd.DataFrame({
            "failure_probability": proba,
            "predicted_failure": (proba >= threshold).astype(int),
        })

    from src.model_utils import predict_new_sample
    return predict_new_sample(model, preprocessor, zscorer, raw_df, threshold=threshold)


def score_dl(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the MLP classifier and the Autoencoder anomaly detector (Role 4)
    on raw input, reusing the same fitted preprocessor/zscorer.
    Returns mlp_probability, anomaly_score, is_anomaly per row.
    """
    mlp, ae, is_mock = load_dl_models()
    preprocessor, zscorer = load_preprocessing()

    if is_mock or preprocessor is None or zscorer is None:
        n = len(raw_df)
        return pd.DataFrame({
            "mlp_probability": _MockClassifier().predict_proba(raw_df)[:, 1],
            "anomaly_score": _MockAutoencoder().reconstruction_error(raw_df),
            "is_anomaly": [False] * n,
        })

    df_fe = engineer_features(raw_df)
    df_fe = zscorer.transform(df_fe)
    X_ready = preprocessor.transform(df_fe)

    mlp_proba = np.asarray(mlp.predict(X_ready, verbose=0)).ravel()
    reconstructed = np.asarray(ae.predict(X_ready, verbose=0))
    recon_error = np.mean((X_ready - reconstructed) ** 2, axis=1)

    return pd.DataFrame({
        "mlp_probability": mlp_proba,
        "anomaly_score": recon_error,
        "is_anomaly": recon_error >= AUTOENCODER_THRESHOLD,
    })


def explain_prediction(input_row_raw: pd.DataFrame, top_n: int = 5) -> dict:
    """
    SHAP-based local explanation for one raw input row, reusing Role 5's
    feature-alignment helper (src/xai_utils.prepare_features) so column
    names/order match what the model was trained on.
    """
    try:
        import shap
        from src.xai_utils import load_expected_features, prepare_features

        model, is_mock = load_classical_model()
        if is_mock:
            raise RuntimeError("classical model file not found")

        expected_features = load_expected_features()
        if not expected_features:
            raise RuntimeError("reports/classical_ml_final_model_metadata.json not found")

        zscorer = load_preprocessing()[1]
        df_fe = engineer_features(input_row_raw)
        if zscorer is not None:
            df_fe = zscorer.transform(df_fe)

        X_named = prepare_features(df_fe, expected_features)

        # WORKAROUND: Role 5's prepare_features() renames the tool-wear-bucket
        # column but leaves its values as text labels ("very_low", "low", ...)
        # instead of the integers the OrdinalEncoder actually produced at
        # training time. Re-apply that same encoding here so SHAP sees the
        # same numeric input the model was trained on. Flag this to Role 5 —
        # src/xai_utils.py's prepare_features() should really do this itself,
        # since generate_shap_plots() has the same bug.
        ord_col = "cat_ordinal__tool_wear_bucket"
        if ord_col in X_named.columns and X_named[ord_col].dtype == object:
            from src.features import TOOL_WEAR_LABELS
            label_to_code = {label: i for i, label in enumerate(TOOL_WEAR_LABELS)}
            X_named[ord_col] = X_named[ord_col].map(label_to_code).fillna(-1).astype(int)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_named.values, check_additivity=False)
        vals = (
            shap_vals[1] if isinstance(shap_vals, list)
            else shap_vals[:, :, 1] if len(shap_vals.shape) == 3
            else shap_vals
        )
        row_vals = vals[0]
        clean_names = [
            c.replace("num__", "").replace("cat_onehot__", "").replace("cat_ordinal__", "")
            for c in X_named.columns
        ]
        ranked = sorted(zip(clean_names, row_vals), key=lambda t: abs(t[1]), reverse=True)[:top_n]
        return {
            "top_features": [(name, float(val)) for name, val in ranked],
            "note": "SHAP values for this prediction — positive pushes toward failure, negative toward normal.",
        }
    except Exception as exc:
        cols = list(input_row_raw.columns)
        top = cols[: min(top_n, len(cols))]
        preview = []
        for c in top:
            val = input_row_raw[c].iloc[0]
            preview.append((c, float(val)) if isinstance(val, (int, float, np.integer, np.floating)) else (c, 0.0))
        return {
            "top_features": preview,
            "note": f"\u26A0\uFE0F Real SHAP explanation unavailable right now ({exc}) — showing a raw-value preview instead.",
        }


# ---------------------------------------------------------------------------
# Mock fallbacks — safety net only, used if an artifact goes missing
# ---------------------------------------------------------------------------
class _MockClassifier:
    def predict_proba(self, X: pd.DataFrame):
        wear = X.get("Tool wear [min]", X.get("tool_wear_min", pd.Series([50] * len(X)))).to_numpy()
        torque = X.get("Torque [Nm]", X.get("torque_nm", pd.Series([40] * len(X)))).to_numpy()
        risk = np.clip((wear / 250) * 0.6 + (torque / 80) * 0.4, 0, 1)
        return np.column_stack([1 - risk, risk])

    def predict(self, X, verbose=0):
        return self.predict_proba(X)[:, 1].reshape(-1, 1)


class _MockAutoencoder:
    def reconstruction_error(self, X: pd.DataFrame):
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, size=len(X))

    def predict(self, X, verbose=0):
        return X
