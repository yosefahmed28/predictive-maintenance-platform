"""
features.py
====================
Role 2 (EDA & Feature Engineering) deliverable — AI4I 2020 Predictive
Maintenance project.

Reproducible, leakage-safe pipeline that turns the raw ai4i2020.csv into
ML-ready train/validation/test arrays for Role 3 (Classical ML) and
Role 4 (Deep Learning).

Pipeline stages
----------------
Raw CSV
  -> Cleaning (no-op today; kept for pipeline generality / Role 1 handoff)
  -> Column selection (drop identifiers + post-failure leakage columns)
  -> Train / validation / test split  <-- split BEFORE any fitting
  -> Feature engineering (stateless transforms + a train-fitted transformer)
  -> Scaling / encoding (fit on train only)
  -> ML-ready numpy arrays

Rationale for every design decision is documented in
Role2_EDA_FeatureEngineering.ipynb (Sections 7-11). This script only
implements the pipeline; see the notebook for the analysis behind it.

Usage
-----
    from src.features import build_pipeline_splits

    splits = build_pipeline_splits('ai4i2020.csv')
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    feature_names = splits['feature_names']
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants — derived from the leakage analysis in the EDA notebook
# ---------------------------------------------------------------------------

TARGET_COL = "Machine failure"

# Identifiers (no reproducible predictive meaning) + post-failure outcome
# labels (leakage). See notebook Section 7 "Data Leakage Analysis".
RAW_DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

RAW_NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TOOL_WEAR_BINS = [-0.1, 50, 100, 150, 200, 253]
TOOL_WEAR_LABELS = ["very_low", "low", "medium", "high", "critical"]


# ---------------------------------------------------------------------------
# Stage 1: load + clean + select
# ---------------------------------------------------------------------------

def load_and_select(path: str) -> pd.DataFrame:
    """Load the raw CSV, apply (currently no-op) cleaning, and drop
    identifier / leakage columns. Keeps the target column.

    NOTE: as of this dataset, there are no missing values or duplicate
    rows (verified in the EDA notebook, Section 1). This function is
    still the designated place to add cleaning logic once Role 1's
    canonical cleaned dataset/schema is finalized, so that Role 3/4 only
    need to point this function at a new source.
    """
    df = pd.read_csv(path)

    # --- cleaning step (no-op today, kept for future/Role-1 integration) ---
    df = df.drop_duplicates()

    # --- column selection (identifiers + leakage columns removed) ---
    drop_cols = [c for c in RAW_DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


# ---------------------------------------------------------------------------
# Stage 2: stateless feature engineering (safe to apply to any split)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic feature engineering that requires no fitted
    statistics. Safe to call identically on train, validation, and test.

    See notebook Section 9 for the rationale behind each feature.
    """
    out = df.copy()

    out["temp_diff_K"] = out["Process temperature [K]"] - out["Air temperature [K]"]

    out["power_W"] = out["Torque [Nm]"] * (
        out["Rotational speed [rpm]"] * 2 * np.pi / 60
    )

    out["torque_x_toolwear"] = out["Torque [Nm]"] * out["Tool wear [min]"]

    out["speed_torque_ratio"] = out["Rotational speed [rpm]"] / out["Torque [Nm]"]

    out["tool_wear_bucket"] = pd.cut(
        out["Tool wear [min]"], bins=TOOL_WEAR_BINS, labels=TOOL_WEAR_LABELS
    )

    return out


class TypeGroupedZScorer(BaseEstimator, TransformerMixin):
    """Leakage-safe version of `air_temp_z_within_type`.

    Fits per-`Type` mean/std of `value_col` on the TRAINING split only
    (via `.fit`), then applies those *fixed* statistics to any split via
    `.transform` — including validation/test, which never influence the
    fitted statistics. This mirrors any other fitted preprocessing step
    (e.g. StandardScaler) and must be fit only once, on the training data.
    """

    def __init__(self, value_col="Air temperature [K]", group_col="Type"):
        self.value_col = value_col
        self.group_col = group_col

    def fit(self, X, y=None):
        stats_df = X.groupby(self.group_col)[self.value_col].agg(["mean", "std"])
        self.group_stats_ = stats_df.to_dict("index")
        self.global_mean_ = X[self.value_col].mean()
        self.global_std_ = X[self.value_col].std()
        return self

    def transform(self, X):
        X = X.copy()

        def z(row):
            g = self.group_stats_.get(row[self.group_col])
            if g is None or g["std"] == 0 or pd.isna(g["std"]):
                return (row[self.value_col] - self.global_mean_) / self.global_std_
            return (row[self.value_col] - g["mean"]) / g["std"]

        X["air_temp_z_within_type"] = X.apply(z, axis=1)
        return X


# ---------------------------------------------------------------------------
# Stage 3: end-to-end reproducible split + fit + transform
# ---------------------------------------------------------------------------

def build_pipeline_splits(
    csv_path: str,
    test_size: float = 0.30,
    val_fraction_of_temp: float = 0.50,
    random_state: int = 42,
):
    """Runs the full pipeline: load -> select -> split -> engineer ->
    fit-on-train-only scale/encode -> return ML-ready arrays.

    Returns
    -------
    dict with keys:
        'train': (X_train_ready, y_train)
        'val':   (X_val_ready, y_val)
        'test':  (X_test_ready, y_test)
        'feature_names': list[str]  (column order of the *_ready arrays)
        'preprocessor': fitted sklearn ColumnTransformer (for inference reuse)
        'zscorer': fitted TypeGroupedZScorer (for inference reuse)
    """
    df = load_and_select(csv_path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Split BEFORE any feature-engineering statistics are computed —
    # this is the critical leakage-prevention step (notebook Section 11).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_temp,
        stratify=y_temp,
        random_state=random_state,
    )

    # Stateless feature engineering — identical function applied to each split
    X_train_fe = engineer_features(X_train)
    X_val_fe = engineer_features(X_val)
    X_test_fe = engineer_features(X_test)

    # Fitted (train-only) grouped z-score feature
    zscorer = TypeGroupedZScorer().fit(X_train_fe)
    X_train_fe = zscorer.transform(X_train_fe)
    X_val_fe = zscorer.transform(X_val_fe)
    X_test_fe = zscorer.transform(X_test_fe)

    numeric_features = RAW_NUMERIC_COLS + [
        "temp_diff_K",
        "power_W",
        "torque_x_toolwear",
        "speed_torque_ratio",
        "air_temp_z_within_type",
    ]
    categorical_onehot = ["Type"]
    categorical_ordinal = ["tool_wear_bucket"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat_onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_onehot,
            ),
            (
                "cat_ordinal",
                OrdinalEncoder(
                    categories=[TOOL_WEAR_LABELS],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_ordinal,
            ),
        ]
    )

    # Fit ONLY on train; transform (never re-fit) val/test.
    X_train_ready = preprocessor.fit_transform(X_train_fe)
    X_val_ready = preprocessor.transform(X_val_fe)
    X_test_ready = preprocessor.transform(X_test_fe)

    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "train": (X_train_ready, y_train.to_numpy()),
        "val": (X_val_ready, y_val.to_numpy()),
        "test": (X_test_ready, y_test.to_numpy()),
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "zscorer": zscorer,
    }


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/ai4i2020_raw.csv"
    splits = build_pipeline_splits(csv_path)

    for name in ["train", "val", "test"]:
        X_part, y_part = splits[name]
        print(f"{name}: X={X_part.shape}, y={y_part.shape}, "
              f"failure_rate={y_part.mean():.4f}")

    print("\nFinal feature columns:")
    for f in splits["feature_names"]:
        print(" -", f)
