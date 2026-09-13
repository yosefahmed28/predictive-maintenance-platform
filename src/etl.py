"""
ETL Pipeline: AI4I 2020 Predictive Maintenance Dataset
Role: Data Engineer & SQL Database Lead

Extract  -> read raw CSV
Transform -> clean, validate, normalize into star-ish schema tables,
             melt wide failure-flag columns into long format,
             synthesize a maintenance-event log
Load     -> bulk insert into MySQL (predictive_maintenance database)

Also writes a cleaned flat CSV and a feature-ready numpy/CSV array for
the downstream ML track (per project deliverable: "prepare feature-ready
data arrays for downstream ML models").
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text

# Works whether this file is run as a standalone script (python scripts/etl.py)
# or its source is pasted into a notebook cell (no __file__ available there).
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

RAW_PATH = BASE_DIR / "data" / "ai4i2020_raw.csv"
CLEAN_PATH = BASE_DIR / "data" / "ai4i2020_clean.csv"
FEATURE_CSV_PATH = BASE_DIR / "data" / "ml_feature_ready.csv"
FEATURE_NPY_PATH = BASE_DIR / "data" / "ml_feature_ready.npy"

# --- Database connection ---
# Edit these if your local MySQL setup differs from sql/00_local_setup.sql
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "pm_user"
DB_PASSWORD = "pm_pass123"
DB_NAME = "predictive_maintenance"
DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

FAILURE_MODE_DESCRIPTIONS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure (unpredictable, not related to process parameters)",
    "UNSPECIFIED": "Machine failure flagged in source data without an attributed specific failure mode",
}

FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# ---------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------
def extract(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------
def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop exact duplicate rows, if any
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} duplicate rows")

    # 2. Standardize column names to snake_case
    df = df.rename(columns={
        "UDI": "machine_id",
        "Product ID": "product_id",
        "Type": "type_code",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
    })

    # 3. Type validation / range checks (documented assumptions from the
    #    UCI AI4I2020 data description: physically plausible ranges)
    numeric_cols = [
        "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    assert df["machine_id"].is_unique, "machine_id (UDI) must be unique"
    assert df["product_id"].is_unique, "product_id must be unique"
    assert df["type_code"].isin(["L", "M", "H"]).all(), "Unexpected type_code value"
    assert (df["air_temperature_k"] > 0).all()
    assert (df["process_temperature_k"] > 0).all()
    assert (df["rotational_speed_rpm"] >= 0).all()
    assert (df["torque_nm"] >= 0).all()
    assert (df["tool_wear_min"] >= 0).all()
    for c in FAILURE_MODE_COLS + ["machine_failure"]:
        assert df[c].isin([0, 1]).all(), f"{c} must be binary"

    # 4. Cross-check machine_failure against the 5 specific failure-mode
    #    flags. Known dataset quirk: a handful of rows disagree. We do
    #    NOT silently overwrite these - we flag them for the data
    #    dictionary / QA log and preserve ground truth as-is.
    any_mode = df[FAILURE_MODE_COLS].max(axis=1)
    mismatch = df[any_mode != df["machine_failure"]]
    if len(mismatch):
        print(f"NOTE: {len(mismatch)} rows have machine_failure "
              f"inconsistent with individual failure-mode flags "
              f"(known AI4I2020 labeling quirk - preserved, not corrected). "
              f"machine_ids: {mismatch['machine_id'].tolist()}")

    return df


def build_dim_tables(df: pd.DataFrame):
    """Build machine_type, machines, sensor_readings, failure_mode dims/facts."""
    machine_type = pd.DataFrame({
        "type_code": ["L", "M", "H"],
        "type_name": ["Low", "Medium", "High"],
        "description": [
            "Low quality variant product line",
            "Medium quality variant product line",
            "High quality variant product line",
        ],
    })

    machines = df[["machine_id", "product_id", "type_code"]].drop_duplicates()

    sensor_readings = df[[
        "machine_id", "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    ]].copy()
    sensor_readings.insert(0, "reading_id", sensor_readings["machine_id"])  # 1:1 in this snapshot

    failure_mode = pd.DataFrame({
        "failure_mode_code": list(FAILURE_MODE_DESCRIPTIONS.keys()),
        "description": list(FAILURE_MODE_DESCRIPTIONS.values()),
    })

    return machine_type, machines, sensor_readings, failure_mode


def build_failure_events(df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide binary failure-flag columns into a long-format fact table."""
    records = []
    for _, row in df.iterrows():
        modes_flagged = [m for m in FAILURE_MODE_COLS if row[m] == 1]
        if row["machine_failure"] == 1 and not modes_flagged:
            # overall failure with no specific mode attributed in source data
            records.append({
                "reading_id": row["machine_id"],
                "machine_id": row["machine_id"],
                "failure_mode_code": "UNSPECIFIED",
                "is_overall_failure": 1,
            })
        for m in modes_flagged:
            records.append({
                "reading_id": row["machine_id"],
                "machine_id": row["machine_id"],
                "failure_mode_code": m,
                "is_overall_failure": int(row["machine_failure"]),
            })
    return pd.DataFrame.from_records(
        records,
        columns=["reading_id", "machine_id", "failure_mode_code", "is_overall_failure"],
    )


def build_maintenance_events(failure_events: pd.DataFrame, sensor_readings: pd.DataFrame) -> pd.DataFrame:
    """
    SYNTHESIZE a corrective-maintenance log: one record per failure_event,
    simulating a CMMS entry logged when the failure was detected.
    This is clearly synthetic (is_synthetic = 1) - see schema comments.
    """
    wear_lookup = sensor_readings.set_index("reading_id")["tool_wear_min"]
    records = []
    # Use the future event_id values (1..N, matches auto-increment insert order)
    for i, row in failure_events.reset_index(drop=True).iterrows():
        records.append({
            "machine_id": row["machine_id"],
            "event_id": i + 1,  # aligns with AUTO_INCREMENT order on insert
            "maintenance_type": "corrective",
            "tool_wear_at_event": int(wear_lookup.get(row["reading_id"], 0)),
            "is_synthetic": 1,
            "notes": f"Simulated corrective maintenance for failure mode {row['failure_mode_code']}",
        })
    return pd.DataFrame(records)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature-ready table for downstream ML models. Kept leakage-safe:
    no failure-mode columns other than the target are included, and no
    post-failure information is used (all features derive from process
    variables measured at/before the observation).
    """
    feat = df.copy()
    feat["temp_diff_k"] = feat["process_temperature_k"] - feat["air_temperature_k"]
    feat["mechanical_power_w"] = feat["torque_nm"] * feat["rotational_speed_rpm"] * (2 * np.pi / 60)
    feat["tool_wear_bucket"] = pd.qcut(feat["tool_wear_min"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    feat = pd.get_dummies(feat, columns=["type_code"], prefix="type")

    feature_cols = [
        "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
        "torque_nm", "tool_wear_min", "temp_diff_k", "mechanical_power_w",
    ] + [c for c in feat.columns if c.startswith("type_")]

    out = feat[["machine_id"] + feature_cols + ["machine_failure"]].copy()
    return out


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------
def load_to_mysql(machine_type, machines, sensor_readings, failure_mode,
                   failure_events, maintenance_events, engine):
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("TRUNCATE TABLE maintenance_events"))
        conn.execute(text("TRUNCATE TABLE failure_events"))
        conn.execute(text("TRUNCATE TABLE sensor_readings"))
        conn.execute(text("TRUNCATE TABLE machines"))
        conn.execute(text("TRUNCATE TABLE failure_mode"))
        conn.execute(text("TRUNCATE TABLE machine_type"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

    machine_type.to_sql("machine_type", engine, if_exists="append", index=False)
    machines.to_sql("machines", engine, if_exists="append", index=False)
    sensor_readings.to_sql("sensor_readings", engine, if_exists="append", index=False)
    failure_mode.to_sql("failure_mode", engine, if_exists="append", index=False)
    failure_events[["reading_id", "machine_id", "failure_mode_code", "is_overall_failure"]].to_sql(
        "failure_events", engine, if_exists="append", index=False
    )
    maintenance_events.to_sql("maintenance_events", engine, if_exists="append", index=False)


def run_pipeline():
    print("EXTRACT: reading raw CSV...")
    raw = extract()

    print("TRANSFORM: cleaning + validating...")
    clean = clean_and_rename(raw)
    clean.to_csv(CLEAN_PATH, index=False)

    print("TRANSFORM: building dimension/fact tables...")
    machine_type, machines, sensor_readings, failure_mode = build_dim_tables(clean)
    failure_events = build_failure_events(clean)
    maintenance_events = build_maintenance_events(failure_events, sensor_readings)

    print("TRANSFORM: engineering ML feature table...")
    features = engineer_features(clean)
    features.to_csv(FEATURE_CSV_PATH, index=False)
    np.save(FEATURE_NPY_PATH, features.drop(columns=["machine_id"]).to_numpy(dtype=float))

    print("LOAD: writing to MySQL (predictive_maintenance)...")
    engine = create_engine(DB_URI)
    load_to_mysql(machine_type, machines, sensor_readings, failure_mode,
                  failure_events, maintenance_events, engine)

    print("ETL COMPLETE.")
    print(f"  machines: {len(machines)} rows")
    print(f"  sensor_readings: {len(sensor_readings)} rows")
    print(f"  failure_events: {len(failure_events)} rows")
    print(f"  maintenance_events: {len(maintenance_events)} rows")
    print(f"  feature table: {features.shape}")
    return engine


if __name__ == "__main__":
    run_pipeline()
