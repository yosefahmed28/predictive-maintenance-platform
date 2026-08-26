import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_expected_features():
    metadata_path = "reports/classical_ml_final_model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            return meta.get("feature_names", [])
    return []

def prepare_features(df, expected_features):
    X = df.copy()
    num_mappings = {
        "Air temperature [K]": "num__Air temperature [K]",
        "Process temperature [K]": "num__Process temperature [K]",
        "Rotational speed [rpm]": "num__Rotational speed [rpm]",
        "Torque [Nm]": "num__Torque [Nm]",
        "Tool wear [min]": "num__Tool wear [min]",
        "temp_diff_K": "num__temp_diff_K",
        "power_W": "num__power_W",
        "torque_x_toolwear": "num__torque_x_toolwear",
        "speed_torque_ratio": "num__speed_torque_ratio",
        "air_temp_z_within_type": "num__air_temp_z_within_type",
        "tool_wear_bucket": "cat_ordinal__tool_wear_bucket"
    }
    X = X.rename(columns=num_mappings)
    if "Type" in X.columns:
        X["cat_onehot__Type_H"] = (X["Type"] == "H").astype(int)
        X["cat_onehot__Type_L"] = (X["Type"] == "L").astype(int)
        X["cat_onehot__Type_M"] = (X["Type"] == "M").astype(int)
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0.0
    return X[expected_features] if expected_features else X

def simulate_sensor_degradation(
    model_path="models/random_forest_tuned_final.pkl", 
    data_path="data/ml_feature_ready.csv",
    output_dir="reports/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    expected_features = load_expected_features()
    
    print("1/4 Loading model and data...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X_clean = prepare_features(df, expected_features)
    
    # Define drift simulation params (5 sequential time steps)
    time_steps = [0, 1, 2, 3, 4]
    failure_probs = []
    
    print("2/4 Simulating progressive sensor drift (Thermal & Torque Noise)...")
    for t in time_steps:
        X_drifted = X_clean.copy()
        
        # Inject linear drift (sensor calibration degradation)
        if "num__Air temperature [K]" in X_drifted.columns:
            X_drifted["num__Air temperature [K]"] += t * 3.5  # Gradual temperature increase
        if "num__Torque [Nm]" in X_drifted.columns:
            noise = np.random.normal(0, t * 2.0, size=len(X_drifted))
            X_drifted["num__Torque [Nm]"] += noise  # Increasing sensor signal noise
            
        # Predict failure probability under drifted state
        preds = model.predict_proba(X_drifted.values)[:, 1]
        avg_prob = np.mean(preds)
        failure_probs.append(avg_prob)
        print(f"   Batch {t}: Mean Predicted Failure Risk = {avg_prob:.4f}")
        
    print("3/4 Plotting sensor degradation impact...")
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, failure_probs, marker='o', color='#e74c3c', linewidth=2.5)
    plt.title("Model Output Sensitivity to Sensor Drift & Noise")
    plt.xlabel("Drift Time Step (Simulated Sensor Degradation)")
    plt.ylabel("Mean Predicted Failure Probability")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    output_path = os.path.join(output_dir, "feature_drift_impact.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    
    print(f"\nSUCCESS! Drift simulation completed. Report saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    simulate_sensor_degradation()