import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
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
    
    # Map raw numeric columns to preprocessed names if present
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
    
    # Handle One-Hot Encoded 'Type' column if present
    if "Type" in X.columns:
        X["cat_onehot__Type_H"] = (X["Type"] == "H").astype(int)
        X["cat_onehot__Type_L"] = (X["Type"] == "L").astype(int)
        X["cat_onehot__Type_M"] = (X["Type"] == "M").astype(int)
    
    # Fill missing expected features with 0
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0.0
            
    # Filter and order strictly by expected features
    if expected_features:
        X = X[expected_features]
        
    return X

def generate_shap_plots(model_path, data_path, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    expected_features = load_expected_features()
    
    print(f"1/4 Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"2/4 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("3/4 Aligning features with model metadata...")
    X_ready = prepare_features(df, expected_features)
    X_sample = X_ready.sample(n=min(100, len(X_ready)), random_state=42)
    
    print(f"4/4 Calculating SHAP values for {X_sample.shape[1]} features...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values on aligned NumPy matrix
    shap_vals = explainer.shap_values(X_sample.values, check_additivity=False)
    
    if isinstance(shap_vals, list):
        vals_to_plot = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        vals_to_plot = shap_vals[:, :, 1]
    else:
        vals_to_plot = shap_vals

    print("Generating and saving SHAP summary plot...")
    
    # Clean feature names for clean plot labeling
    clean_feature_names = [c.replace("num__", "").replace("cat_onehot__", "").replace("cat_ordinal__", "") for c in X_sample.columns]
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals_to_plot, X_sample, feature_names=clean_feature_names, show=False)
    
    output_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close("all")
    
    print(f"\nSUCCESS! File saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    MODEL_PATH = "models/random_forest_tuned_final.pkl"
    DATA_PATH = "data/ml_feature_ready.csv"
    
    generate_shap_plots(MODEL_PATH, DATA_PATH)