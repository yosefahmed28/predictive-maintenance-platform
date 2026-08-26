import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow

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

def run_mlflow_xai_pipeline(
    model_path="models/random_forest_tuned_final.pkl", 
    data_path="data/ml_feature_ready.csv",
    output_dir="reports/figures"
):
    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_experiment("XAI_and_Drift_Monitoring")

    with mlflow.start_run(run_name="Role_5_XAI_Drift_Pipeline"):
        print("1/5 Loading model and data...")
        expected_features = load_expected_features()
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        X_ready = prepare_features(df, expected_features)
        X_sample = X_ready.sample(n=min(100, len(X_ready)), random_state=42)

        # Log parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("sample_size", len(X_sample))
        mlflow.log_param("num_features", len(expected_features))

        # 2. Compute SHAP Values
        print("2/5 Generating SHAP summary plot...")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample.values, check_additivity=False)
        
        vals_to_plot = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        clean_feature_names = [c.replace("num__", "").replace("cat_onehot__", "").replace("cat_ordinal__", "") for c in X_sample.columns]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals_to_plot, X_sample, feature_names=clean_feature_names, show=False)
        shap_plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(shap_plot_path, bbox_inches="tight", dpi=300)
        plt.close("all")

        # 3. Simulate Sensor Drift
        print("3/5 Simulating sensor drift across 5 time steps...")
        time_steps = [0, 1, 2, 3, 4]
        failure_probs = []

        for t in time_steps:
            X_drifted = X_ready.copy()
            if "num__Air temperature [K]" in X_drifted.columns:
                X_drifted["num__Air temperature [K]"] += t * 3.5
            if "num__Torque [Nm]" in X_drifted.columns:
                X_drifted["num__Torque [Nm]"] += np.random.normal(0, t * 2.0, size=len(X_drifted))

            preds = model.predict_proba(X_drifted.values)[:, 1]
            avg_prob = float(np.mean(preds))
            failure_probs.append(avg_prob)
            mlflow.log_metric(f"mean_failure_prob_step_{t}", avg_prob)

        # Plot Drift Impact
        plt.figure(figsize=(8, 5))
        plt.plot(time_steps, failure_probs, marker='o', color='#e74c3c', linewidth=2.5)
        plt.title("Model Sensitivity to Simulated Sensor Degradation")
        plt.xlabel("Drift Time Step")
        plt.ylabel("Mean Predicted Failure Probability")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        drift_plot_path = os.path.join(output_dir, "feature_drift_impact.png")
        plt.savefig(drift_plot_path, dpi=300, bbox_inches="tight")
        plt.close("all")

        # 4. Log Artifacts to MLflow
        print("4/5 Logging artifacts to MLflow...")
        mlflow.log_artifact(shap_plot_path)
        mlflow.log_artifact(drift_plot_path)

        print(f"\n5/5 SUCCESS! Pipeline execution complete.")
        print(f"   - Reports generated inside: {os.path.abspath(output_dir)}")
        print("   - All metrics, parameters, and plots logged to MLflow successfully.")

if __name__ == "__main__":
    run_mlflow_xai_pipeline()