"""
app.py — Predictive Maintenance Machine-Health Dashboard
==========================================================
Role 6 (Application Developer & DevOps Lead)

Now wired to the real artifacts received from the team (see
Handoff_to_DL_MLOPS.md): Role 3's tuned Random Forest, Role 4's MLP +
Autoencoder, and Role 5's SHAP explainability. loaders.py falls back to
mock components automatically if a specific file is missing.

Local run:
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd

from loaders import (
    get_raw_input_columns,
    load_classical_model,
    load_dl_models,
    load_clean_data,
    clean_df_to_raw,
    score_raw_dataframe,
    score_dl,
    explain_prediction,
    DEFAULT_CLASSICAL_THRESHOLD,
)

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="\U0001F3ED",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load components once per session
# ---------------------------------------------------------------------------
raw_cols = get_raw_input_columns()
_, classical_is_mock = load_classical_model()
_, _, dl_is_mock = load_dl_models()

if classical_is_mock or dl_is_mock:
    st.warning(
        "\u26A0\uFE0F Running on mock models because a model file wasn't found under `models/`. "
        "Check that random_forest_tuned_final.pkl, tabular_mlp_tuned.keras and "
        "autoencoder_anomaly_detector_tuned.keras were copied into that folder.",
        icon="\u26A0\uFE0F",
    )

st.title("\U0001F3ED Predictive Maintenance — Machine Health Dashboard")

st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Classical model decision threshold",
    min_value=0.0, max_value=1.0, value=DEFAULT_CLASSICAL_THRESHOLD, step=0.01,
    help=(
        "Not finalized yet (see Handoff_to_DL_MLOPS.md, Section 2) — "
        "lower thresholds raise recall at the cost of more false alarms. "
        "Adjust to explore the tradeoff while Role 5's cost simulation is pending."
    ),
)

tab_status, tab_batch, tab_dl, tab_explain, tab_about = st.tabs(
    ["\U0001F4CA Machine Status", "\U0001F4C1 Batch Scoring",
     "\U0001F9E0 DL & Anomaly", "\U0001F50D Explainability", "\u2139\uFE0F About"]
)

# ---------------------------------------------------------------------------
# Tab 1: Machine status (overview, scored with the classical model)
# ---------------------------------------------------------------------------
with tab_status:
    st.subheader("Overview of machine status")
    df = load_clean_data()

    if all(c in df.columns for c in ["air_temperature_k", "process_temperature_k",
                                       "rotational_speed_rpm", "torque_nm", "tool_wear_min"]):
        raw_view = clean_df_to_raw(df)
        scored = score_raw_dataframe(raw_view, threshold=threshold)
        df["failure_risk"] = scored["failure_probability"].values
        df["status"] = pd.cut(
            df["failure_risk"],
            bins=[-0.01, 0.3, 0.7, 1.01],
            labels=["\U0001F7E2 Healthy", "\U0001F7E1 Needs monitoring", "\U0001F534 High failure risk"],
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total machines", len(df))
        col2.metric("Machines at risk \U0001F534", int((df["status"] == "\U0001F534 High failure risk").sum()))
        col3.metric("Average risk score", f"{df['failure_risk'].mean() * 100:.1f}%")

        display_cols = ["machine_id", "failure_risk", "status", "air_temperature_k",
                         "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values("failure_risk", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("data/ml_feature_ready.csv doesn't have the expected columns yet.")

# ---------------------------------------------------------------------------
# Tab 2: Upload CSV and score a batch of machines (raw AI4I columns)
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV file to score a batch of machines")
    st.caption(f"Expected raw columns: {', '.join(raw_cols)}")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head(), use_container_width=True)

        missing = [c for c in raw_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns in the file: {missing}")
        else:
            scored = score_raw_dataframe(batch_df[raw_cols], threshold=threshold)
            result_df = pd.concat([batch_df.reset_index(drop=True), scored], axis=1)
            st.success("Scoring completed successfully \u2705")
            st.dataframe(
                result_df.sort_values("failure_probability", ascending=False),
                use_container_width=True,
            )
            st.download_button(
                "Download results as CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_machines.csv",
                mime="text/csv",
            )
    else:
        st.caption("No file uploaded yet — try a CSV with the raw AI4I columns listed above.")

# ---------------------------------------------------------------------------
# Tab 3: Deep Learning classifier + Autoencoder anomaly score
# ---------------------------------------------------------------------------
with tab_dl:
    st.subheader("Deep Learning classifier vs. Autoencoder anomaly score")
    st.caption(
        "Compares Role 4's MLP failure probability against the Autoencoder's "
        "reconstruction-error anomaly score, on the same clean dataset."
    )
    df = load_clean_data()

    if all(c in df.columns for c in ["air_temperature_k", "process_temperature_k",
                                       "rotational_speed_rpm", "torque_nm", "tool_wear_min"]):
        raw_view = clean_df_to_raw(df)
        dl_scores = score_dl(raw_view)
        merged = pd.concat([df.reset_index(drop=True), dl_scores], axis=1)

        col1, col2 = st.columns(2)
        col1.metric("Flagged as anomaly \U0001F6A8", int(merged["is_anomaly"].sum()))
        col2.metric("Avg. MLP failure probability", f"{merged['mlp_probability'].mean() * 100:.1f}%")

        show_cols = [c for c in ["machine_id", "mlp_probability", "anomaly_score", "is_anomaly"] if c in merged.columns]
        st.dataframe(
            merged[show_cols].sort_values("anomaly_score", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("data/ml_feature_ready.csv doesn't have the expected columns yet.")

# ---------------------------------------------------------------------------
# Tab 4: Explainability (SHAP, from Role 5)
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Why this prediction?")
    df = load_clean_data()

    if "machine_id" in df.columns and len(df) > 0:
        machine_choice = st.selectbox("Choose a machine", df["machine_id"])
        row = df[df["machine_id"] == machine_choice].iloc[[0]]
        raw_row = clean_df_to_raw(row)

        explanation = explain_prediction(raw_row)
        st.write(explanation.get("note", ""))
        st.write("Top factors influencing this prediction:")
        for name, val in explanation.get("top_features", []):
            st.write(f"- **{name}**: {val:.3f}")
    else:
        st.info("Not enough data available yet to show an explanation.")

# ---------------------------------------------------------------------------
# Tab 5: About the project
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **Industrial Predictive Maintenance & Failure Prevention**

        A platform for estimating industrial machine failure risk and
        recommending maintenance actions, built on the AI4I 2020 dataset.

        - Role 1: Data Engineering & SQL
        - Role 2: EDA & Feature Engineering
        - Role 3: Classical ML (tuned Random Forest)
        - Role 4: Deep Learning & Anomaly Detection (MLP + Autoencoder)
        - Role 5: Explainable AI & MLOps (SHAP)
        - Role 6: Application & DevOps (this app)

        **Note:** the AI4I 2020 dataset is synthetic. Results here
        demonstrate the modeling/deployment workflow and should not be
        read as evidence of real industrial reliability.
        """
    )
