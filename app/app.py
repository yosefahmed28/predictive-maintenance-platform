"""
app.py — Predictive Maintenance Machine-Health Dashboard
==========================================================
Role 6 (Application Developer & DevOps Lead)

The main dashboard. Currently running on mock components until real
models and features are received from the rest of the team
(see MODEL_CONTRACT.md).

Local run:
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd

from loaders import (
    get_feature_columns,
    load_classical_model,
    load_dl_models,
    explain_prediction,
    load_clean_data,
)

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="\U0001F3ED",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load components (automatically falls back to mock if real files are missing)
# ---------------------------------------------------------------------------
feature_cols = get_feature_columns()
classical_model, classical_is_mock = load_classical_model()
mlp_model, ae_model, dl_is_mock = load_dl_models()

if classical_is_mock or dl_is_mock:
    st.warning(
        "\u26A0\uFE0F The dashboard is currently running on mock models until "
        "the real ones are received from Role 3 and Role 4. Update the paths "
        "in `app/loaders.py` (the PATHS dict) once you receive them.",
        icon="\u26A0\uFE0F",
    )

st.title("\U0001F3ED Predictive Maintenance — Machine Health Dashboard")

tab_status, tab_batch, tab_explain, tab_about = st.tabs(
    ["\U0001F4CA Machine Status", "\U0001F4C1 Batch Scoring", "\U0001F50D Explainability", "\u2139\uFE0F About"]
)

# ---------------------------------------------------------------------------
# Tab 1: Machine status (overview)
# ---------------------------------------------------------------------------
with tab_status:
    st.subheader("Overview of machine status")
    df = load_clean_data()

    # Compute failure risk for each machine using the model (real or mock)
    available_cols = [c for c in feature_cols if c in df.columns]
    if available_cols:
        proba = classical_model.predict_proba(df[available_cols])[:, 1]
        df["failure_risk"] = proba
        df["status"] = pd.cut(
            df["failure_risk"],
            bins=[-0.01, 0.3, 0.7, 1.01],
            labels=["\U0001F7E2 Healthy", "\U0001F7E1 Needs monitoring", "\U0001F534 High failure risk"],
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total machines", len(df))
        col2.metric("Machines at risk \U0001F534", int((df["status"] == "\U0001F534 High failure risk").sum()))
        col3.metric(
            "Average risk score",
            f"{df['failure_risk'].mean() * 100:.1f}%",
        )

        st.dataframe(
            df[["machine_id", "failure_risk", "status"] + available_cols]
            .sort_values("failure_risk", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No matching feature columns yet — make sure feature_columns.json from Role 2 has been received.")

# ---------------------------------------------------------------------------
# Tab 2: Upload CSV and score a batch of machines
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV file to score a batch of machines")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head(), use_container_width=True)

        cols_present = [c for c in feature_cols if c in batch_df.columns]
        missing = [c for c in feature_cols if c not in batch_df.columns]

        if missing:
            st.error(f"Missing columns in the file: {missing}")
        else:
            proba = classical_model.predict_proba(batch_df[cols_present])[:, 1]
            batch_df["failure_risk"] = proba
            st.success("Scoring completed successfully \u2705")
            st.dataframe(
                batch_df.sort_values("failure_risk", ascending=False),
                use_container_width=True,
            )
            st.download_button(
                "Download results as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_machines.csv",
                mime="text/csv",
            )
    else:
        st.caption("No file uploaded yet — try uploading a sample CSV with the same feature columns.")

# ---------------------------------------------------------------------------
# Tab 3: Explainability
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Why this prediction?")
    df = load_clean_data()
    available_cols = [c for c in feature_cols if c in df.columns]

    if available_cols and len(df) > 0:
        machine_choice = st.selectbox("Choose a machine", df["machine_id"] if "machine_id" in df else df.index)
        row = df[df["machine_id"] == machine_choice] if "machine_id" in df else df.iloc[[0]]

        explanation = explain_prediction(classical_model, row[available_cols])
        st.write(explanation.get("note", ""))
        st.write("Top factors influencing this prediction:")
        for name, val in explanation.get("top_features", []):
            st.write(f"- **{name}**: {val:.2f}")
    else:
        st.info("Not enough data available yet to show an explanation.")

# ---------------------------------------------------------------------------
# Tab 4: About the project
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
        - Role 3: Classical ML
        - Role 4: Deep Learning & Anomaly Detection
        - Role 5: Explainable AI & MLOps
        - Role 6: Application & DevOps (this app)
        """
    )
