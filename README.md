# Industrial Predictive Maintenance & Failure Prevention

An end-to-end pipeline that takes the raw **AI4I 2020 Predictive Maintenance Dataset** and turns it into a tuned, explainable, AI-assisted machine-health dashboard — covering data engineering, EDA, classical ML, deep learning, explainability/MLOps (with a RAG maintenance assistant), and a containerized Streamlit app.

---

## Architecture

```
Raw AI4I 2020 CSV
       │
       ▼
 SQL / ETL layer  ───────────────►  MySQL (predictive_maintenance)
       │
       ▼
 EDA & Feature Engineering  ─────►  leakage-safe train/val/test splits
       │
       ├──────────────┐
       ▼              ▼
 Classical ML     Deep Learning
 (Random Forest)  (MLP, Autoencoder, Deep SVDD)
       │              │
       └──────┬───────┘
              ▼
     Explainable AI & MLOps
     (SHAP · logging · drift/cost sim · RAG maintenance assistant)
              │
              ▼
     Streamlit Dashboard  ─────►  Docker container
```

Data flows from the raw dataset through the SQL/ETL layer, into classical ML and deep learning models, through an explainability layer, and finally into the Streamlit dashboard, which is containerized with Docker.

---

## Team & Role Breakdown

| # | Role | Owns |
|---|------|------|
| 1 | **Data Engineering & SQL Database Lead** | Cleaning, normalized MySQL schema, ETL pipeline, 12 analytical SQL queries |
| 2 | **EDA & Feature Engineering Lead** | Exploratory analysis, leakage checks, feature engineering, leakage-safe train/val/test pipeline |
| 3 | **Classical Machine Learning Lead** | Baseline-to-tuned models (Logistic Regression → Random Forest/XGBoost), imbalance handling, threshold tuning |
| 4 | **Deep Learning & Anomaly Detection Specialist** | Supervised MLP, Autoencoder, Deep SVDD; DL vs. classical ML vs. anomaly-score comparison |
| 5 | **Explainable AI (XAI) & MLOps Specialist** | SHAP interpretability, prediction logging/versioning, drift & cost simulation, RAG maintenance assistant |
| 6 | **Application Developer & DevOps Lead** | Streamlit dashboard, Docker container, test suite, repo structure |


---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository) — 10,000 rows, fully synthetic, cross-sectional (each row is one independently-simulated machine observed once, not a time series of one physical machine). There is **no timestamp column**, no missing values, and no duplicate rows.

- **Target:** `Machine failure` (binary) — **3.39% positive class**, a ~28.5:1 imbalance
- **Features:** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- **Excluded as predictors (leakage):** `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — post-hoc failure-mode outcome labels, ~99.7% agreement with the target, would let a model "cheat"
- **Known data-quality note:** 27 rows have `Machine failure` inconsistent with the five failure-mode flags — a documented AI4I2020 quirk, preserved as ground truth rather than "corrected"

Full EDA: `notebooks/EDA_FeatureEngineering.ipynb`.

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up the database (Role 1)

```bash
mysql -u root -p < sql/00_local_setup.sql   # one-time: creates DB + app user
python scripts/etl.py                        # extract, transform, load
```

Connection settings (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`) are defined near the top of `scripts/etl.py` and default to match `sql/00_local_setup.sql`.

Run the analytical queries directly against the loaded database:

```bash
mysql -u pm_user -p predictive_maintenance < sql/02_analytical_queries.sql
```

### 3. Explore the EDA & feature pipeline (Role 2)

Open `notebooks/EDA_FeatureEngineering.ipynb`, or reuse the packaged, importable pipeline directly:

```python
from src.features import build_pipeline_splits
splits = build_pipeline_splits('data/ai4i2020_raw.csv')
```

### 4. Train / inspect the models (Roles 3 & 4)

Classical ML: `notebooks/ML_Modeling.ipynb` (or `src/model_utils.train_and_evaluate()`).
Deep learning: `notebook/DL_Models.ipynb`.

To score new raw data with the saved classical model:

```python
import joblib
from src.model_utils import predict_new_sample

model = joblib.load('models/random_forest_tuned_final.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
zscorer = joblib.load('models/zscorer.pkl')
result = predict_new_sample(model, preprocessor, zscorer, raw_df, threshold=0.5)
```

**Never refit `preprocessor.pkl` / `zscorer.pkl`** — they are the exact fitted objects from training time; only `.transform()` them.

### 5. Build the RAG maintenance assistant's knowledge base (Role 5)

```bash
python src/rag/build_vectorstore.py     # chunks + embeds data/docs into ChromaDB
```

Requires a local [Ollama](https://ollama.com) install with `llama3.2` pulled (`ollama pull llama3.2`). Then run the assistant:

```bash
python src/rag/assistant.py
```

```
User Query > what should I do about machine 4045?
```

The agent checks live telemetry and predicted failure risk first, then — if risk looks elevated — retrieves the matching repair protocol from the embedded technical manuals before giving a recommendation.

### 6. Run the dashboard (Role 6)

```bash
streamlit run app/app.py
```

Or via Docker:

```bash
docker build -t predictive-maintenance .
docker run -p 8501:8501 predictive-maintenance
```

Then open `http://localhost:8501`. The dashboard has 5 tabs: **Machine Status**, **Batch Scoring**, **DL & Anomaly**, **Explainability**, **About**. If any teammate's model artifact is missing, `app/loaders.py` automatically falls back to a small mock so the app never crashes.

### 7. Run the tests

```bash
pytest tests/ -v
```

---

## Results Summary

| Model | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **Random Forest (tuned)** — final model | **0.905** | 0.765 | 0.975 | 0.857 |
| XGBoost (class-weighted) | 0.869 | 0.804 | 0.804 | 0.804 |
| Supervised MLP | 0.842 | 0.784 | 0.667 | 0.721 |
| Decision Tree (depth=5) | 0.832 | 0.824 | 0.913 | 0.866 |
| Logistic Regression (baseline) | 0.396 | 0.216 | 0.550 | 0.310 |
| Tuned Autoencoder | 0.162 | 0.451 | 0.178 | 0.256 |
| Deep SVDD | 0.148 | 0.216 | 0.407 | 0.282 |

The tuned Random Forest was selected as the production model — best PR-AUC and precision of every model tested, per the project's evaluation priority (recall/PR-AUC/F1/false-negative rate over raw accuracy, given the ~3.4% failure rate). The deployment decision threshold (default 0.5 vs. a more aggressive ~0.22–0.30) is intentionally left open pending a joint cost-based alerting decision between Classical ML and XAI/MLOps.

Full comparison tables: `reports/model_comparison_classical_ml.csv`, `reports/ml_vs_dl_comparison.csv`.

---

## Limitations

- **AI4I 2020 is a synthetic benchmark.** Results demonstrate methodology, not real-world industrial reliability.
- **No real temporal structure** exists in the source data (no timestamp, every unit appears once) — rolling/lag/time-window features were deliberately not built. Any "live stream" simulation in the dashboard is an explicitly-labeled row-replay, not real sequential sensor history.
- **27 rows** carry an unresolved inconsistency between `Machine failure` and the five failure-mode flags — preserved as-is, not corrected.
- **The deployment decision threshold is still open**, pending Role 5's cost-based alerting simulation to replace the `$5,000`/failure placeholder used in the SQL cost-feeder query.

## Next Steps

- Finalize the production decision threshold jointly (Classical ML + XAI/MLOps).
- Expand feature-drift and sensor-degradation testing beyond the current static benchmark.
- Grow the RAG assistant's manual/protocol knowledge base and evaluate its answers against known-correct fixes.
- Revisit resampling vs. class-weighting trade-offs once real deployment cost data exists.

---

## License

Educational / capstone project. AI4I 2020 dataset used under its original UCI Machine Learning Repository terms.
