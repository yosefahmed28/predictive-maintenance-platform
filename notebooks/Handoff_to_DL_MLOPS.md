# Handoff Document — Classical ML → Deep Learning & XAI/MLOps
**Project:** Industrial Predictive Maintenance & Failure Prevention
**Source dataset:** AI4I 2020 Predictive Maintenance Dataset, `ai4i2020.csv`
**Upstream handoff:** see `EDA_FeatureEngineering.ipynb` and `reports/` from EDA & Feature Engineering (Role 2)

**Companion files:**
- `notebooks/02_classical_ml.ipynb` — full model comparison, tuning, and reasoning
- `notebooks/03_predict_example.ipynb` — minimal standalone example showing how to load the saved model and score new/raw data
- `src/model_utils.py` — importable `train_and_evaluate()`, plotting functions, and `predict_new_sample()`
- `reports/model_comparison_classical_ml.csv` — final comparison table, all models
- `reports/classical_ml_final_model_metadata.json` — best hyperparameters, threshold, validation metrics

---

## 1. Models trained and compared

Six models were trained and evaluated on the same `X_val`/`y_val` split produced by `build_pipeline_splits()` (from Role 2's `src/features.py`):

| Model | PR-AUC | Recall | Precision | F1 | FN Rate |
|---|---|---|---|---|---|
| Random Forest | 0.905 | 0.765 | 0.975 | 0.857 | 0.235 |
| XGBoost (class-weighted) | 0.869 | 0.804 | 0.804 | 0.804 | 0.196 |
| XGBoost (weighted, threshold=0.004) | 0.869 | 0.980 | 0.301 | 0.461 | 0.020 |
| XGBoost | 0.852 | 0.804 | 0.854 | 0.828 | 0.196 |
| Decision Tree (max_depth=5) | 0.832 | 0.824 | 0.913 | 0.866 | 0.176 |
| Logistic Regression (baseline) | 0.396 | 0.216 | 0.550 | 0.310 | 0.784 |
| Random Forest (class-weighted) | 0.787 | 0.824 | 0.545 | 0.656 | 0.176 |
| Random Forest (weighted, threshold=0.224) | 0.787 | 0.980 | 0.303 | 0.463 | 0.020 |

Note -> "Class weighting was chosen over SMOTE to avoid introducing synthetic sensor readings that may not reflect physically realistic machine states, and because it directly integrates with tree-based models' native class_weight parameter without extra preprocessing steps." That shows the choice was deliberate, not just "whichever was easier."

Full table (including confusion matrix counts) is in
`reports/model_comparison_classical_ml.csv`.

## 2. Final model selected

**Random Forest, hyperparameter-tuned** (via `RandomizedSearchCV`,
5-fold stratified CV, scored on PR-AUC).

- Chosen over XGBoost because unweighted Random Forest had the best
  PR-AUC (0.905) and precision (0.975) of all models tested — this was
  treated as the primary selection metric per the project's stated
  evaluation priority.
- Best hyperparameters and exact validation metrics for the tuned model
  are saved in `reports/classical_ml_final_model_metadata.json`.
- **Default threshold (0.5) is used in the saved model as of this
  handoff.** A more aggressive threshold (~0.22–0.30) was tested and can
  push recall to ~98%, but at a steep precision cost (~30%, ~115 false
  alarms per 1500 machines). This tradeoff is documented in
  `02_classical_ml.ipynb` (Sections 5–6) but **not finalized** — the
  actual deployment threshold should be set jointly with XAI/MLOps once
  the cost-based alerting rules (cost of a missed failure vs. an
  unnecessary inspection) are defined. Do not assume 0.5 is the "right"
  threshold for the dashboard without that conversation.

## 3. Saved artifacts (what's in `models/`)

| File | What it is | Who needs it |
|---|---|---|
| `random_forest_tuned_final.pkl` | Final tuned Random Forest model | XAI/MLOps (SHAP), App/DevOps (Streamlit) |
| `preprocessor.pkl` | Fitted `ColumnTransformer` (scaler + encoders) — fit on train only | Anyone scoring new raw data |
| `zscorer.pkl` | Fitted `TypeGroupedZScorer` (per-`Type` air temp z-score) — fit on train only | Same as above |

**Important:** these two `.pkl` files are the *exact fitted objects* used
during training — do not refit them on new data. To score any new raw
sample, load these three files and use `predict_new_sample()` in
`src/model_utils.py` (see `03_predict_example.ipynb` for a working
example). Do not call `build_pipeline_splits()` at prediction time —
that function re-splits and re-fits from scratch and is for training
only.

## 4. Validation strategy used

- Same stratified 70/15/15 split as Role 2 (`random_state=42`) —
  `X_train`/`y_val` etc. from `build_pipeline_splits()`, never re-split.
- Hyperparameter search used 5-fold stratified cross-validation on the
  **training set only** — `X_val`/`y_val` were held out completely
  during the search and used only afterward to report final metrics.
- **Test set (`X_test`/`y_test`) has not been touched yet** — reserved
  for the final one-time evaluation once all tracks (classical ML, DL)
  are complete, so it stays an honest, unbiased final check.

## 5. Known limitations / things to be aware of

- Class imbalance (3.39% positive) makes recall/PR-AUC estimates
  sensitive to which few positive cases land in validation vs. test —
  numbers may shift somewhat if the split changes for any reason (it
  shouldn't, since `random_state=42` is fixed everywhere).
- Class-weighting was tested on both XGBoost and Random Forest;
  weighting **improves recall at a fixed default threshold** but
  actually **lowers PR-AUC** for Random Forest (0.905 → 0.787) — i.e.
  it makes the model rank candidates less reliably overall, even though
  it changes the 0.5-threshold decision favorably. This nuance matters
  if DL/XAI compare "weighted" vs "unweighted" models — compare on
  PR-AUC, not just recall-at-default-threshold, to avoid a misleading
  conclusion.
- Per the project's synthetic-benchmark caveat carried over from Role 2:
  these results should not be presented as evidence of real industrial
  reliability.

## 6. What I need from / expect of downstream roles

**Deep Learning (Role 4):**
- Please evaluate on the exact same `X_val`/`y_val` (via
  `build_pipeline_splits()`, same file path and `random_state`) so our
  results are directly comparable.
- Append your MLP/Autoencoder results to
  `reports/model_comparison_classical_ml.csv` using the same column
  schema (`model, threshold, precision, recall, f1, pr_auc,
  false_negative_rate, tn, fp, fn, tp`) rather than creating a separate
  file, so there's one unified comparison table for the final report.

**XAI & MLOps (Role 5):**
- `random_forest_tuned_final.pkl` is ready for SHAP — use
  `feature_names` from `splits['feature_names']` (same order as the
  model's input columns) to label SHAP plots correctly.
- The threshold question (Section 2 above) is unresolved — I'd like to
  finalize it together once your cost-based alerting simulation has
  real cost estimates, rather than deciding it in isolation here.

**App/DevOps (Role 6):**
- Use `predict_new_sample()` from `src/model_utils.py` — it handles
  raw-input → feature engineering → scaling → prediction in one call,
  so you shouldn't need to reimplement any preprocessing logic in the
  Streamlit app.
- See `03_predict_example.ipynb` for a minimal working example of the
  exact input format expected (raw column names, matching the original
  dataset).