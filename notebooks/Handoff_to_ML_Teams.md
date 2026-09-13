# Handoff Document — EDA & Feature Engineering → Classical ML & Deep Learning
**Project:** Industrial Predictive Maintenance & Failure Prevention
**Source dataset:** AI4I 2020 Predictive Maintenance Dataset, `ai4i2020.csv`

**Companion files:**
- `EDA_FeatureEngineering.ipynb` — full EDA, leakage analysis, and reasoning
- `src/features.py` — importable, reproducible pipeline (use this directly rather than re-implementing)
- `Statistical_Analysis_Summary.md` — condensed statistical findings

---

## 1. Target column

**`Machine failure`** (binary: 1 = failure, 0 = normal).

- Class balance: **3.39% positive** (339 / 10,000). Severe imbalance.
- **Do not use accuracy as the primary evaluation metric.** Use recall,
  PR-AUC, F1, and false-negative rate, per the project's stated priority
  on avoiding missed failures. ROC-AUC can look artificially good under
  this level of imbalance — prefer PR-AUC alongside or instead of it.
- Consider class weighting, resampling (e.g. SMOTE — evaluate carefully,
  only on the training fold), or threshold tuning against a
  business-relevant recall/precision trade-off. This decision is left to
  Classical ML/Deep Learning depending on model family.

## 2. Final feature list

Use `features.build_pipeline_splits()` to get these already
scaled/encoded. Raw column names before encoding:

**Numeric (standardized within the pipeline):**
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `temp_diff_K` *(engineered: Process − Air temperature)*
- `power_W` *(engineered: Torque × angular velocity)*
- `torque_x_toolwear` *(engineered: Torque × Tool wear)*
- `speed_torque_ratio` *(engineered: Rotational speed ÷ Torque)*
- `air_temp_z_within_type` *(engineered: Air temperature z-scored within product Type, fit on train only)*

**Categorical:**
- `Type` (one-hot encoded: H / L / M)
- `tool_wear_bucket` (ordinal encoded: very_low < low < medium < high < critical)

Final ML-ready array has **14 columns** after encoding (see
`src/features.py` output for exact order via
`splits['feature_names']`).

## 3. Features removed, and why

| Feature | Reason |
|---|---|
| `UDI` | Row identifier; no generalizable predictive value |
| `Product ID` | High-cardinality unique identifier; its only informative part (leading letter) is already captured by `Type` |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | **Data leakage.** These are post-failure diagnostic/outcome labels (which failure mechanism occurred), not pre-failure sensor readings. A simple "any mode flagged" rule agrees with `Machine failure` ~99.7% of the time — using these as features would let a model trivially "predict" the target from a near-restatement of it. Full reasoning in notebook Section 7. |

**If you are tempted to re-add any of the five failure-mode columns as a
feature "just to see the score go up" — don't.** That would defeat the
purpose of the leakage analysis and produce a model that cannot work on
genuinely new, unlabeled data (where those flags won't exist yet).

## 4. Preprocessing requirements

Use `features.build_pipeline_splits('ai4i2020.csv')` directly
rather than re-deriving preprocessing logic. It handles, in this order:

1. Column selection (drops identifiers + leakage columns)
2. **Stratified train/val/test split (70/15/15) BEFORE any statistic is fit**
3. Deterministic feature engineering (stateless — safe on any split)
4. `TypeGroupedZScorer` — fit on train only, applied (not refit) to val/test
5. `StandardScaler` (numeric) + `OneHotEncoder` (`Type`) + `OrdinalEncoder`
   (`tool_wear_bucket`) — all fit on train only, applied to val/test

**Do not fit any scaler/encoder/statistic on the full dataset before
splitting** — this is the single most important leakage risk left in the
pipeline design, since several features (notably
`air_temp_z_within_type` and any future scaling) depend on fitted
statistics. The pipeline script already handles this correctly; if you
build your own variant (e.g., for a different framework), preserve this
split-then-fit ordering.

## 5. Train / validation / test strategy

- **Stratified split** (70% train / 15% val / 15% test), stratified on
  `Machine failure` to preserve the ~3.4% failure rate in every split
  (`random_state=42` for reproducibility).
- Given the small absolute number of positive cases (339 total, so
  ~237/51/51 per split), consider **k-fold cross-validation on the
  training set** rather than relying on a single validation split when
  tuning hyperparameters, to reduce variance in recall/PR-AUC estimates.
- Do not use `UDI` order as a time-based split — it is not a real
  timestamp (see Section 6 below).

## 6. Important assumptions & limitations

- **No real temporal structure exists in the raw data.** There is no
  timestamp, and every `Product ID` is unique (one row per machine, not
  repeated readings). **Rolling/lag/time-window features are not
  justified** on this dataset and were not built. If a later stage
  (e.g. the Streamlit dashboard) needs to simulate a live stream, that
  should be an explicitly-labeled row-replay simulation, not a claim of
  real sequential sensor history.
- **AI4I 2020 is a synthetic benchmark.** Model performance on this
  dataset should not be presented as evidence of real-world industrial
  reliability — the whole exercise (per project rules) is a realistic
  *methodology* demonstration, not a real deployment result.
- The five failure-mode flags are **not perfectly internally consistent**
  with `Machine failure` (27 rows total disagree). This is a dataset
  quirk, not a EDA & Feature Engineering processing error — documented, not "fixed."
  `Machine failure` is treated as the sole ground truth throughout.
- `air_temp_z_within_type`'s fitted per-`Type` statistics are specific to
  the training split used; if the training set composition changes
  (e.g. new data added), re-fit `TypeGroupedZScorer` rather than reusing
  old statistics.
- Coordinate with Data Engineering on the three open items listed in notebook
  Section 12 (temporal assumptions in the SQL schema, stable use of
  `UDI` as a join key, and consistent target derivation) before finalizing
  any shared database view that both roles depend on.

## 7. Suggested next steps for Classical ML & Deep Learning

- Classical ML: start with a couple of imbalance-aware baselines
  (e.g. logistic regression with `class_weight='balanced'`, and a
  tree-based ensemble) evaluated on PR-AUC/recall/F1; use the ordinal
  `tool_wear_bucket` as-is for tree models, but consider one-hot encoding
  it instead for linear models if the ordinal assumption doesn't hold up.
- Deep learning: given the small positive-class count (339),
  expect high variance in a small validation split — consider stratified
  k-fold or a focal-loss-style objective to address imbalance directly
  in training rather than only via resampling.
