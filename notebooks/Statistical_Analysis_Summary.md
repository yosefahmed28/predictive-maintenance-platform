# Statistical Analysis Summary — EDA & Feature Engineering (EDA & Feature Engineering)
**Project:** Industrial Predictive Maintenance & Failure Prevention
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI), `ai4i2020.csv` (10,000 rows, 14 raw columns)

This document summarizes the statistical findings behind the full analysis in
`EDA_FeatureEngineering.ipynb`. It is the concise reference version;
the notebook has the full plots, code, and reasoning.

---

## 1. Dataset quality

- 10,000 rows, 14 columns, **0 missing values**, **0 duplicate rows**.
- `UDI` is a plain sequential row index (1–10000); `Product ID` is unique
  per row — every row is a distinct product unit, not a repeated
  observation of the same machine.
- `Type` (product quality tier: L/M/H) matches the `Product ID` prefix
  letter for 100% of rows.

## 2. Target class balance

| | Count | % |
|---|---|---|
| Normal (`Machine failure` = 0) | 9,661 | 96.61% |
| Failure (`Machine failure` = 1) | 339 | 3.39% |

**Implication:** ~28.5:1 imbalance. Accuracy is not a usable primary
metric — a trivial always-normal classifier scores 96.6% accuracy while
catching zero failures. Recommended metrics: recall, PR-AUC, F1,
false-negative rate, consistent with the project's stated priority on
avoiding missed failures.

## 3. Failure-mode breakdown

| Mode | Meaning | Count | % of all rows | % of failure rows |
|---|---|---|---|---|
| HDF | Heat Dissipation Failure | 115 | 1.15% | ~34% |
| OSF | Overstrain Failure | 98 | 0.98% | ~29% |
| PWF | Power Failure | 95 | 0.95% | ~28% |
| TWF | Tool Wear Failure | 46 | 0.46% | ~14% |
| RNF | Random Failure | 19 | 0.19% | ~6% |

- Mode flags are **not perfectly consistent** with `Machine failure`:
  9 rows are labeled `Machine failure=1` with no mode flagged; 18 rows
  are labeled `Machine failure=0` with a mode flagged. This is a known
  quirk of the AI4I 2020 dataset, documented and left uncorrected (no
  ground truth exists to "fix" it).
- A simple "any mode flagged" rule agrees with `Machine failure` for
  ~99.7% of rows — near-perfect agreement, which is exactly why the
  mode columns are excluded as predictive features (they are functionally
  almost a restatement of the target).
- Each failure mode shows a distinct sensor signature consistent with the
  known AI4I generative design: TWF ↔ high tool wear; HDF ↔ small
  temperature gap + low speed; PWF ↔ power outside normal band; OSF ↔
  high torque × high tool wear; RNF ↔ no clear sensor signature (by design).

## 4. Association tests

**Mann-Whitney U test** (normal vs. failure groups), all five raw sensor
variables and six engineered numeric features, non-parametric because
`Tool wear [min]` is non-normal and group sizes are highly unequal
(9,661 vs 339):

- **All candidate numeric features are statistically significant**
  (p < 0.05) in separating failure vs. normal groups.
- Ranked by effect size (**|Cohen's d|**, most to least separating):
  `torque_x_toolwear`, `power_W`, `Torque [Nm]`, `Tool wear [min]`,
  `temp_diff_K`, `Rotational speed [rpm]`, then the remaining raw/derived
  variables with smaller effect sizes.
- With n=10,000, statistical significance is easy to achieve; effect size
  was used as the primary ranking signal rather than p-value alone.

**Chi-square test of independence** (`Type` vs. `Machine failure`):
significant association (p < 0.05) — lower-quality (`L`) units show a
visibly higher failure rate than `M`/`H`. `Type` is a legitimate,
leakage-free feature (assigned at production time).

## 5. Correlation / redundancy

- `Air temperature [K]` and `Process temperature [K]`: strongly positively
  correlated — motivates the `temp_diff_K` engineered feature as a way to
  capture their relationship in one column.
- `Rotational speed [rpm]` and `Torque [Nm]`: moderate negative
  correlation, consistent with an approximately constant-power design —
  motivates the `power_W` and `speed_torque_ratio` features.
- No raw+engineered feature pair exceeds |r| > 0.85 — engineered features
  add information rather than duplicating existing columns.

## 6. Leakage decisions (see notebook Section 7 for full reasoning)

| Removed | Reason |
|---|---|
| `UDI`, `Product ID` | Identifiers, no generalizable predictive meaning |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Post-failure outcome/diagnostic labels — leakage (≈99.7% agreement with target) |

| Kept | Reason |
|---|---|
| `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | All available prior to/at the moment of failure; no dependency on outcome information |

## 7. Feature engineering decisions

Six features engineered, all derived only from leakage-safe raw columns:
`temp_diff_K`, `power_W`, `torque_x_toolwear`, `tool_wear_bucket`,
`speed_torque_ratio`, `air_temp_z_within_type`. Full rationale and
per-feature leakage check in notebook Section 9. Note:
`air_temp_z_within_type` requires per-`Type` mean/std statistics that
must be fit on the training split only — implemented as a fitted
`TypeGroupedZScorer` transformer in `src/features.py`, not as a
stateless function, to keep the fit/transform boundary explicit.

## 8. Temporal structure

No timestamp column exists; `Product ID` never repeats, so the dataset is
a cross-sectional snapshot, not a repeated-measures time series.
**Rolling/time-window features are not justified** on this raw file (see
notebook Section 8 for the full reasoning and the recommendation for how
a streaming demo should be built instead).

## 9. Caveat

AI4I 2020 is a **synthetic benchmark dataset**. All statistical findings
above describe this benchmark's generative logic and should not be
presented as claims about real industrial machine reliability.
