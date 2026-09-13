# Data Dictionary — AI4I 2020 Predictive Maintenance

## 1. Source Dataset

**Source:** UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset
**Rows:** 10,000 | **Columns:** 14 | **Type:** Fully synthetic, cross-sectional (one row per machine, no repeated time-series readings)

⚠️ **Critical assumption (per project risk checklist):** This dataset does **not** contain real
industrial sensor history. Each row is an independently-simulated machine observed once. There is
no timestamp column. Any "trend," "history," or "moving average" query in this project uses either
(a) fleet-wide cross-sectional aggregation, or (b) `Tool wear [min]` as an ordering proxy for
time-in-service, because tool wear is the one variable in this dataset that is monotonically
increasing over a real tool's life. No calendar dates were fabricated. Do not present results from
this dataset as evidence of real industrial reliability.

### Raw columns

| Source Column | Type | Description | Range / Notes |
|---|---|---|---|
| UDI | int | Unique row identifier | 1–10000, unique |
| Product ID | string | Product serial, prefixed by quality variant (L/M/H) | e.g. `M14860` |
| Type | categorical | Product quality variant | `L` (Low, 60%), `M` (Medium, 30%), `H` (High, 10%) |
| Air temperature [K] | float | Ambient air temperature | ~295–304 K |
| Process temperature [K] | float | Process/machine temperature | ~305–314 K, correlated with air temp |
| Rotational speed [rpm] | int | Spindle rotational speed | ~1160–2900 rpm |
| Torque [Nm] | float | Applied torque | ~3.8–76.6 Nm |
| Tool wear [min] | int | Cumulative tool-wear minutes | 0–253 min |
| Machine failure | binary | Overall failure flag (label) | 1 = failed (339 of 10000, 3.39%) |
| TWF | binary | Tool Wear Failure flag | Sub-mode of Machine failure |
| HDF | binary | Heat Dissipation Failure flag | Sub-mode |
| PWF | binary | Power Failure flag | Sub-mode |
| OSF | binary | Overstrain Failure flag | Sub-mode |
| RNF | binary | Random Failure flag (unpredictable) | Sub-mode |

### Known data-quality quirk (preserved, not "fixed")

27 rows have `Machine failure` inconsistent with the five specific failure-mode flags:
- 18 rows: a specific mode flag = 1 but `Machine failure` = 0
- 9 rows: `Machine failure` = 1 but no specific mode flag is set

This is a documented characteristic of the AI4I2020 dataset, not a data-entry error we introduced.
We preserve it as ground truth in `failure_events` (via the `UNSPECIFIED` failure mode code and the
`is_overall_failure` column) rather than silently overwriting labels, since the downstream ML team
needs the true labels, warts and all.

---

## 2. Cleaned / Normalized Schema (MySQL, `predictive_maintenance` database)

### `machine_type` (lookup)
| Column | Type | Description |
|---|---|---|
| type_code | CHAR(1) PK | `L` / `M` / `H` |
| type_name | VARCHAR(20) | Low / Medium / High |
| description | VARCHAR(255) | Free text |

### `machines` (dimension, 10,000 rows)
| Column | Type | Description |
|---|---|---|
| machine_id | INT PK | = source UDI |
| product_id | VARCHAR(20) UNIQUE | = source Product ID |
| type_code | CHAR(1) FK → machine_type | Quality variant |

### `sensor_readings` (fact, 10,000 rows — 1:1 with machines in this snapshot)
| Column | Type | Description |
|---|---|---|
| reading_id | INT PK | = machine_id in this snapshot (kept as a separate key so the table can hold >1 reading per machine if real time-series data is added later) |
| machine_id | INT FK → machines | |
| air_temperature_k | DECIMAL(6,2) | |
| process_temperature_k | DECIMAL(6,2) | |
| rotational_speed_rpm | INT | |
| torque_nm | DECIMAL(6,2) | |
| tool_wear_min | INT | Used as the time-in-service ordering proxy |

### `failure_mode` (lookup)
| failure_mode_code | description |
|---|---|
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |
| UNSPECIFIED | `Machine failure`=1 with no specific mode attributed in source data |

### `failure_events` (fact, long format — 382 rows)
Normalizes the wide TWF/HDF/PWF/OSF/RNF/Machine failure columns into one row per
(reading, failure_mode) where the flag fired. This avoids sparse wide binary columns and makes
"how many failure modes did this machine trigger" a simple `GROUP BY ... HAVING COUNT(*) > 1`.

| Column | Type | Description |
|---|---|---|
| event_id | INT PK AUTO_INCREMENT | |
| reading_id | INT FK → sensor_readings | |
| machine_id | INT FK → machines | |
| failure_mode_code | VARCHAR(20) FK → failure_mode | |
| is_overall_failure | TINYINT(1) | Mirrors source `Machine failure` for this reading |

### `maintenance_events` (fact, **synthesized** — 382 rows)
⚠️ **Synthetic data, not real maintenance history.** The source dataset contains no maintenance
log. One record is generated per `failure_events` row to simulate a CMMS (maintenance management
system) entry logged at the point of failure detection, so the schema and SQL queries have
something realistic to join against for the monitoring/dashboard workstreams. `is_synthetic = 1`
flags every row so downstream consumers never mistake this for real maintenance history.

| Column | Type | Description |
|---|---|---|
| maintenance_id | INT PK AUTO_INCREMENT | |
| machine_id | INT FK → machines | |
| event_id | INT FK → failure_events (nullable) | |
| maintenance_type | ENUM('corrective','preventive') | Always 'corrective' currently (generated at failure time) |
| tool_wear_at_event | INT | Tool wear at time of synthesized maintenance |
| is_synthetic | TINYINT(1) | Always 1 — flags this row as simulated, not real |
| notes | VARCHAR(255) | |

---

## 3. Feature-Ready Output for ML Team

`data/ml_feature_ready.csv` and `.npy` — one row per machine, leakage-safe (no post-failure or
failure-mode columns used as predictors, only `machine_failure` as the target):

| Feature | Description |
|---|---|
| air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min | Raw process variables |
| temp_diff_k | process_temperature_k − air_temperature_k |
| mechanical_power_w | torque_nm × rotational_speed_rpm × (2π/60) — angular mechanical power |
| type_H, type_L, type_M | One-hot encoded quality variant |
| machine_failure | **Target label** |

---

## 4. Assumptions Log

1. No real timestamps exist in the source data; none were fabricated. `tool_wear_min` is used as
   the sole "time-in-service" ordering axis where a window function genuinely needs one.
2. `sensor_readings` is modeled 1:1 with `machines` today but kept as a separate table so the
   schema can absorb repeated/streaming readings without a redesign, if a real time-series
   extension (per the project's optional UCI time-series dataset) is added later.
3. `maintenance_events` is entirely synthetic (generated post-failure) — it exists to give the
   schema/queries a maintenance table to demonstrate against, not as a real maintenance record.
4. The 27-row label inconsistency between `Machine failure` and the specific mode flags is
   preserved as-is (ground truth), not corrected.
5. Cost figures used in the business-simulation query (`sql/02_analytical_queries.sql`, Q12) are
   illustrative placeholders (`$5,000` per failure), not real cost data — must be replaced with
   real or reasonably justified cost assumptions before Deliverable 10 (business simulation).
