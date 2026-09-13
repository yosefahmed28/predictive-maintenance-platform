-- =====================================================================
-- Analytical SQL Queries — Predictive Maintenance
-- Role: Data Engineer & SQL Database Lead
-- Engine: MySQL / MariaDB 10.11+ (window functions, CTEs)
--
-- NOTE ON "HISTORY": AI4I2020 is a cross-sectional snapshot (one row
-- per machine), not a longitudinal time series. Where the project asks
-- to "aggregate sensor history" / "track equipment metrics", these
-- queries interpret that as fleet-wide cross-sectional aggregation,
-- using tool_wear_min as an ordering proxy for "time-in-service" where
-- a temporal axis is genuinely needed for a window function. This
-- interpretation is documented in docs/data_dictionary.md.
-- =====================================================================
USE predictive_maintenance;

-- ---------------------------------------------------------------------
-- Q1. Failure rate by machine quality Type, with each Type's share of
--     total fleet failures (JOIN + window function).
-- ---------------------------------------------------------------------
SELECT
    mt.type_name,
    COUNT(*) AS machine_count,
    SUM(sr_failure.machine_failure) AS failures,
    ROUND(100.0 * SUM(sr_failure.machine_failure) / COUNT(*), 2) AS failure_rate_pct,
    ROUND(100.0 * SUM(sr_failure.machine_failure)
          / SUM(SUM(sr_failure.machine_failure)) OVER (), 2) AS pct_of_all_failures
FROM machines m
JOIN machine_type mt ON mt.type_code = m.type_code
JOIN (
    SELECT sr.machine_id, sr.reading_id,
           MAX(CASE WHEN fe.reading_id IS NOT NULL THEN 1 ELSE 0 END) AS machine_failure
    FROM sensor_readings sr
    LEFT JOIN failure_events fe ON fe.reading_id = sr.reading_id AND fe.is_overall_failure = 1
    GROUP BY sr.machine_id, sr.reading_id
) sr_failure ON sr_failure.machine_id = m.machine_id
GROUP BY mt.type_name;

-- ---------------------------------------------------------------------
-- Q2. Rank machines within each Type by torque (window function: RANK).
-- ---------------------------------------------------------------------
SELECT
    m.machine_id, m.product_id, mt.type_name, sr.torque_nm,
    RANK() OVER (PARTITION BY mt.type_code ORDER BY sr.torque_nm DESC) AS torque_rank_in_type
FROM machines m
JOIN machine_type mt ON mt.type_code = m.type_code
JOIN sensor_readings sr ON sr.machine_id = m.machine_id
ORDER BY mt.type_code, torque_rank_in_type
LIMIT 30;

-- ---------------------------------------------------------------------
-- Q3. CTE + window functions: flag readings whose sensor values are
--     statistical outliers (>2 std dev) relative to their own Type's
--     mean — candidate anomalies for the anomaly-detection workstream.
-- ---------------------------------------------------------------------
WITH type_stats AS (
    SELECT
        sr.reading_id, m.machine_id, mt.type_name, sr.torque_nm,
        AVG(sr.torque_nm) OVER (PARTITION BY mt.type_code) AS type_avg_torque,
        STDDEV(sr.torque_nm) OVER (PARTITION BY mt.type_code) AS type_std_torque
    FROM sensor_readings sr
    JOIN machines m ON m.machine_id = sr.machine_id
    JOIN machine_type mt ON mt.type_code = m.type_code
)
SELECT
    machine_id, type_name, torque_nm, ROUND(type_avg_torque, 2) AS type_avg_torque,
    ROUND((torque_nm - type_avg_torque) / NULLIF(type_std_torque, 0), 2) AS torque_zscore
FROM type_stats
WHERE ABS((torque_nm - type_avg_torque) / NULLIF(type_std_torque, 0)) > 2
ORDER BY ABS((torque_nm - type_avg_torque) / NULLIF(type_std_torque, 0)) DESC
LIMIT 25;

-- ---------------------------------------------------------------------
-- Q4. Top 15 highest-risk machines fleet-wide via a composite mechanical
--     stress score, using PERCENT_RANK() (window function).
-- ---------------------------------------------------------------------
SELECT machine_id, product_id, type_name, torque_nm, rotational_speed_rpm,
       tool_wear_min, risk_percentile
FROM (
    SELECT
        m.machine_id, m.product_id, mt.type_name,
        sr.torque_nm, sr.rotational_speed_rpm, sr.tool_wear_min,
        PERCENT_RANK() OVER (
            ORDER BY sr.torque_nm * sr.tool_wear_min DESC
        ) AS risk_percentile
    FROM machines m
    JOIN machine_type mt ON mt.type_code = m.type_code
    JOIN sensor_readings sr ON sr.machine_id = m.machine_id
) ranked
ORDER BY risk_percentile ASC
LIMIT 15;

-- ---------------------------------------------------------------------
-- Q5. Failure-mode breakdown with running cumulative share of failures
--     (CTE + window SUM() OVER ORDER BY).
-- ---------------------------------------------------------------------
WITH mode_counts AS (
    SELECT fm.failure_mode_code, fm.description, COUNT(*) AS n
    FROM failure_events fe
    JOIN failure_mode fm ON fm.failure_mode_code = fe.failure_mode_code
    WHERE fe.failure_mode_code <> 'UNSPECIFIED'
    GROUP BY fm.failure_mode_code, fm.description
)
SELECT
    failure_mode_code, description, n,
    ROUND(100.0 * n / SUM(n) OVER (), 2) AS pct_of_mode_failures,
    ROUND(100.0 * SUM(n) OVER (ORDER BY n DESC) / SUM(n) OVER (), 2) AS running_cumulative_pct
FROM mode_counts
ORDER BY n DESC;

-- ---------------------------------------------------------------------
-- Q6. Tool-wear quartile analysis: bucket machines into wear quartiles
--     (NTILE window function) and compute failure rate per bucket, per
--     Type (CTE + GROUP BY).
-- ---------------------------------------------------------------------
WITH wear_buckets AS (
    SELECT
        sr.machine_id, m.type_code, sr.tool_wear_min,
        NTILE(4) OVER (PARTITION BY m.type_code ORDER BY sr.tool_wear_min) AS wear_quartile
    FROM sensor_readings sr
    JOIN machines m ON m.machine_id = sr.machine_id
),
failures AS (
    SELECT DISTINCT machine_id FROM failure_events WHERE is_overall_failure = 1
)
SELECT
    wb.type_code, wb.wear_quartile,
    COUNT(*) AS machine_count,
    SUM(CASE WHEN f.machine_id IS NOT NULL THEN 1 ELSE 0 END) AS failures,
    ROUND(100.0 * SUM(CASE WHEN f.machine_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_rate_pct
FROM wear_buckets wb
LEFT JOIN failures f ON f.machine_id = wb.machine_id
GROUP BY wb.type_code, wb.wear_quartile
ORDER BY wb.type_code, wb.wear_quartile;

-- ---------------------------------------------------------------------
-- Q7. Multi-mode failures: machines that triggered more than one
--     specific failure mode simultaneously (GROUP BY ... HAVING).
-- ---------------------------------------------------------------------
SELECT
    m.machine_id, m.product_id, mt.type_name,
    COUNT(*) AS distinct_modes_triggered,
    GROUP_CONCAT(fe.failure_mode_code ORDER BY fe.failure_mode_code) AS modes
FROM failure_events fe
JOIN machines m ON m.machine_id = fe.machine_id
JOIN machine_type mt ON mt.type_code = m.type_code
WHERE fe.failure_mode_code <> 'UNSPECIFIED'
GROUP BY m.machine_id, m.product_id, mt.type_name
HAVING COUNT(*) > 1
ORDER BY distinct_modes_triggered DESC;

-- ---------------------------------------------------------------------
-- Q8. Maintenance history per machine: sequence maintenance events per
--     machine and compute the "gap" in tool-wear minutes between
--     consecutive maintenance actions (LAG window function).
-- ---------------------------------------------------------------------
SELECT
    me.machine_id,
    me.maintenance_id,
    me.maintenance_type,
    me.tool_wear_at_event,
    ROW_NUMBER() OVER (PARTITION BY me.machine_id ORDER BY me.maintenance_id) AS maint_sequence,
    LAG(me.tool_wear_at_event) OVER (PARTITION BY me.machine_id ORDER BY me.maintenance_id) AS prev_tool_wear,
    me.tool_wear_at_event
        - LAG(me.tool_wear_at_event) OVER (PARTITION BY me.machine_id ORDER BY me.maintenance_id) AS wear_minutes_since_last_maint
FROM maintenance_events me
ORDER BY me.machine_id, maint_sequence;

-- ---------------------------------------------------------------------
-- Q9. Moving average of torque ordered by tool wear, per Type — a
--     smoothed trend line for the monitoring dashboard (window frame:
--     ROWS BETWEEN ... PRECEDING).
-- ---------------------------------------------------------------------
SELECT
    m.machine_id, mt.type_code, sr.tool_wear_min, sr.torque_nm,
    ROUND(AVG(sr.torque_nm) OVER (
        PARTITION BY mt.type_code
        ORDER BY sr.tool_wear_min
        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
    ), 2) AS torque_moving_avg_20
FROM sensor_readings sr
JOIN machines m ON m.machine_id = sr.machine_id
JOIN machine_type mt ON mt.type_code = m.type_code
ORDER BY mt.type_code, sr.tool_wear_min
LIMIT 50;

-- ---------------------------------------------------------------------
-- Q10. Multi-sensor anomaly flag: CTE chain computing per-Type z-scores
--      for THREE sensor variables at once, then flagging any reading
--      where any one variable exceeds 2.5 std devs (compound anomaly
--      rule feeding the anomaly-detection / drift-monitoring track).
-- ---------------------------------------------------------------------
WITH stats AS (
    SELECT
        sr.reading_id, m.machine_id, mt.type_code,
        sr.air_temperature_k, sr.process_temperature_k, sr.rotational_speed_rpm,
        AVG(sr.air_temperature_k) OVER (PARTITION BY mt.type_code) AS avg_air,
        STDDEV(sr.air_temperature_k) OVER (PARTITION BY mt.type_code) AS std_air,
        AVG(sr.process_temperature_k) OVER (PARTITION BY mt.type_code) AS avg_proc,
        STDDEV(sr.process_temperature_k) OVER (PARTITION BY mt.type_code) AS std_proc,
        AVG(sr.rotational_speed_rpm) OVER (PARTITION BY mt.type_code) AS avg_rpm,
        STDDEV(sr.rotational_speed_rpm) OVER (PARTITION BY mt.type_code) AS std_rpm
    FROM sensor_readings sr
    JOIN machines m ON m.machine_id = sr.machine_id
    JOIN machine_type mt ON mt.type_code = m.type_code
),
zscores AS (
    SELECT
        machine_id, type_code,
        (air_temperature_k - avg_air) / NULLIF(std_air, 0) AS z_air,
        (process_temperature_k - avg_proc) / NULLIF(std_proc, 0) AS z_proc,
        (rotational_speed_rpm - avg_rpm) / NULLIF(std_rpm, 0) AS z_rpm
    FROM stats
)
SELECT machine_id, type_code, ROUND(z_air,2) z_air, ROUND(z_proc,2) z_proc, ROUND(z_rpm,2) z_rpm
FROM zscores
WHERE ABS(z_air) > 2.5 OR ABS(z_proc) > 2.5 OR ABS(z_rpm) > 2.5
ORDER BY GREATEST(ABS(z_air), ABS(z_proc), ABS(z_rpm)) DESC;

-- ---------------------------------------------------------------------
-- Q11. Machines whose torque is above the 90th percentile within their
--      own Type (window function: PERCENT_RANK, filtered in outer query).
-- ---------------------------------------------------------------------
SELECT machine_id, type_code, torque_nm, pct_rank
FROM (
    SELECT
        m.machine_id, mt.type_code, sr.torque_nm,
        PERCENT_RANK() OVER (PARTITION BY mt.type_code ORDER BY sr.torque_nm) AS pct_rank
    FROM sensor_readings sr
    JOIN machines m ON m.machine_id = sr.machine_id
    JOIN machine_type mt ON mt.type_code = m.type_code
) t
WHERE pct_rank >= 0.90
ORDER BY type_code, torque_nm DESC;

-- ---------------------------------------------------------------------
-- Q12. Business-simulation feeder query: failure counts and a simple
--      placeholder cost estimate by Type and failure mode, for the
--      "cost of missed failures vs. unnecessary maintenance" business
--      simulation (downstream MLOps/business-analysis workstream).
--      Cost constants are illustrative placeholders — see docs.
-- ---------------------------------------------------------------------
WITH failure_costs AS (
    SELECT
        mt.type_name,
        fe.failure_mode_code,
        COUNT(*) AS failure_count
    FROM failure_events fe
    JOIN machines m ON m.machine_id = fe.machine_id
    JOIN machine_type mt ON mt.type_code = m.type_code
    WHERE fe.failure_mode_code <> 'UNSPECIFIED'
    GROUP BY mt.type_name, fe.failure_mode_code
)
SELECT
    type_name,
    failure_mode_code,
    failure_count,
    failure_count * 5000 AS estimated_downtime_cost_usd_placeholder,  -- ILLUSTRATIVE ONLY
    ROUND(100.0 * failure_count / SUM(failure_count) OVER (PARTITION BY type_name), 2) AS pct_of_type_failures
FROM failure_costs
ORDER BY type_name, failure_count DESC;
