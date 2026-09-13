-- =====================================================================
-- Predictive Maintenance Database Schema (MySQL / MariaDB 10.11+ compatible)
-- Project 5: Industrial Predictive Maintenance & Failure Prevention
-- Role: Data Engineer & SQL Database Lead
--
-- DESIGN NOTE / DATA ASSUMPTION (documented per project risk checklist):
-- The AI4I 2020 source dataset is a CROSS-SECTIONAL snapshot: each row
-- (UDI) represents ONE independent synthetic machine observed ONCE, not
-- repeated time-series readings of the same physical machine. There is
-- no wall-clock timestamp in the source data.
--
-- To keep the schema honest (no fabricated calendar timestamps), we:
--   1. Model `machines` and `sensor_readings` as separate tables (1:1 in
--      this dataset today, but the design supports 1:many if a real
--      sensor stream / repeated readings per machine is added later,
--      e.g. from the optional UCI time-series extension).
--   2. Use `tool_wear_min` as the natural, monotonically-increasing
--      "time-in-service" proxy for a given tool/machine when window
--      functions need an ORDER BY axis (tool wear only accumulates
--      forward in time for a physical cutting tool).
--   3. Treat `maintenance_events` as a SYNTHESIZED/DERIVED log (a
--      corrective-maintenance record is generated whenever a failure is
--      observed). This is clearly a simulation of what a real CMMS
--      (maintenance management system) log would look like, not real
--      maintenance history. This is flagged again in the data dictionary.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS predictive_maintenance CHARACTER SET utf8mb4;
USE predictive_maintenance;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS maintenance_events;
DROP TABLE IF EXISTS failure_events;
DROP TABLE IF EXISTS sensor_readings;
DROP TABLE IF EXISTS machines;
DROP TABLE IF EXISTS failure_mode;
DROP TABLE IF EXISTS machine_type;
SET FOREIGN_KEY_CHECKS = 1;

-- ---------------------------------------------------------------------
-- 1. machine_type: lookup/dimension table for the product quality tier
--    (L = Low, M = Medium, H = High), decoded from the Product ID prefix.
-- ---------------------------------------------------------------------
CREATE TABLE machine_type (
    type_code   CHAR(1)      NOT NULL PRIMARY KEY,
    type_name   VARCHAR(20)  NOT NULL,
    description VARCHAR(255) NULL
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 2. machines: one row per physical/synthetic machine (dimension table).
-- ---------------------------------------------------------------------
CREATE TABLE machines (
    machine_id  INT          NOT NULL PRIMARY KEY,      -- source: UDI
    product_id  VARCHAR(20)  NOT NULL UNIQUE,            -- source: Product ID
    type_code   CHAR(1)      NOT NULL,
    CONSTRAINT fk_machines_type
        FOREIGN KEY (type_code) REFERENCES machine_type(type_code)
) ENGINE=InnoDB;

CREATE INDEX idx_machines_type ON machines(type_code);

-- ---------------------------------------------------------------------
-- 3. sensor_readings: fact table of operating-condition measurements.
--    1:1 with machines in the current snapshot dataset, modeled as a
--    separate table (FK, not merged into `machines`) so the schema
--    scales to repeated/streaming readings without redesign.
-- ---------------------------------------------------------------------
CREATE TABLE sensor_readings (
    reading_id               INT           NOT NULL PRIMARY KEY,  -- source: UDI
    machine_id               INT           NOT NULL,
    air_temperature_k        DECIMAL(6,2)  NOT NULL,
    process_temperature_k    DECIMAL(6,2)  NOT NULL,
    rotational_speed_rpm     INT           NOT NULL,
    torque_nm                DECIMAL(6,2)  NOT NULL,
    tool_wear_min            INT           NOT NULL,
    CONSTRAINT fk_readings_machine
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
    CONSTRAINT chk_air_temp CHECK (air_temperature_k > 0),
    CONSTRAINT chk_process_temp CHECK (process_temperature_k > 0),
    CONSTRAINT chk_rot_speed CHECK (rotational_speed_rpm >= 0),
    CONSTRAINT chk_torque CHECK (torque_nm >= 0),
    CONSTRAINT chk_tool_wear CHECK (tool_wear_min >= 0)
) ENGINE=InnoDB;

CREATE INDEX idx_readings_machine ON sensor_readings(machine_id);
CREATE INDEX idx_readings_toolwear ON sensor_readings(tool_wear_min);

-- ---------------------------------------------------------------------
-- 4. failure_mode: lookup table decoding the five specific failure
--    mechanisms plus a placeholder for unattributed overall failures.
-- ---------------------------------------------------------------------
CREATE TABLE failure_mode (
    failure_mode_code VARCHAR(20) NOT NULL PRIMARY KEY,
    description        VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 5. failure_events: long-format normalization of the wide binary flag
--    columns (TWF, HDF, PWF, OSF, RNF, Machine failure) in the source
--    data. One row per (reading, failure_mode) where the flag = 1.
--    A small number of source rows (documented in the data dictionary)
--    have `Machine failure = 1` without any specific mode flag set, or
--    a mode flag set without `Machine failure = 1`, these are preserved
--    as-is using the 'UNSPECIFIED' / is_overall_failure columns rather
--    than silently "corrected", since they are part of the ground truth.
-- ---------------------------------------------------------------------
CREATE TABLE failure_events (
    event_id           INT AUTO_INCREMENT PRIMARY KEY,
    reading_id          INT NOT NULL,
    machine_id          INT NOT NULL,
    failure_mode_code   VARCHAR(20) NOT NULL,
    is_overall_failure  TINYINT(1) NOT NULL DEFAULT 0,  -- mirrors source `Machine failure` for this reading
    CONSTRAINT fk_failure_reading
        FOREIGN KEY (reading_id) REFERENCES sensor_readings(reading_id),
    CONSTRAINT fk_failure_machine
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
    CONSTRAINT fk_failure_mode
        FOREIGN KEY (failure_mode_code) REFERENCES failure_mode(failure_mode_code)
) ENGINE=InnoDB;

CREATE INDEX idx_failure_machine ON failure_events(machine_id);
CREATE INDEX idx_failure_mode ON failure_events(failure_mode_code);

-- ---------------------------------------------------------------------
-- 6. maintenance_events: SYNTHESIZED corrective-maintenance log.
--    One record is generated per failure_event to simulate what a real
--    Computerized Maintenance Management System (CMMS) would log when
--    a failure is detected. `tool_wear_at_event` reuses the reading's
--    tool wear as the "time-in-service" marker for the maintenance
--    action, since no real maintenance history exists in the source
--    data. This table exists to satisfy the schema requirement and to
--    give the Streamlit/monitoring workstream something to query, it
--    must not be presented as real maintenance history.
-- ---------------------------------------------------------------------
CREATE TABLE maintenance_events (
    maintenance_id       INT AUTO_INCREMENT PRIMARY KEY,
    machine_id           INT NOT NULL,
    event_id             INT NULL,
    maintenance_type     ENUM('corrective','preventive') NOT NULL DEFAULT 'corrective',
    tool_wear_at_event    INT NOT NULL,
    is_synthetic         TINYINT(1) NOT NULL DEFAULT 1,
    notes                VARCHAR(255) NULL,
    CONSTRAINT fk_maint_machine
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id),
    CONSTRAINT fk_maint_event
        FOREIGN KEY (event_id) REFERENCES failure_events(event_id)
) ENGINE=InnoDB;

CREATE INDEX idx_maint_machine ON maintenance_events(machine_id);
