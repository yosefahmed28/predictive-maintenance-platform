-- =====================================================================
-- 00_local_setup.sql
-- RUN THIS FIRST, ONCE, connected as your MySQL 'root' (or admin) user.
--
-- How to run it (pick one):
--   A) MySQL Workbench: open this file, connect as root, click Execute.
--   B) Command line (from this project folder):
--        mysql -u root -p < sql\00_local_setup.sql        (Windows)
--        mysql -u root -p < sql/00_local_setup.sql        (Mac/Linux)
--
-- This creates:
--   - the `predictive_maintenance` database
--   - an application user `pm_user` with password `pm_pass123`
--     (change the password below if you want a different one -
--      just also update DB_PASSWORD in the notebook's config cell)
-- =====================================================================

CREATE DATABASE IF NOT EXISTS predictive_maintenance CHARACTER SET utf8mb4;

CREATE USER IF NOT EXISTS 'pm_user'@'localhost' IDENTIFIED BY 'pm_pass123';

GRANT ALL PRIVILEGES ON predictive_maintenance.* TO 'pm_user'@'localhost';

FLUSH PRIVILEGES;

-- Sanity check
SELECT User, Host FROM mysql.user WHERE User = 'pm_user';
