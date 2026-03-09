-- Run this against an existing database that was created from the original init.sql.
-- Safe to run multiple times (uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
--
--   psql -U postgres -d movementscreen -f db/migrate_existing.sql

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS threshold_config (
    key         VARCHAR(100) PRIMARY KEY,
    value       DOUBLE PRECISION NOT NULL,
    description TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
