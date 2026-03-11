"""Idempotent schema initialiser — safe to run on every deploy.

Creates all tables and indexes if they don't already exist, so this script
can be called unconditionally at container startup without risk of data loss
or duplicate-table errors.

Usage (run from project root):
    python db/schema_init.py

The DATABASE_URL environment variable must be set.
"""
import asyncio
import os
import sys

import asyncpg

_DDL = """
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    name          VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_admin      BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS assessments (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    screen_type    VARCHAR(50) NOT NULL,
    camera_angle   VARCHAR(50) NOT NULL,
    lead_side      VARCHAR(10),
    frame_count    INTEGER     NOT NULL,
    worst_severity VARCHAR(20) NOT NULL,
    has_findings   BOOLEAN     NOT NULL,
    recorded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS findings (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID         NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    severity      VARCHAR(20)  NOT NULL,
    description   TEXT         NOT NULL,
    metric_value  NUMERIC(10,4),
    metric_label  VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS angle_stats (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID         NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    min_value     NUMERIC(10,4),
    max_value     NUMERIC(10,4),
    mean_value    NUMERIC(10,4)
);

CREATE TABLE IF NOT EXISTS threshold_config (
    key         VARCHAR(100) PRIMARY KEY,
    value       DOUBLE PRECISION NOT NULL,
    description TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assessments_user_id       ON assessments(user_id);
CREATE INDEX IF NOT EXISTS idx_assessments_recorded_at   ON assessments(recorded_at);
CREATE INDEX IF NOT EXISTS idx_findings_assessment_id    ON findings(assessment_id);
CREATE INDEX IF NOT EXISTS idx_angle_stats_assessment_id ON angle_stats(assessment_id);
"""


async def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("Connecting to database...")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(_DDL)
        print("Schema initialised successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
