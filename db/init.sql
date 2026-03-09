-- MovementScreen initial schema
-- Run with: psql -U postgres -d movementscreen -f db/init.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE users (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    name          VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE assessments (
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

CREATE TABLE findings (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID         NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    severity      VARCHAR(20)  NOT NULL,
    description   TEXT         NOT NULL,
    metric_value  NUMERIC(8,2),
    metric_label  VARCHAR(255)
);

CREATE TABLE angle_stats (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID         NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    name          VARCHAR(255) NOT NULL,
    min_value     NUMERIC(8,2),
    max_value     NUMERIC(8,2),
    mean_value    NUMERIC(8,2)
);

CREATE INDEX idx_assessments_user_id    ON assessments(user_id);
CREATE INDEX idx_assessments_recorded_at ON assessments(recorded_at);
CREATE INDEX idx_findings_assessment_id  ON findings(assessment_id);
CREATE INDEX idx_angle_stats_assessment_id ON angle_stats(assessment_id);
