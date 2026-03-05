-- ─────────────────────────────────────────────────────────────────
-- DIPEX — PostgreSQL initialization script
-- Runs once on first container start
-- ─────────────────────────────────────────────────────────────────

-- Pipeline run results table
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            SERIAL PRIMARY KEY,
    run_id        VARCHAR(128) NOT NULL UNIQUE,
    source        VARCHAR(256),
    rows_ingested INTEGER,
    gate_decision VARCHAR(16),
    confidence    FLOAT,
    model_winner  VARCHAR(64),
    auc_score     FLOAT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Quality gate failures
CREATE TABLE IF NOT EXISTS gate_failures (
    id         SERIAL PRIMARY KEY,
    run_id     VARCHAR(128) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    column_name VARCHAR(128),
    severity   VARCHAR(16),
    message    TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ML proposal log
CREATE TABLE IF NOT EXISTS proposals (
    id            SERIAL PRIMARY KEY,
    run_id        VARCHAR(128) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    proposer_type VARCHAR(64),
    proposal_text TEXT,
    confidence    FLOAT,
    accepted      BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id         SERIAL PRIMARY KEY,
    run_id     VARCHAR(128),
    event_type VARCHAR(64),
    detail     JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_runs_created  ON pipeline_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_decision ON pipeline_runs(gate_decision);
CREATE INDEX IF NOT EXISTS idx_audit_run     ON audit_log(run_id);
