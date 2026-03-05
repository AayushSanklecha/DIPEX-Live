"""
api/app.py
-----------
Simplified Analytics Platform — Core workflow only.

Features:
  1. Upload dataset
  2. Clean data (preprocessing)
  3. Run EDA automatically
  4. Detect anomalies
  5. Generate charts/dashboards
  6. Generate simple report
"""

from __future__ import annotations

# Load .env FIRST — before any other module reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to system env vars

import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from api.metrics import get_metrics_response
from api.routes import (
    audit, ingest, ingest_v2, pipeline_run, preprocess,
    report, results, run, stats, analyst, cohort, exports
)

# ── Startup timestamp ─────────────────────────────────────────────────────────
_START_TIME: float = time.time()

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DIPEX — Data Intelligence Platform for Expert Analysis",
    description=(
        "5-Layer Architecture: "
        "Data Source → Data Processing → QA/Governance → AI Analytics → Presentation"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Routes (No Authentication Required) ───────────────────────────────────────

# Core workflow
app.include_router(ingest.router)       # Upload dataset
app.include_router(preprocess.router)   # Clean data
app.include_router(stats.router)        # EDA & anomaly detection
app.include_router(report.router)       # Generate reports (/report/)
app.include_router(report.api_router)   # Reports API (/api/reports/)
app.include_router(results.router)      # View results
app.include_router(run.router)          # Run pipeline
app.include_router(audit.router)        # Audit logs
app.include_router(analyst.router)      # Analyst ops
app.include_router(cohort.router)       # Cohort analysis

# Universal Data Intake Layer
app.include_router(ingest_v2.router)

# Simplified pipeline endpoint
app.include_router(pipeline_run.router)

# PRESENTATION LAYER — Exports (CSV / JSON / Parquet / Report download)
app.include_router(exports.router)

# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """API root — lists workflow steps."""
    return {
        "name":    "DIPEX — Data Intelligence Platform for Expert Analysis",
        "version": "2.0.0",
        "status":  "operational",
        "architecture": [
            "Layer 1 — Data Source Layer (CSV | Excel | Database | API | Kafka)",
            "Layer 2 — Data Processing Layer (Ingestion | Normalization | Profiling | Streaming)",
            "Layer 3 — QA, Governance & Control Layer (Validation | Verifiers | Rules | Confidence | Audit)",
            "Layer 4 — AI & Analytics Service Layer (AutoEDA | FeatureEng | Insight Ranking | Retry | LLM)",
            "Layer 5 — Presentation Layer (Dashboards | Reports | APIs | Exports)",
        ],
        "docs":      "/docs",
        "dashboard": "/dashboard",
        "exports":   "/api/export/list",
    }


@app.get("/health", tags=["System"])
async def health():
    """Health check."""
    uptime_seconds = round(time.time() - _START_TIME, 2)

    # DuckDB check
    db_ok = False
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        con.execute("SELECT 1").fetchone()
        con.close()
        db_ok = True
    except Exception:
        db_ok = False

    status = "healthy" if db_ok else "degraded"

    return {
        "status":    status,
        "version":   "1.0.0",
        "uptime":    uptime_seconds,
        "db_ok":     db_ok,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/prom-metrics", tags=["System"], include_in_schema=False)
async def prom_metrics():
    """Prometheus scrape endpoint."""
    body, content_type = get_metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/metrics", tags=["System"])
async def metrics():
    """Operational metrics — pipeline stats and uptime."""
    audit_log_path = "audit/audit.jsonl"
    total_runs = passed_runs = 0

    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") == "PIPELINE_RUN":
                        total_runs += 1
                        if entry.get("gate_decision") == "PASS":
                            passed_runs += 1
                except Exception:
                    pass

    n_reports = (
        len([f for f in os.listdir("reports") if f.endswith(".html")])
        if os.path.isdir("reports") else 0
    )

    return {
        "total_pipeline_runs": total_runs,
        "passed_runs":         passed_runs,
        "pass_rate":           round(passed_runs / total_runs, 4) if total_runs else 0.0,
        "reports_generated":   n_reports,
        "uptime_seconds":      round(time.time() - _START_TIME, 2),
    }


# ── Dashboard static mount ────────────────────────────────────────────────────
if os.path.exists("dashboard"):
    app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")
