"""
api/app.py
-----------
DIPEX Enterprise Analytics Platform — FastAPI application.

Production-grade API with:
  - JWT authentication (RBAC: VIEWER/ANALYST/ADMIN roles)
  - Rate limiting middleware
  - CORS (configurable origins via env)
  - Prometheus /prom-metrics endpoint (scrape target for Grafana)
  - /health: status|db_ok|model_registry_ok|uptime
  - /metrics: operational metrics (runs, pass rate, retry count, LLM tokens)
  - 18 route modules registered (incl. analyst tier endpoints and unified /api/pipeline/run)
  - Dashboard static file serving
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from api.metrics import get_metrics_response
from api.routes import (
    analyst_ops, analyst_tiers, audit, cohort, drift, feedback, governance,
    ingest, ingest_v2, model, pipeline_run, preprocess, query, report,
    results, run, stats,
)
from api.routes import auth as auth_router
from auth.jwt_auth import get_current_user
from middleware.rate_limiter import RateLimiterMiddleware

# ── Startup timestamp ─────────────────────────────────────────────────────────
_START_TIME: float = time.time()

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DIPEX Enterprise Analytics Platform",
    description=(
        "Deterministic-first, statistically disciplined, AI-assisted, "
        "governance-enforced analytics operating system. Automates the complete "
        "cognitive workflow of a professional data analyst — from ingestion to "
        "executive reporting."
    ),
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGINS", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(
    RateLimiterMiddleware,
    requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "120")),
    burst=int(os.getenv("RATE_LIMIT_BURST", "20")),
)

# ── Routes ────────────────────────────────────────────────────────────────────
_protected = [Depends(get_current_user)]

# Core pipeline
app.include_router(ingest.router,    dependencies=_protected)
app.include_router(run.router,       dependencies=_protected)
app.include_router(results.router,   dependencies=_protected)
app.include_router(audit.router,     dependencies=_protected)
app.include_router(feedback.router,  dependencies=_protected)

# Auth (no auth guard on the auth router itself)
app.include_router(auth_router.router)

# Enterprise analytics
app.include_router(preprocess.router, dependencies=_protected)
app.include_router(query.router,      dependencies=_protected)
app.include_router(stats.router,      dependencies=_protected)
app.include_router(model.router,      dependencies=_protected)
app.include_router(report.router,     dependencies=_protected)
app.include_router(governance.router, dependencies=_protected)

# Advanced analytics
app.include_router(drift.router,  dependencies=_protected)
app.include_router(cohort.router, dependencies=_protected)

# Universal Data Intake Layer v2
app.include_router(ingest_v2.router, dependencies=_protected)

# Unified ingest + pipeline endpoint
app.include_router(pipeline_run.router, dependencies=_protected)

# Analyst Intelligence Automation (10 endpoints)
app.include_router(analyst_ops.router, dependencies=_protected)

# Analyst Tier Automation — Phase 15 (junior), 16 (mid), 17 (senior)  — 20 endpoints
app.include_router(analyst_tiers.router, dependencies=_protected)

# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """API root — lists capabilities and links."""
    return {
        "name":    "DIPEX Enterprise Analytics Platform",
        "version": "3.1.0",
        "status":  "operational",
        "capabilities": [
            "data_ingestion", "preprocessing", "sql_analytics",
            "statistical_analysis", "hypothesis_testing", "drift_detection",
            "ml_modeling", "cohort_analysis", "executive_reporting",
            "governance_enforcement", "audit_logging",
            "analyst_intelligence", "rl_optimization", "streaming_ingestion",
            "junior_analyst_automation", "mid_analyst_automation",
            "senior_analyst_automation", "feedback_retry_controller",
            "insight_ranking", "feature_proposals", "anomaly_flagging",
        ],
        "docs":      "/docs",
        "dashboard": "/dashboard",
    }


@app.get("/health", tags=["System"])
async def health():
    """
    Production health check.

    Returns:
        status            : healthy | degraded | unhealthy
        version           : semantic version string
        uptime            : seconds since process start (float)
        db_ok             : DuckDB in-process connectivity (bool)
        model_registry_ok : model registry directory readable (bool)
        timestamp         : ISO-8601 UTC
    """
    uptime_seconds = round(time.time() - _START_TIME, 2)

    # DuckDB in-process probe (doesn't require a remote DB)
    db_ok = False
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        con.execute("SELECT 1").fetchone()
        con.close()
        db_ok = True
    except Exception:
        db_ok = False

    # Model registry readability
    registry_path = os.getenv("MODEL_REGISTRY_PATH", "data/model_registry")
    model_registry_ok = os.path.isdir(registry_path) and os.access(registry_path, os.R_OK)

    if db_ok and model_registry_ok:
        status = "healthy"
    elif db_ok or model_registry_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status":            status,
        "version":           "3.0.0",
        "uptime":            uptime_seconds,
        "db_ok":             db_ok,
        "model_registry_ok": model_registry_ok,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


@app.get("/prom-metrics", tags=["System"], include_in_schema=False)
async def prom_metrics():
    """Prometheus scrape endpoint — text/plain exposition format."""
    body, content_type = get_metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/metrics", tags=["System"])
async def metrics():
    """Operational metrics — pipeline stats, approvals, LLM usage, uptime."""
    audit_log_path = "audit/audit.jsonl"
    total_runs = passed_runs = retry_runs = 0

    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") == "PIPELINE_RUN":
                        total_runs += 1
                        if entry.get("gate_decision") == "PASS":
                            passed_runs += 1
                        if (entry.get("retry_count") or 0) > 0:
                            retry_runs += 1
                except Exception:
                    pass

    retry_log = "audit/retry_escalations.jsonl"
    retry_escalations = 0
    if os.path.exists(retry_log):
        with open(retry_log, "r", encoding="utf-8") as f:
            retry_escalations = sum(1 for ln in f if ln.strip())

    n_approved = (
        len(os.listdir("data/approved_outputs"))
        if os.path.isdir("data/approved_outputs") else 0
    )
    n_reports = (
        len([f for f in os.listdir("reports") if f.endswith(".html")])
        if os.path.isdir("reports") else 0
    )
    n_models = (
        len(os.listdir("data/model_registry"))
        if os.path.isdir("data/model_registry") else 0
    )

    llm_tokens = 0
    llm_log = "audit/llm_cost_log.jsonl"
    if os.path.exists(llm_log):
        with open(llm_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    llm_tokens += json.loads(line).get("total_tokens", 0)
                except Exception:
                    pass

    return {
        "total_pipeline_runs": total_runs,
        "passed_runs":         passed_runs,
        "pass_rate":           round(passed_runs / total_runs, 4) if total_runs else 0.0,
        "retry_runs":          retry_runs,
        "retry_escalations":   retry_escalations,
        "approved_outputs":    n_approved,
        "reports_generated":   n_reports,
        "models_in_registry":  n_models,
        "llm_tokens_consumed": llm_tokens,
        "uptime_seconds":      round(time.time() - _START_TIME, 2),
    }


# ── Dashboard static mount ────────────────────────────────────────────────────
if os.path.exists("dashboard"):
    app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")
