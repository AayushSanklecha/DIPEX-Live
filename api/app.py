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
import math
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Any

class NaNHandlingJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # FastAPI's default JSONResponse does not allow NaNs, or rather Python's json
        # allows it by default but it generates invalid strict JSON (e.g. `NaN`).
        # Or it raises ValueError if allow_nan=False. The Starlette JSONResponse 
        # doesn't handle allow_nan easily. We'll use json.dumps with handling.
        text = json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
        )
        # Convert non-compliant floats to null
        text = text.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
        return text.encode("utf-8")

from api.metrics import get_metrics_response
from api.routes import (
    audit, ingest, ingest_v2, pipeline_run, preprocess,
    report, results, run, stats, analyst, cohort, exports, explorer,
    feedback, analytics,
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
    default_response_class=NaNHandlingJSONResponse,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# Allow all origins so the app works on Hugging Face Spaces (*.hf.space),
# local dev, and any other host without needing explicit whitelisting.
# allow_credentials MUST be False when allow_origins=["*"] per the CORS spec;
# the browser will reject responses that set both.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ── API Key Authentication ────────────────────────────────────────────────────
from api.middleware.auth import APIKeyMiddleware
app.add_middleware(APIKeyMiddleware)

# ── Upload size limit ─────────────────────────────────────────────────────────
# This guard exists only to prevent truly runaway requests from exhausting memory.
# Kafka / DB / API source kinds send only a small JSON config (< 10 KB) so they
# will never hit this limit. File uploads can be large datasets — set to 5 GB.
# Uvicorn itself has no hard limit by default, so this is the only safeguard.
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

@app.middleware("http")
async def limit_upload_size(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large. Maximum allowed is {_MAX_UPLOAD_BYTES // (1024**3)} GB."},
        )
    return await call_next(request)

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

# DATA EXPLORER — Raw database preview
app.include_router(explorer.router)

# ANALYST INSTRUCTION LOOP — Satisfaction feedback + RL reward recording
app.include_router(feedback.router)

# ANALYTICS DASHBOARD — Power BI-style enriched analytical payload
app.include_router(analytics.router)

# ── System endpoints ──────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse



@app.get("/api/status", tags=["System"])
async def system_status():
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



@app.get("/health", tags=["System"], include_in_schema=False)
async def health_check():
    """Liveness/readiness probe — used by Docker, load-balancers, and automated tests."""
    import os
    from datetime import datetime, timezone

    # Check model registry dir
    model_registry_ok = os.path.isdir(os.environ.get("MODEL_REGISTRY_PATH", "data/model_registry"))

    # Check audit + data dirs (lightweight proxy for db_ok in dev/test)
    db_ok = os.path.isdir("audit") or os.path.isdir("data")

    uptime = round(time.time() - _START_TIME, 2)

    # Determine overall status
    if db_ok:
        status = "healthy"
    else:
        status = "degraded"

    return {
        "status":            status,
        "version":           "2.0.0",
        "uptime":            uptime,
        "db_ok":             db_ok,
        "model_registry_ok": model_registry_ok,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "service":           "dipex-api",
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


# ── Dashboard static mount and SPA fallback ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dashboard_path = os.path.join(BASE_DIR, "dashboard")

# Mount static assets if dashboard has been built
_assets_path = os.path.join(dashboard_path, "assets")
if os.path.isdir(_assets_path):
    app.mount("/assets", StaticFiles(directory=_assets_path), name="assets")

from fastapi import Request
from fastapi.responses import FileResponse


@app.get("/", include_in_schema=False)
async def serve_root(request: Request):
    """Serve the React dashboard root. Falls back to /docs if not built yet."""
    index_path = os.path.join(dashboard_path, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    # dashboard not built yet (dev mode or first deploy) → redirect to API docs
    return RedirectResponse(url="/docs")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa_fallback(request: Request, full_path: str):
    """SPA catch-all: serve static files or index.html for React Router routes."""
    # Do not intercept API routes
    if full_path.startswith("api/") or full_path == "api":
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")

    # Try to serve the exact file (favicon.ico, manifest.json, etc.)
    file_path = os.path.join(dashboard_path, full_path)
    if full_path and os.path.isfile(file_path):
        return FileResponse(file_path)

    # Fallback → index.html for React Router client-side routing
    index_path = os.path.join(dashboard_path, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)

    # Dashboard not built: redirect to API docs so users see something useful
    return RedirectResponse(url="/docs")
