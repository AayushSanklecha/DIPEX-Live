"""
api/routes/exports.py
----------------------
PRESENTATION LAYER — Report & Data Exports

Exposes exactly two outputs:
  1. GET /api/export/list                — list available HTML reports + run json
  2. GET /api/export/report/{run_id}     — download a generated HTML report
  3. GET /api/export/results/json        — pipeline run records (for dashboard KPIs)

No CSV / Parquet / raw dump endpoints — the only outputs are the
Power BI-styled dashboard and the generated HTML executive reports.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

logger = logging.getLogger("dipex.api.routes.exports")

router = APIRouter(prefix="/api/export", tags=["Exports"])

_AUDIT_PATH  = "audit/audit.jsonl"
_REPORTS_DIR = "reports"


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_run_records(limit: int = 500) -> List[dict]:
    records: List[dict] = []
    if not os.path.exists(_AUDIT_PATH):
        return records
    try:
        with open(_AUDIT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("event") == "PIPELINE_RUN":
                        records.append({
                            "run_id":           entry.get("run_id", ""),
                            "dataset_id":       entry.get("dataset_id", ""),
                            "gate_decision":    entry.get("gate_decision", ""),
                            "gate1_decision":   entry.get("gate1_decision", ""),
                            "gate2_decision":   entry.get("gate2_decision", ""),
                            "confidence_score": entry.get("confidence_score", 0.0),
                            "source_type":      entry.get("source_type", ""),
                            "row_count":        entry.get("row_count", 0),
                            "timestamp":        entry.get("timestamp", ""),
                        })
                except json.JSONDecodeError:
                    pass
        records = records[-limit:]
    except Exception as exc:
        logger.warning("Could not read audit log: %s", exc)
    return records


def _list_report_files() -> List[dict]:
    reports = []
    if not os.path.isdir(_REPORTS_DIR):
        return reports
    for fname in sorted(os.listdir(_REPORTS_DIR), reverse=True):
        if fname.endswith((".html", ".htm")):
            fpath = os.path.join(_REPORTS_DIR, fname)
            run_id_guess = fname.replace(".html", "").replace(".htm", "")
            reports.append({
                "filename":     fname,
                "size_bytes":   os.path.getsize(fpath),
                "download_url": f"/api/export/report/{run_id_guess}",
            })
    return reports


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/list",
    summary="List available reports and run summary",
)
async def list_exports():
    """
    Returns available HTML reports (for the Report tab)
    and a summary count of pipeline runs (for KPI tiles).
    """
    reports = _list_report_files()
    records = _load_run_records(limit=1)   # just for the count
    total_count = 0
    if os.path.exists(_AUDIT_PATH):
        with open(_AUDIT_PATH) as f:
            total_count = sum(1 for line in f if "PIPELINE_RUN" in line)

    return {
        "available_reports":   reports,
        "total_pipeline_runs": total_count,
    }


@router.get(
    "/results/json",
    summary="Pipeline run records (for dashboard KPI tiles and charts)",
)
async def export_results_json(
    limit: int = Query(default=500, ge=1, le=5000),
):
    """
    Returns pipeline run records from the audit trail.
    Used by the Power BI-styled dashboard for KPI tiles, charts, and table.
    """
    records = _load_run_records(limit=limit)
    return {"total": len(records), "results": records}


@router.get(
    "/report/{run_id}",
    summary="Download HTML executive report for a pipeline run",
    response_class=Response,
)
async def download_report(run_id: str):
    """
    Download the generated HTML executive report for run_id.
    Called by the Report tab's View and Download buttons.
    Also supports plain filename lookup (e.g. /api/export/report/report_abc.html).
    """
    # Try matching by run_id substring in filename
    report_file: Optional[str] = None
    if os.path.isdir(_REPORTS_DIR):
        for ext in ("html", "htm"):
            # Exact match first
            exact = os.path.join(_REPORTS_DIR, f"{run_id}.{ext}")
            if os.path.exists(exact):
                report_file = exact
                break
            # Partial match
            matches = glob.glob(os.path.join(_REPORTS_DIR, f"*{run_id}*.{ext}"))
            if matches:
                report_file = matches[0]
                break

    if not report_file:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No report found for '{run_id}'. "
                "Run the pipeline to generate a report, then refresh the Report tab."
            ),
        )

    with open(report_file, "rb") as f:
        content = f.read()

    filename = os.path.basename(report_file)
    return Response(
        content=content,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
