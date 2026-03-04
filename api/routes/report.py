"""
api/routes/report.py
----------------------
Executive report generation and download endpoints.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

router = APIRouter(prefix="/report", tags=["Reporting"])
logger = logging.getLogger("dipex.api.report")

REPORT_DIR = "reports"


class ReportRequest(BaseModel):
    run_id: str


@router.post("/executive")
async def generate_executive_report(req: ReportRequest):
    """Generate an executive HTML report for a run_id."""
    import yaml
    from reporting_service.executive_report import ExecutiveReportGenerator
    from reporting_service.risk_communicator import RiskCommunicator

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load approved output if it exists
    approved_dir = config.get("storage", {}).get("approved_output_dir", "data/approved_outputs")
    approved_path = os.path.join(approved_dir, f"{req.run_id}.json")

    conf_vec: Dict[str, Any] = {"confidence_score": 0.0}
    gate1, gate2 = "UNKNOWN", "UNKNOWN"
    narrative, fingerprint, schema_version = "", "", "1.0"
    analyst_flags = []
    row_count = col_count = flag_count = retry_count = 0
    model_metrics: Dict = {}
    gov_decision = "N/A"

    if os.path.exists(approved_path):
        with open(approved_path, "r") as f:
            ao = json.load(f)
        conf_vec = ao.get("confidence_vector", {})
        gate1 = ao.get("gate1_decision", "UNKNOWN")
        gate2 = ao.get("gate2_decision", "UNKNOWN")
        narrative = ao.get("narrative", "")
        fingerprint = ao.get("fingerprint", "")
        schema_version = ao.get("schema_version", "1.0")
        retry_count = ao.get("retry_count", 0)
        profile = ao.get("profile_summary", {})
        analyst_flags = profile.get("analyst_flags_sample", [])
        flag_count = profile.get("flag_count", 0)
        shape = profile.get("dataset_shape", {})
        row_count = shape.get("rows", 0)
        col_count = shape.get("columns", 0)
        proposal = ao.get("proposal_summary", {})
        model_metrics = {k: v for k, v in proposal.items() if k not in ("features",) and v is not None}

    # Risk assessment
    rc = RiskCommunicator.from_config(config)
    risk_report = rc.evaluate(
        run_id=req.run_id,
        confidence_vector=conf_vec,
        gate1_decision=gate1,
        gate2_decision=gate2,
        analyst_flags=analyst_flags,
        retry_count=retry_count,
    )

    reporter = ExecutiveReportGenerator.from_config(config)
    report_path = reporter.generate(
        run_id=req.run_id,
        confidence_vector=conf_vec,
        gate1_decision=gate1,
        gate2_decision=gate2,
        narrative=narrative,
        analyst_flags=analyst_flags,
        model_metrics=model_metrics,
        risk_flags=[f.to_dict() for f in risk_report.flags],
        fingerprint=fingerprint,
        schema_version=schema_version,
        row_count=row_count,
        col_count=col_count,
        flag_count=flag_count,
        retry_count=retry_count,
        gov_decision=gov_decision,
    )

    if not report_path:
        raise HTTPException(500, detail="Report generation failed. Check that jinja2 is installed.")

    return {
        "status": "generated",
        "run_id": req.run_id,
        "report_path": report_path,
        "risk_level": risk_report.overall_risk_level,
        "download_url": f"/report/{req.run_id}",
    }


@router.get("/list")
async def list_reports():
    """List all generated executive reports."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    files = [
        {
            "filename": f,
            "run_id": f.replace("_executive_report.html", ""),
            "size_bytes": os.path.getsize(os.path.join(REPORT_DIR, f)),
            "download_url": f"/report/{f.replace('_executive_report.html', '')}",
        }
        for f in os.listdir(REPORT_DIR)
        if f.endswith("_executive_report.html")
    ]
    return {"count": len(files), "reports": files}


@router.get("/{run_id}")
async def download_report(run_id: str):
    """Download (serve) the executive HTML report for a run_id."""
    path = os.path.join(REPORT_DIR, f"{run_id}_executive_report.html")
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"Report for run_id '{run_id}' not found. Generate it first via POST /report/executive.")
    return FileResponse(path, media_type="text/html", filename=f"{run_id}_executive_report.html")
