import json
import os
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("", summary="List all pipeline run IDs")
async def list_results(limit: int = 20):
    """Returns the most recent pipeline run IDs from the audit log."""
    audit_log = "audit/audit.jsonl"
    runs = []
    if os.path.exists(audit_log):
        with open(audit_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") == "PIPELINE_RUN":
                        runs.append({
                            "run_id": entry.get("run_id"),
                            "dataset_id": entry.get("dataset_id"),
                            "gate_decision": entry.get("gate_decision"),
                            "timestamp": entry.get("timestamp"),
                        })
                except Exception:
                    pass
    return {"runs": runs[-limit:], "total": len(runs)}


@router.get("/{run_id}")
async def get_results(run_id: str):
    """Returns the results, confidence scores, and explanations for a run."""
    # Placeholder for database retrieval
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "confidence_score": 0.94,
        "insight": "High correlation detected between X and Y.",
        "explanation": "### Summary\nBased on past precedents...",
        "profile_report_url": f"/api/reports/{run_id}"
    }
