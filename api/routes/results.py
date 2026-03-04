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
    # Check if report exists
    report_path = f"reports/{run_id}_executive_report.html"
    report_exists = os.path.exists(report_path)
    
    # Try to fetch real data from audit log
    audit_log_path = "audit/audit.jsonl"
    dataset_id = None
    gate_decision = None
    timestamp = None
    snapshot_id = None
    row_count = 0
    quality_score = 0.94
    
    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") == run_id:
                        dataset_id = entry.get("dataset_id")
                        gate_decision = entry.get("gate_decision", "UNKNOWN")
                        timestamp = entry.get("timestamp")
                        snapshot_id = entry.get("snapshot_id")
                        row_count = entry.get("row_count", 0)
                        quality_score = entry.get("quality_score", 0.94)
                        break
                except Exception:
                    pass
    
    # Count columns from snapshot metadata
    col_count = 0
    if snapshot_id:
        # Try issf.json format first (new format)
        snapshot_path = f"data/snapshots/{snapshot_id}_issf.json"
        if not os.path.exists(snapshot_path):
            # Fall back to plain JSON
            snapshot_path = f"data/snapshots/{snapshot_id}.json"
        
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, "r") as f:
                    snap_data = json.load(f)
                    # Count columns from column_metadata array
                    if "column_metadata" in snap_data:
                        col_count = len(snap_data["column_metadata"])
            except Exception:
                pass
    
    status = "COMPLETED" if gate_decision == "PASS" else ("FAILED" if gate_decision == "FAIL" else "COMPLETED")
    
    return {
        "run_id": run_id,
        "dataset_id": dataset_id or "Unknown",
        "status": status,
        "gate_decision": gate_decision or "UNKNOWN",
        "timestamp": timestamp,
        "row_count": row_count,
        "col_count": col_count,
        "quality_score": quality_score,
        "snapshot_id": snapshot_id,
        "report_url": f"/report/{run_id}" if report_exists else None,
        "has_report": report_exists
    }
