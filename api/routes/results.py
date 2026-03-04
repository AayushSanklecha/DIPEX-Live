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


@router.get("/latest")
async def get_latest_result():
    """Returns the most recent pipeline run ID from the audit log."""
    audit_log = "audit/audit.jsonl"
    latest_run_id = None
    if os.path.exists(audit_log):
        with open(audit_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") == "PIPELINE_RUN":
                        latest_run_id = entry.get("run_id")
                except Exception:
                    pass
    if not latest_run_id:
        raise HTTPException(status_code=404, detail="No pipeline runs found")
    
    # Re-use the get_results logic by calling it directly
    return await get_results(latest_run_id)



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
    quality_score = None
    
    confidence_score = 0.0
    gate1_decision = "UNKNOWN"
    gate2_decision = "UNKNOWN"
    dimensions = {
        "data_quality": 0,
        "statistical_strength": 0,
        "stability": 0,
        "compliance": 0
    }
    
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
                        quality_score = entry.get("quality_score")
                        
                        gate1_decision = entry.get("gate1_decision", "UNKNOWN")
                        gate2_decision = entry.get("gate2_decision", "UNKNOWN")
                        confidence_score = entry.get("confidence_score", 0.0)
                        
                        cv = entry.get("confidence_vector", {})
                        comps = cv.get("components", {})
                        if comps:
                            dimensions = {
                                "data_quality": comps.get("base_quality", 0),
                                "statistical_strength": cv.get("confidence_score", 0),
                                "stability": 1.0 - comps.get("retry_penalty", 0),
                                "compliance": 1.0 if comps.get("gate2_passed") else 0.0
                            }
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
                    if quality_score is None:
                        quality_score = snap_data.get("quality_score")
                    if not row_count:
                        row_count = snap_data.get("row_count", 0)
            except Exception:
                pass
    
    if gate_decision == "PASS":
        status = "COMPLETED"
    elif gate_decision == "FAIL":
        status = "FAILED"
    else:
        status = "UNKNOWN"
    
    return {
        "run_id": run_id,
        "dataset_id": dataset_id or "Unknown",
        "status": status,
        "gate_decision": gate_decision or "UNKNOWN",
        "gate1_decision": gate1_decision,
        "gate2_decision": gate2_decision,
        "confidence_score": confidence_score,
        "dimensions": dimensions,
        "timestamp": timestamp,
        "row_count": row_count,
        "col_count": col_count,
        "quality_score": quality_score,
        "snapshot_id": snapshot_id,
        "report_url": f"/report/{run_id}" if report_exists else None,
        "has_report": report_exists,
        "narrative": {
            "title": "Pipeline Execution Overview",
            "body": f"The pipeline completed with a final decision of {gate_decision} and a confidence score of {confidence_score*100:.1f}%. Data quality evaluated to {quality_score}."
        }
    }
