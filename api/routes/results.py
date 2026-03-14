"""
api/routes/results.py
----------------------
Results API: list all runs, get latest, get by run_id.

Sample rows for the dashboard are loaded from the Parquet snapshot file
(data/snapshots/{snapshot_id}_issf.parquet) — NOT from the metadata JSON.
The JSON sidecar is metadata-only (column stats, quality scores); actual
row data lives in the Parquet file written by SnapshotManager.

Falls back to a 'data' key in the JSON sidecar for legacy snapshots that
pre-date the Parquet storage convention.
"""

import json
import os
import io
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger("dipex.api.results")

router = APIRouter(prefix="/api/results", tags=["results"])


def _load_df(run_id: str):
    """Best-effort DataFrame loader: tries Parquet first, then JSON sidecar."""
    try:
        import pandas as pd
        audit_log_path = "audit/audit.jsonl"
        if not os.path.exists(audit_log_path):
            return None
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") == run_id:
                        snap = entry.get("snapshot_id")
                        if snap:
                            p = f"data/snapshots/{snap}_issf.parquet"
                            if os.path.exists(p):
                                return pd.read_parquet(p)
                            j = f"data/snapshots/{snap}.json"
                            if os.path.exists(j):
                                with open(j, "r") as jf:
                                    sidecar = json.load(jf)
                                rows = sidecar.get("data", [])
                                if rows:
                                    return pd.DataFrame(rows)
                except Exception:
                    pass
        return None
    except Exception:
        return None



@router.get("", summary="List all pipeline run IDs")
async def list_results(limit: int = 50):
    """Returns the most recent pipeline runs from the audit log with rich metadata."""
    audit_log = "audit/audit.jsonl"
    runs = []
    if os.path.exists(audit_log):
        with open(audit_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event") == "PIPELINE_RUN":
                        runs.append({
                            "run_id":           entry.get("run_id"),
                            "dataset_id":       entry.get("dataset_id"),
                            "gate_decision":    entry.get("gate_decision"),
                            "gate1_decision":   entry.get("gate1_decision"),
                            "gate2_decision":   entry.get("gate2_decision"),
                            "confidence_score": entry.get("confidence_score", 0.0),
                            "quality_score":    entry.get("quality_score"),
                            "row_count":        entry.get("row_count", 0),
                            "source_kind":      entry.get("source_kind", "file"),
                            "timestamp":        entry.get("timestamp"),
                        })
                except Exception:
                    pass
    return {"runs": runs[-limit:], "total": len(runs)}


@router.get("/latest")
async def get_latest_result():
    """Returns the most recent pipeline run result."""
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
    return await get_results(latest_run_id)


@router.get("/{run_id}/export/powerbi")
async def export_powerbi(run_id: str):
    """Export the processed dataset as a CSV specifically formatted for Power BI."""
    audit_log_path = "audit/audit.jsonl"
    snapshot_id = None
    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") == run_id:
                        snapshot_id = entry.get("snapshot_id")
                        break
                except Exception:
                    pass
    
    if not snapshot_id:
        raise HTTPException(status_code=404, detail="Run ID not found")
        
    parquet_path = f"data/snapshots/{snapshot_id}_issf.parquet"
    if not os.path.exists(parquet_path):
        raise HTTPException(status_code=404, detail="Processed data not found for this run")
        
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=powerbi_export_{run_id}.csv"
    return response


@router.get("/{run_id}/intelligence")
async def get_intelligence(run_id: str):
    """
    Returns the Phase 1 Combinatorial Analysis of the dataset.
    This includes dynamically categorized charts (Scatter, Line, Stacked Bar)
    ranked by mathematical variance/relevance, plus textual insight feeds.
    """
    audit_log_path = "audit/audit.jsonl"
    snapshot_id = None
    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") == run_id:
                        snapshot_id = entry.get("snapshot_id")
                        break
                except Exception:
                    pass
    
    if not snapshot_id:
        raise HTTPException(status_code=404, detail="Run ID not found")
        
    import pandas as pd
    from reporting_service.intelligence_engine import IntelligenceEngine
    
    # Primary path: full processed Parquet snapshot
    parquet_path = f"data/snapshots/{snapshot_id}_issf.parquet"
    
    try:
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            # Fallback: try the raw JSON sidecar data
            df = _load_df(run_id)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="No processable data found for this run.")
        
        engine = IntelligenceEngine()
        analysis = engine.analyze_dataset(df, max_charts=20)
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence analysis failed for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence analysis failed: {str(e)}")


@router.get("/{run_id}")
async def get_results(run_id: str):
    """Returns full results, metrics, stages, risk flags, and explanations for a run."""
    report_path = f"reports/{run_id}_executive_report.html"
    report_exists = os.path.exists(report_path)

    audit_log_path = "audit/audit.jsonl"
    dataset_id      = None
    gate_decision   = None
    timestamp       = None
    snapshot_id     = None
    row_count       = 0
    quality_score   = None
    source_kind     = "file"
    target_col      = None
    retry_count     = 0

    confidence_score  = 0.0
    gate1_decision    = "UNKNOWN"
    gate2_decision    = "UNKNOWN"
    confidence_vector = {}
    model_metrics     = {}
    analyst_flags     = []
    stages            = []

    dimensions = {
        "data_quality":       0.0,
        "statistical_strength": 0.0,
        "stability":          0.0,
        "compliance":         0.0,
    }

    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") == run_id:
                        dataset_id       = entry.get("dataset_id")
                        gate_decision    = entry.get("gate_decision", "UNKNOWN")
                        timestamp        = entry.get("timestamp")
                        snapshot_id      = entry.get("snapshot_id")
                        row_count        = entry.get("row_count", 0)
                        quality_score    = entry.get("quality_score")
                        source_kind      = entry.get("source_kind", "file")
                        target_col       = entry.get("target_column_used")
                        retry_count      = entry.get("retry_count", 0)

                        gate1_decision   = entry.get("gate1_decision", "UNKNOWN")
                        gate2_decision   = entry.get("gate2_decision", "UNKNOWN")
                        confidence_score = entry.get("confidence_score", 0.0)
                        confidence_vector = entry.get("confidence_vector", {})
                        model_metrics    = entry.get("model_metrics", {}) or {}
                        analyst_flags    = entry.get("analyst_flags", []) or []
                        stages           = entry.get("stages", []) or []

                        cv = confidence_vector
                        comps = cv.get("components", {})
                        if comps:
                            dimensions = {
                                "data_quality":         comps.get("base_quality", 0.0),
                                "statistical_strength": cv.get("confidence_score", 0.0),
                                "stability":            round(1.0 - comps.get("retry_penalty", 0.0), 3),
                                "compliance":           1.0 if comps.get("gate2_passed") else 0.0,
                            }
                        break
                except Exception:
                    pass

    # ── Column count + quality score from snapshot ──────────────────────────
    col_count = 0
    column_metadata = []
    if snapshot_id:
        for snap_path in [
            f"data/snapshots/{snapshot_id}_issf.json",
            f"data/snapshots/{snapshot_id}.json",
        ]:
            if os.path.exists(snap_path):
                try:
                    with open(snap_path, "r", encoding="utf-8") as f:
                        snap_data = json.load(f)
                    column_metadata = snap_data.get("column_metadata", [])
                    col_count = len(column_metadata)
                    if quality_score is None:
                        quality_score = snap_data.get("quality_score")
                    if not row_count:
                        row_count = snap_data.get("row_count", 0)
                except Exception:
                    pass
                break

    # ── Status string ────────────────────────────────────────────────────────
    status = {"PASS": "COMPLETED", "FAIL": "FAILED"}.get(gate_decision, "UNKNOWN")

    # ── Sample rows for heatmap / dashboard ─────────────────────────────────
    sample_rows = []
    try:
        if snapshot_id:
            # Primary: load from Parquet (the actual data file)
            parquet_path = f"data/snapshots/{snapshot_id}_issf.parquet"
            if os.path.exists(parquet_path):
                import pandas as pd
                df_snap = pd.read_parquet(parquet_path)
                sample_df = df_snap.head(500)
                sample_rows = sample_df.where(sample_df.notna(), None).to_dict(orient="records")
            else:
                # Fallback: look for a data field in the JSON metadata file
                for snap_path in [
                    f"data/snapshots/{snapshot_id}_issf.json",
                    f"data/snapshots/{snapshot_id}.json",
                ]:
                    if os.path.exists(snap_path):
                        with open(snap_path, "r", encoding="utf-8") as sf:
                            snap_data = json.load(sf)
                        if isinstance(snap_data.get("data"), list):
                            sample_rows = snap_data["data"][:500]
                        break
    except Exception:
        pass

    # ── Derive risk flags from analyst_flags + gate outcomes ─────────────────
    risk_flags = []
    for flag in analyst_flags:
        if isinstance(flag, dict):
            risk_flags.append({
                "severity":    flag.get("severity", "MEDIUM"),
                "column":      flag.get("column", ""),
                "description": flag.get("message") or flag.get("description", str(flag)),
            })
        elif isinstance(flag, str):
            risk_flags.append({"severity": "MEDIUM", "column": "", "description": flag})

    if gate1_decision not in ("PASS", "UNKNOWN"):
        risk_flags.insert(0, {
            "severity":    "HIGH",
            "column":      "",
            "description": f"Gate 1 (Quality) did not pass ({gate1_decision}). Data quality may be insufficient for reliable modelling.",
        })
    if gate2_decision not in ("PASS", "UNKNOWN"):
        risk_flags.insert(0, {
            "severity":    "HIGH",
            "column":      "",
            "description": f"Gate 2 (Statistical) did not pass ({gate2_decision}). Feature distributions may be unstable.",
        })
    if retry_count > 0:
        risk_flags.append({
            "severity":    "MEDIUM",
            "column":      "",
            "description": f"Pipeline required {retry_count} retry(s). The dataset may be borderline quality.",
        })

    # ── Build default stage list if backend didn't log one ───────────────────
    if not stages:
        STAGE_NAMES = [
            "Universal Intake",
            "Data Triage",
            "Missing Patterns",
            "Preprocessing",
            "Drift Detection",
            "Gate 1 — Validation & Compliance",
            "Profiling",
            "Analytics Layer",
            "Governance",
            "Statistics",
            "Leakage & Multicollinearity",
            "AutoML & Calibration",
            "Gate 2 — Verification",
            "RL Feedback & Experience Memory",
            "Report Generation",
        ]
        stages = [
            {"name": s, "status": "PASS" if gate_decision == "PASS" else "UNKNOWN"}
            for s in STAGE_NAMES
        ]
        # Mark gate failures at their correct indices
        if gate1_decision == "FAIL":
            stages[5]["status"] = "FAIL"   # Gate 1 — Validation & Compliance
        if gate2_decision == "FAIL":
            stages[12]["status"] = "FAIL"  # Gate 2 — Verification

    return {
        "run_id":           run_id,
        "dataset_id":       dataset_id or "Unknown",
        "status":           status,
        "gate_decision":    gate_decision or "UNKNOWN",
        "gate1_decision":   gate1_decision,
        "gate2_decision":   gate2_decision,
        "confidence_score": confidence_score,
        "confidence_vector": confidence_vector,
        "dimensions":       dimensions,
        "timestamp":        timestamp,
        "row_count":        row_count,
        "col_count":        col_count,
        "quality_score":    quality_score,
        "source_kind":      source_kind,
        "target_col":       target_col,
        "retry_count":      retry_count,
        "snapshot_id":      snapshot_id,
        "sample_rows":      sample_rows,
        "column_metadata":  column_metadata,
        "model_metrics":    model_metrics,
        "analyst_flags":    analyst_flags,
        "risk_flags":       risk_flags,
        "stages":           stages,
        "report_url":       f"/report/{run_id}" if report_exists else None,
        "has_report":       report_exists,
        "narrative": {
            "title": "Pipeline Execution Overview",
            "body":  (
                f"The pipeline completed with a final decision of {gate_decision} "
                f"and a confidence score of {confidence_score * 100:.1f}%. "
                f"Data quality evaluated to {quality_score}. "
                f"{'No retries were needed.' if retry_count == 0 else f'The pipeline required {retry_count} retry(s).'}"
            ),
        },
    }
