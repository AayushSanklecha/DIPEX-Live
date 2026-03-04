"""
api/routes/pipeline_run.py
---------------------------
Unified endpoint: POST /api/pipeline/run

Combines UDIL file ingestion + full PipelineBridge execution in a single
API call. This is the canonical end-to-end flow for:
  file upload → ISSF snapshot → 13-stage pipeline → executive report

Older separate endpoints (/api/ingest + /api/run) remain available for
incremental / batch workflows.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("dipex.api.pipeline_run")
router = APIRouter(prefix="/api", tags=["Pipeline"])


def _load_config() -> dict:
    """Load config.yaml gracefully."""
    try:
        import yaml
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


@router.post(
    "/pipeline/run",
    summary="Ingest a file and run the full DIPEX pipeline in one step",
    response_model=None,
)
async def pipeline_run(
    file: UploadFile = File(...),
    target_col: str = Form("", description="Target column for supervised ML (leave blank for unsupervised)"),
    dataset_id: str = Form("", description="Stable dataset identifier (defaults to filename stem)"),
    file_format: Optional[str] = Form(None, description="Force file format: csv|excel|json|xml|parquet"),
    skip_stages: str = Form("", description="Comma-separated stage names to skip, e.g. 'modeling,rl_update'"),
    save_snapshot: bool = Form(True, description="Persist ISSF snapshot to data/snapshots/"),
) -> Dict[str, Any]:
    """
    **Unified ingest + pipeline endpoint.**

    1. Uploads the file to a temp path
    2. Runs `UniversalIntake.ingest()` → ISSFSnapshot
    3. Optionally saves the snapshot to ``data/snapshots/``
    4. Runs `PipelineBridge.run(snapshot)` — all 13 stages
    5. Returns the full ``PipelineResult.summary()`` + snapshot metadata

    This endpoint unifies the previously disconnected UDIL ingestion
    (``POST /ingest/file``) and pipeline execution (``POST /api/run``)
    flows into a single, atomic operation.
    """
    config = _load_config()
    run_id = str(uuid.uuid4())
    dataset_id = dataset_id or os.path.splitext(file.filename or "upload")[0]
    suffix = os.path.splitext(file.filename or "")[1] or ".csv"

    # ── Save upload to temp ───────────────────────────────────────────────────
    tmp_path = os.path.join(tempfile.gettempdir(), f"dipex_{run_id}{suffix}")
    try:
        with open(tmp_path, "wb") as f_buf:
            shutil.copyfileobj(file.file, f_buf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File upload failed: {exc}") from exc

    try:
        # ── UDIL Ingestion ────────────────────────────────────────────────────
        from ingestion.universal_intake import UniversalIntake, SourceConfig

        intake = UniversalIntake.from_yaml("config.yaml") if os.path.exists("config.yaml") \
            else UniversalIntake(config=config)

        cfg = SourceConfig(
            source_type="file",
            dataset_id=dataset_id,
            data_mode="batch",
            path=tmp_path,
            file_format=file_format,
        )
        snapshot = intake.ingest(cfg)

        if save_snapshot:
            snapshot.save(
                directory=config.get("storage", {}).get("snapshot_dir", "data/snapshots")
            )

        if snapshot.data is None or snapshot.data.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Ingestion produced an empty DataFrame for dataset '{dataset_id}'. "
                    f"Validation status: {snapshot.validation_status}. "
                    f"Errors: {[e.to_dict() for e in snapshot.error_logs[:3]]}"
                ),
            )

        # ── Pipeline Bridge ───────────────────────────────────────────────────
        from ingestion.pipeline_bridge import PipelineBridge

        skip_list = [s.strip() for s in skip_stages.split(",") if s.strip()]
        bridge = PipelineBridge(config=config)
        bridge_result = bridge.run(
            snapshot,
            target_col=target_col or None,
            run_id=run_id,
            skip_stages=skip_list or None,
        )

        # ── Build response ────────────────────────────────────────────────────
        return {
            "status": "ok",
            "run_id": run_id,
            "gate_decision": bridge_result.gate_decision,
            "gate1_decision": bridge_result.gate1_decision,
            "gate2_decision": bridge_result.gate2_decision,
            "confidence_vector": bridge_result.confidence_vector,
            "report_path": bridge_result.report_path,
            "model_metrics": bridge_result.model_metrics,
            "retry_count": bridge_result.retry_count,
            "stages": bridge_result.summary()["stages"],
            "snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "dataset_id": snapshot.dataset_id,
                "row_count": snapshot.row_count,
                "quality_score": snapshot.quality_score,
                "validation_status": snapshot.validation_status,
                "schema_version": snapshot.schema_version,
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("pipeline/run failed for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
