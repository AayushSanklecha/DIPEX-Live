"""
api/routes/ingest_v2.py
-------------------------
FastAPI routes for the Universal Data Intake & Processing Layer (UDIL v2).

Endpoints
---------
POST /ingest/file          — multipart file upload → ISSF snapshot
POST /ingest/api           — trigger pull from external API → ISSF
POST /ingest/db            — extract from database → ISSF
POST /ingest/stream/events — process in-memory event batch → ISSF list
GET  /ingest/schema/{id}   — schema version history for a dataset
GET  /ingest/quality/{id}  — quality report for a snapshot
GET  /ingest/snapshots     — list all stored snapshots
GET  /ingest/snapshot/{id} — retrieve a specific ISSF snapshot metadata
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ingestion.error_handler import (
    DBConnectionError, DataFormatError, EncodingError,
    QualityGateError, SchemaError, StreamLagError,
)
from ingestion.schema_registry import SchemaRegistry
from ingestion.universal_intake import SourceConfig, UniversalIntake

logger = logging.getLogger("dipex.api.ingest_v2")
router = APIRouter(prefix="/ingest", tags=["Universal Intake"])


def _get_intake() -> UniversalIntake:
    """Dependency that builds UniversalIntake from config.yaml if present."""
    if os.path.exists("config.yaml"):
        try:
            return UniversalIntake.from_yaml("config.yaml")
        except Exception:  # noqa: BLE001
            pass
    return UniversalIntake()


def _err(status: int, msg: str, detail: Any = None):
    raise HTTPException(status_code=status, detail={"error": msg, "detail": detail})


# ── POST /ingest/file ─────────────────────────────────────────────────────────

@router.post("/file", summary="Upload a file and ingest it as an ISSF snapshot")
async def ingest_file(
    file: UploadFile = File(...),
    dataset_id: str  = Form(""),
    file_format: Optional[str] = Form(None),
    sheet_name: Optional[str]  = Form(None),
    require_quality_pass: bool = Form(True),
    block_on_schema_break: bool = Form(True),
    intake: UniversalIntake = Depends(_get_intake),
):
    dataset_id = dataset_id or os.path.splitext(file.filename or "upload")[0]
    suffix     = os.path.splitext(file.filename or "")[1] or ".csv"

    # Save upload to a temp file
    tmp_path = os.path.join(tempfile.gettempdir(), f"dipex_upload_{uuid.uuid4()}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        _err(500, "File upload failed", str(exc))

    try:
        cfg = SourceConfig(
            source_type="file",
            dataset_id=dataset_id,
            data_mode="batch",
            path=tmp_path,
            file_format=file_format,
            sheet_name=sheet_name,
            require_quality_pass=require_quality_pass,
            block_on_schema_break=block_on_schema_break,
        )
        snapshot = intake.ingest(cfg)
        return snapshot.to_dict()
    except SchemaError as exc:
        _err(422, "Breaking schema drift detected", str(exc))
    except QualityGateError as exc:
        _err(422, "Quality gate failed", str(exc))
    except (DataFormatError, EncodingError) as exc:
        _err(400, "File format or encoding error", str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("File ingestion error")
        _err(500, "Ingestion failed", str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── POST /ingest/api ──────────────────────────────────────────────────────────

@router.post("/api", summary="Pull data from an external API and ingest as ISSF")
async def ingest_api(
    body: Dict[str, Any],
    intake: UniversalIntake = Depends(_get_intake),
):
    """
    Body fields::

        dataset_id     : str
        api_config     : dict  (maps to APISourceConfig fields)
        require_quality_pass   : bool (default true)
        block_on_schema_break  : bool (default true)
    """
    from ingestion.readers.api_reader import APISourceConfig, AuthConfig, PaginationConfig
    try:
        api_raw = body.get("api_config", {})
        auth_raw = api_raw.pop("auth", {}) if isinstance(api_raw, dict) else {}
        pag_raw  = api_raw.pop("pagination", {}) if isinstance(api_raw, dict) else {}
        auth = AuthConfig(**auth_raw) if auth_raw else AuthConfig()
        pag  = PaginationConfig(**pag_raw) if pag_raw else PaginationConfig()
        api_cfg = APISourceConfig(auth=auth, pagination=pag, **api_raw)

        cfg = SourceConfig(
            source_type="api",
            dataset_id=body.get("dataset_id", "api_pull"),
            data_mode="live",
            api_config=api_cfg,
            require_quality_pass=body.get("require_quality_pass", True),
            block_on_schema_break=body.get("block_on_schema_break", True),
        )
        snapshot = intake.ingest(cfg)
        return snapshot.to_dict()
    except SchemaError as exc:
        _err(422, "Breaking schema drift", str(exc))
    except QualityGateError as exc:
        _err(422, "Quality gate failed", str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("API ingestion error")
        _err(500, "API ingestion failed", str(exc))


# ── POST /ingest/db ───────────────────────────────────────────────────────────

@router.post("/db", summary="Extract from a database and ingest as ISSF")
async def ingest_db(
    body: Dict[str, Any],
    intake: UniversalIntake = Depends(_get_intake),
):
    """
    Body fields::

        dataset_id   : str
        db_config    : dict (maps to DBSourceConfig fields)
        require_quality_pass  : bool
        block_on_schema_break : bool
    """
    from ingestion.readers.db_reader import DBSourceConfig
    from .pipeline_run import _db_cfg_from_uri
    try:
        db_raw = body.get("db_config", {})
        # If db_config is a string (URI or JSON), parse it
        if isinstance(db_raw, str):
            source_kind = body.get("source_kind", "database")
            db_cfg_dict = _db_cfg_from_uri(db_raw, source_kind)
        else:
            # If it's already a dict, we assume it's the structured config
            db_cfg_dict = db_raw

        db_cfg = DBSourceConfig(**db_cfg_dict)
        cfg = SourceConfig(
            source_type="database",
            dataset_id=body.get("dataset_id", "db_extract"),
            data_mode="batch",
            db_config=db_cfg_dict,
            require_quality_pass=body.get("require_quality_pass", True),
            block_on_schema_break=body.get("block_on_schema_break", True),
        )
        snapshot = intake.ingest(cfg)
        return snapshot.to_dict()
    except DBConnectionError as exc:
        _err(503, "Database connection failed", str(exc))
    except SchemaError as exc:
        _err(422, "Breaking schema drift", str(exc))
    except QualityGateError as exc:
        _err(422, "Quality gate failed", str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("DB ingestion error")
        _err(500, "DB ingestion failed", str(exc))


# ── POST /ingest/stream/events ────────────────────────────────────────────────

@router.post("/stream/events", summary="Ingest a batch of in-memory events through windowing")
async def ingest_stream_events(
    body: Dict[str, Any],
    intake: UniversalIntake = Depends(_get_intake),
):
    """
    Body fields::

        dataset_id    : str
        events        : list[dict]   — list of event dicts
        window_config : dict         — WindowConfig fields (strategy, window_size_s, …)
    """
    from ingestion.readers.stream_reader import StreamReader, WindowConfig
    from ingestion.universal_intake import SourceConfig
    try:
        events     = body.get("events", [])
        window_raw = body.get("window_config", {})
        window_cfg = WindowConfig(**window_raw) if window_raw else WindowConfig()
        reader     = StreamReader()
        stream_results = reader.collect_events(events, window_cfg)

        snapshots_out = []
        for sr in stream_results:
            cfg = SourceConfig(
                source_type="stream",
                dataset_id=body.get("dataset_id", "event_stream"),
                data_mode="stream",
            )
            # Inline: normalise + quality check without re-reading
            df, col_meta = intake.normaliser.normalise(sr.data, dataset_id=cfg.dataset_id)
            schema = {c.name: c.dtype for c in col_meta}
            drift  = intake.schema_reg.register(cfg.dataset_id, schema, len(df))
            qr     = intake.quality_gate.check(df, cfg.dataset_id)
            from ingestion.issf import ISSFSnapshot
            snap = ISSFSnapshot(
                dataset_id=cfg.dataset_id,
                schema_version=drift.new_version,
                data_mode="stream",
                source_type="stream",
                source_uri=body.get("source_uri", "event_hub"),
                column_metadata=col_meta,
                row_count=len(df),
                quality_score=qr.quality_score,
                validation_status=qr.validation_status,
                data=df,
                extra_meta={"window_start": sr.window_start, "window_end": sr.window_end, "late_dropped": sr.late_dropped},
            )
            snap.save(intake.snapshot_dir)
            snapshots_out.append(snap.to_dict())
        return {"windows_created": len(snapshots_out), "snapshots": snapshots_out}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stream event ingestion error")
        _err(500, "Stream ingestion failed", str(exc))


# ── GET /ingest/schema/{dataset_id} ──────────────────────────────────────────

@router.get("/schema/{dataset_id}", summary="Retrieve schema version history")
async def get_schema_history(dataset_id: str):
    try:
        registry = SchemaRegistry()
        history  = registry.get_history(dataset_id)
        if not history:
            _err(404, f"No schema history found for dataset '{dataset_id}'")
        return {"dataset_id": dataset_id, "versions": history}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _err(500, "Schema history retrieval failed", str(exc))


# ── GET /ingest/quality/{snapshot_id} ────────────────────────────────────────

@router.get("/quality/{snapshot_id}", summary="Retrieve quality report for a snapshot")
async def get_quality_report(snapshot_id: str, snapshot_dir: str = "data/snapshots"):
    import pathlib
    meta_path = pathlib.Path(snapshot_dir) / f"{snapshot_id}_issf.json"
    if not meta_path.exists():
        _err(404, f"Snapshot '{snapshot_id}' not found")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "snapshot_id":       meta.get("snapshot_id"),
        "dataset_id":        meta.get("dataset_id"),
        "quality_score":     meta.get("quality_score"),
        "validation_status": meta.get("validation_status"),
        "row_count":         meta.get("row_count"),
        "error_count":       len(meta.get("error_logs", [])),
        "errors":            meta.get("error_logs", []),
        "schema_version":    meta.get("schema_version"),
    }


# ── GET /ingest/snapshots ─────────────────────────────────────────────────────

@router.get("/snapshots", summary="List all ingestion snapshots")
async def list_snapshots(
    snapshot_dir: str = "data/snapshots",
    dataset_id: Optional[str] = None,
    limit: int = 50,
):
    import pathlib
    snap_dir = pathlib.Path(snapshot_dir)
    if not snap_dir.exists():
        return {"snapshots": [], "total": 0}
    files = sorted(snap_dir.glob("*_issf.json"), key=os.path.getmtime, reverse=True)
    results = []
    for f in files[:limit]:
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            if dataset_id and meta.get("dataset_id") != dataset_id:
                continue
            results.append({
                "snapshot_id":       meta.get("snapshot_id"),
                "dataset_id":        meta.get("dataset_id"),
                "ingestion_timestamp": meta.get("ingestion_timestamp"),
                "row_count":         meta.get("row_count"),
                "quality_score":     meta.get("quality_score"),
                "validation_status": meta.get("validation_status"),
                "schema_version":    meta.get("schema_version"),
                "source_type":       meta.get("source_type"),
            })
        except Exception:  # noqa: BLE001
            pass
    return {"snapshots": results, "total": len(results)}


# ── GET /ingest/snapshot/{snapshot_id} ───────────────────────────────────────

@router.get("/snapshot/{snapshot_id}", summary="Retrieve full ISSF metadata for one snapshot")
async def get_snapshot(snapshot_id: str, snapshot_dir: str = "data/snapshots"):
    import pathlib
    meta_path = pathlib.Path(snapshot_dir) / f"{snapshot_id}_issf.json"
    if not meta_path.exists():
        _err(404, f"Snapshot '{snapshot_id}' not found")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


# ── GET /ingest/insights ── Adaptive Learner KB ───────────────────────────────

@router.get("/insights", summary="Adaptive learner knowledge base insights")
async def get_learner_insights(dataset_id: Optional[str] = None):
    """
    Return insights from the AdaptiveLearner:
    - Global stats (total runs, pass rate, avg quality)
    - Top recurring errors with root causes and fix suggestions
    - Per-dataset quality history and schema drift history
    - Learned strategies with success rates
    """
    try:
        from ingestion.adaptive_learner import AdaptiveLearner
        learner = AdaptiveLearner()
        insights = learner.get_insights(dataset_id=dataset_id)
        strategies = learner.get_strategy_report()
        return {
            "insights": insights,
            "learned_strategies": strategies[:20],
        }
    except Exception as exc:  # noqa: BLE001
        _err(500, "Failed to read learner knowledge base", str(exc))


# ── GET /ingest/learner/recommend ─────────────────────────────────────────────

@router.get("/learner/recommend", summary="Get pre-flight strategy recommendation")
async def get_recommendation(
    dataset_id: str,
    source_type: str = "file",
):
    """
    Ask the AdaptiveLearner what strategy to use for the next ingest
    of this dataset (format, encoding, expected quality issues, etc.)
    """
    try:
        from ingestion.adaptive_learner import AdaptiveLearner
        learner = AdaptiveLearner()
        rec = learner.recommend(dataset_id=dataset_id, source_type=source_type)
        return {
            "dataset_id": rec.dataset_id,
            "recommended_format": rec.recommended_format,
            "recommended_encoding": rec.recommended_encoding,
            "recommended_delimiter": rec.recommended_delimiter,
            "skip_stages": rec.skip_stages,
            "quality_pre_checks": rec.quality_pre_checks,
            "warnings": rec.warnings,
            "confidence": rec.confidence,
            "reason": rec.reason,
        }
    except Exception as exc:  # noqa: BLE001
        _err(500, "Recommendation failed", str(exc))


# ── POST /ingest/learner/feedback ─────────────────────────────────────────────

@router.post("/learner/feedback", summary="Manually record ingestion outcome for learning")
async def record_feedback(body: Dict[str, Any]):
    """
    Manually record a custom ingestion outcome so the learner can improve.
    Useful for recording outcomes from external pipelines.

    Body: IngestionOutcome fields as JSON.
    """
    try:
        from ingestion.adaptive_learner import AdaptiveLearner, IngestionOutcome
        learner = AdaptiveLearner()
        outcome = IngestionOutcome(**{k: v for k, v in body.items()
                                     if k in IngestionOutcome.__dataclass_fields__})
        learner.record(outcome)
        return {"status": "recorded", "dataset_id": outcome.dataset_id}
    except Exception as exc:  # noqa: BLE001
        _err(400, "Failed to record feedback", str(exc))


# ── GET /ingest/layers — list all layer records ───────────────────────────────

@router.get("/layers", summary="List all Bronze/Silver/Gold layer records")
async def list_layers(
    layer: Optional[str] = None,
    dataset_id: Optional[str] = None,
):
    """
    List persisted layer records across Bronze, Silver, and Gold.
    Filterable by layer name (bronze|silver|gold) and dataset_id.
    """
    try:
        from ingestion.data_layers import LayerManager
        lm = LayerManager()
        layers_to_scan = [layer] if layer else ["bronze", "silver", "gold"]
        results = []
        for lyr in layers_to_scan:
            records = lm.list_layer(lyr, dataset_id=dataset_id)
            for r in records:
                r["_layer"] = lyr
                results.append(r)
        return {"total": len(results), "records": results}
    except Exception as exc:  # noqa: BLE001
        _err(500, "Layer listing failed", str(exc))


# ── GET /ingest/layers/{layer}/{dataset_id} ───────────────────────────────────

@router.get("/layers/{layer}/{dataset_id}", summary="Layer records for a specific dataset")
async def get_layer_records(layer: str, dataset_id: str):
    """Get all records for a specific layer + dataset combination."""
    try:
        from ingestion.data_layers import LayerManager
        lm = LayerManager()
        records = lm.list_layer(layer, dataset_id=dataset_id)
        return {"layer": layer, "dataset_id": dataset_id, "records": records, "total": len(records)}
    except Exception as exc:  # noqa: BLE001
        _err(500, "Layer record retrieval failed", str(exc))


# ── GET /ingest/lineage/{dataset_id} ─────────────────────────────────────────

@router.get("/lineage/{dataset_id}", summary="Data lineage records for a dataset")
async def get_lineage(dataset_id: str):
    """
    Return all Gold lineage records for a given dataset — showing
    which Silver snapshot each Gold artefact was derived from,
    which component created it, and what transformation steps were applied.
    """
    try:
        from ingestion.lineage import LineageStore
        store = LineageStore()
        records = store.list_for_dataset(dataset_id)
        return {
            "dataset_id": dataset_id,
            "total": len(records),
            "lineage": [r.to_dict() for r in records],
        }
    except Exception as exc:  # noqa: BLE001
        _err(500, "Lineage query failed", str(exc))





# ── GET /ingest/layer/verify ──────────────────────────────────────────────────

@router.get("/layer/verify", summary="Verify Bronze/Silver layer checksum integrity")
async def verify_layer_integrity(
    dataset_id: str,
    snapshot_id: str,
    layer: str = "silver",
):
    """
    Verify that a stored Bronze or Silver layer Parquet matches
    its SHA-256 lock file. Returns verified=true/false.
    Raises 422 if mismatch detected (corruption or tampering).
    """
    try:
        from ingestion.data_layers import LayerManager
        from ingestion.immutability_guard import ChecksumMismatchError
        lm = LayerManager()
        report = lm.verify_layer(dataset_id=dataset_id, layer=layer, snapshot_id=snapshot_id)
        if not report.get("verified"):
            _err(422, "Layer integrity check failed", report.get("error"))
        return report
    except ChecksumMismatchError as exc:
        _err(422, "Checksum mismatch — possible tampering detected", str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _err(500, "Layer verification failed", str(exc))


# ── POST /ingest/all-databases — Multi-DB aggregation trigger ─────────────────

@router.post("/all-databases", summary="Trigger ingestion from all configured databases")
async def ingest_all_databases():
    """
    Trigger the multi-database aggregation pipeline.
    Pulls from all configured sources (MongoDB, Redis, PostgreSQL, Neo4j,
    Kafka, DuckDB, Parquet) through the UniversalIntake pipeline.
    Returns a status summary of each source ingest attempt.
    """
    import importlib
    import traceback
    results = []
    try:
        ia = importlib.import_module("scripts.ingest_all_databases")
        # run_aggregation returns a list of dicts or raises
        raw = ia.run_aggregation()
        if isinstance(raw, list):
            results = raw
        else:
            results = [{"source": "all", "status": "ok", "detail": str(raw)}]
    except Exception as exc:  # noqa: BLE001
        logger.error("ingest_all_databases failed: %s", exc)
        tb = traceback.format_exc(limit=5)
        results = [{"source": "all", "status": "error", "detail": str(exc), "traceback": tb}]
    return {
        "status": "completed" if all(r.get("status") != "error" for r in results) else "partial",
        "sources_attempted": len(results),
        "results": results,
    }


