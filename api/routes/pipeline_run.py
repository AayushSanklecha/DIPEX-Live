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

import json
import logging
import os
import shutil
import tempfile
import uuid
from urllib.parse import parse_qs, urlparse
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


def _guess_target_col(df) -> Optional[str]:
    """Pick a sensible default target column when user does not provide one."""
    if df is None or getattr(df, "empty", True):
        return None
    lower_to_real = {str(c).lower(): str(c) for c in df.columns}
    for candidate in ("target", "label", "class", "y", "churn", "fraud"):
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    return None


def _stage_brief(stages: list[dict]) -> list[dict]:
    return [
        {
            "stage": s.get("stage"),
            "status": s.get("status"),
            "elapsed_ms": s.get("elapsed_ms"),
        }
        for s in stages
    ]


def _db_cfg_from_uri(uri: str, source_kind: str) -> Dict[str, Any]:
    parsed = urlparse(uri)
    scheme = (parsed.scheme or "").lower()
    q = parse_qs(parsed.query)
    path_parts = [p for p in parsed.path.split("/") if p]

    if source_kind == "graph_db" or scheme == "neo4j":
        if parsed.username:
            os.environ["DIPEX_NEO4J_USER"] = parsed.username
        if parsed.password:
            os.environ["DIPEX_NEO4J_PASS"] = parsed.password

        return {
            "backend": "neo4j",
            "database": path_parts[0] if path_parts else (q.get("database", ["neo4j"])[0]),
            "neo4j_uri": f"{parsed.scheme or 'bolt'}://{parsed.hostname or 'localhost'}:{parsed.port or 7687}",
            "neo4j_cypher": q.get("cypher", ["MATCH (n) RETURN n LIMIT 5000"])[0],
            "username_env": "DIPEX_NEO4J_USER",
            "password_env": "DIPEX_NEO4J_PASS",
            "table_or_collection": q.get("label", [""])[0],
        }

    backend = scheme or "postgres"
    if parsed.username:
        os.environ["DIPEX_DB_USER"] = parsed.username
    if parsed.password:
        os.environ["DIPEX_DB_PASS"] = parsed.password

    database = path_parts[0] if path_parts else q.get("database", [""])[0]
    table = q.get("table", [path_parts[1] if len(path_parts) > 1 else ""])[0]
    return {
        "backend": backend,
        "host": parsed.hostname or "localhost",
        "port": parsed.port,
        "database": database,
        "table_or_collection": table,
        "query": q.get("query", [""])[0],
        "username_env": "DIPEX_DB_USER",
        "password_env": "DIPEX_DB_PASS",
    }


def _api_cfg_from_input(source_input: str) -> Dict[str, Any]:
    if source_input.strip().startswith("{"):
        parsed = json.loads(source_input)
        if not isinstance(parsed, dict):
            raise ValueError("API input JSON must be an object")
        if "url" not in parsed:
            raise ValueError("API input JSON requires 'url'")
        return parsed
    return {"url": source_input.strip(), "method": "GET"}


def _stream_cfg_from_input(source_input: str, config: Dict[str, Any]) -> Dict[str, Any]:
    topic = source_input.strip()
    brokers = config.get("streaming", {}).get("kafka_bootstrap", os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))

    if topic.startswith("{"):
        parsed = json.loads(topic)
        if not isinstance(parsed, dict):
            raise ValueError("Live input JSON must be an object")
        return {
            "brokers": parsed.get("brokers", brokers),
            "topic": parsed.get("topic", "events"),
            "group_id": parsed.get("group_id", "dipex-consumer"),
            "max_messages": int(parsed.get("max_messages", 10000)),
        }

    if "/" in topic and ":" in topic.split("/")[0]:
        maybe_brokers, maybe_topic = topic.split("/", 1)
        brokers = maybe_brokers
        topic = maybe_topic

    return {
        "brokers": brokers,
        "topic": topic or "events",
        "group_id": "dipex-consumer",
        "max_messages": 10000,
    }


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


@router.post(
    "/pipeline/simple-run",
    summary="Zero-config run: ingest one source and execute full DIPEX pipeline",
    response_model=None,
)
async def pipeline_simple_run(
    source_kind: str = Form("file", description="file|database|graph_db|api|live"),
    source_input: str = Form("", description="URI/topic/url or optional JSON config"),
    dataset_id: str = Form("", description="Optional dataset label"),
    target_col: str = Form("", description="Optional target column; auto-detected when empty"),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """
    Minimal one-click endpoint used by simplified UI.

    Accepts a single source definition and automatically runs:
      intake -> ISSF snapshot -> full pipeline bridge -> formatted final result
    """
    from ingestion.pipeline_bridge import PipelineBridge
    from ingestion.universal_intake import SourceConfig, UniversalIntake
    from ingestion.readers.stream_reader import KafkaSourceConfig, WindowConfig

    config = _load_config()
    run_id = str(uuid.uuid4())
    source_kind = (source_kind or "file").strip().lower()
    dataset_id = (dataset_id or "").strip()

    intake = UniversalIntake.from_yaml("config.yaml") if os.path.exists("config.yaml") else UniversalIntake(config=config)
    tmp_path = ""

    try:
        if source_kind == "file":
            if file is None:
                raise HTTPException(status_code=400, detail="For source_kind='file', please provide a file")

            dataset_id = dataset_id or os.path.splitext(file.filename or "upload")[0]
            suffix = os.path.splitext(file.filename or "")[1] or ".csv"
            tmp_path = os.path.join(tempfile.gettempdir(), f"dipex_simple_{run_id}{suffix}")

            with open(tmp_path, "wb") as f_buf:
                shutil.copyfileobj(file.file, f_buf)

            cfg = SourceConfig(
                source_type="file",
                dataset_id=dataset_id,
                data_mode="batch",
                path=tmp_path,
            )

        elif source_kind in ("database", "graph_db"):
            if not source_input.strip():
                raise HTTPException(status_code=400, detail="Provide source_input as connection URI")

            db_cfg = _db_cfg_from_uri(source_input, source_kind)
            dataset_id = dataset_id or f"{db_cfg.get('backend', 'database')}_dataset"
            cfg = SourceConfig(
                source_type="database",
                dataset_id=dataset_id,
                data_mode="batch",
                db_config=db_cfg,
            )

        elif source_kind == "api":
            if not source_input.strip():
                raise HTTPException(status_code=400, detail="Provide source_input as API URL or JSON config")

            api_cfg = _api_cfg_from_input(source_input)
            dataset_id = dataset_id or "api_live_dataset"
            cfg = SourceConfig(
                source_type="api",
                dataset_id=dataset_id,
                data_mode="live",
                api_config=api_cfg,
            )

        elif source_kind == "live":
            stream_cfg_dict = _stream_cfg_from_input(source_input, config)
            stream_cfg = KafkaSourceConfig(**stream_cfg_dict)
            dataset_id = dataset_id or f"live_{stream_cfg.topic}"
            cfg = SourceConfig(
                source_type="stream",
                dataset_id=dataset_id,
                data_mode="stream",
                stream_config=stream_cfg,
                window_config=WindowConfig(strategy="tumbling", window_size_s=30, watermark_delay_s=5),
                max_stream_windows=1,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid source_kind. Use one of: file, database, graph_db, api, live",
            )

        snapshot = intake.ingest(cfg)
        if snapshot.data is None or snapshot.data.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Source produced empty data for dataset '{snapshot.dataset_id}'. "
                    f"Validation status: {snapshot.validation_status}"
                ),
            )

        effective_target = target_col.strip() or _guess_target_col(snapshot.data)

        bridge = PipelineBridge(config=config)
        bridge_result = bridge.run(
            snapshot,
            target_col=effective_target,
            run_id=run_id,
            skip_stages=None,
        )

        summary = bridge_result.summary()
        return {
            "status": "ok",
            "run_id": run_id,
            "source_kind": source_kind,
            "dataset_id": snapshot.dataset_id,
            "snapshot_id": snapshot.snapshot_id,
            "final_result": {
                "gate_decision": bridge_result.gate_decision,
                "gate1_decision": bridge_result.gate1_decision,
                "gate2_decision": bridge_result.gate2_decision,
                "quality_score": snapshot.quality_score,
                "target_column_used": effective_target,
                "report_path": bridge_result.report_path,
                "model_metrics": bridge_result.model_metrics,
                "retry_count": bridge_result.retry_count,
                "confidence_vector": bridge_result.confidence_vector,
            },
            "stages": _stage_brief(summary.get("stages", [])),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("pipeline/simple-run failed for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
