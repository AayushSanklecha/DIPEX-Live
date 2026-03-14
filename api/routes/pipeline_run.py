"""
api/routes/pipeline_run.py
---------------------------
Pipeline execution endpoints:

  POST /api/pipeline/run         — Full pipeline run (upload file separately first)
  POST /api/pipeline/simple-run  — Unified: ingest + full 13-stage pipeline in one call

Supported source types (source_kind):
  file      — CSV, Excel, JSON, Parquet file upload
  database  — PostgreSQL / MongoDB / Neo4j via db_uri + db_table
  api       — REST API endpoint polling (api_url)
  live      — Kafka stream consumer (kafka_topic)

dataset_id is forwarded to the audit log and result API. If not provided by
the user it is auto-derived from the uploaded filename (for file mode) or from
the source descriptor (for other modes).

target_col is the supervised-learning label column. If omitted, _guess_target_col()
searches for common names (target, label, class, y, churn, fraud). If none found,
the pipeline still runs in unsupervised/EDA mode.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
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
    if uri.strip().startswith("{"):
        parsed = json.loads(uri)
        if not isinstance(parsed, dict):
            raise ValueError("Database input JSON must be an object")
        
        if source_kind == "graph_db" or parsed.get("backend") == "neo4j":
            if parsed.get("username"):
                os.environ["DIPEX_NEO4J_USER"] = parsed.get("username")
            if parsed.get("password"):
                os.environ["DIPEX_NEO4J_PASS"] = parsed.get("password")
            
            return {
                "backend": "neo4j",
                "database": parsed.get("database", "neo4j"),
                "neo4j_uri": f"{parsed.get('scheme', 'bolt')}://{parsed.get('host', 'localhost')}:{parsed.get('port', 7687)}",
                "neo4j_cypher": parsed.get("query", "MATCH (n) RETURN n LIMIT 5000"),
                "username_env": "DIPEX_NEO4J_USER",
                "password_env": "DIPEX_NEO4J_PASS",
                "table_or_collection": parsed.get("table", ""),
            }
        
        backend = parsed.get("backend", "postgres")
        if parsed.get("username"):
            os.environ["DIPEX_DB_USER"] = parsed.get("username")
        if parsed.get("password"):
            os.environ["DIPEX_DB_PASS"] = parsed.get("password")
            
        return {
            "backend": backend,
            "host": parsed.get("host", "localhost"),
            "port": parsed.get("port"),
            "database": parsed.get("database", ""),
            "table_or_collection": parsed.get("table", ""),
            "query": parsed.get("query", ""),
            "username_env": "DIPEX_DB_USER",
            "password_env": "DIPEX_DB_PASS",
        }

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
    brokers = config.get("streaming", {}).get("kafka_bootstrap", os.getenv("KAFKA_BOOTSTRAP", "kafka:29092"))

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
    domain: str = Form(
        "",
        description=(
            "Primary data domain — determines which regulatory rules are enforced. "
            "Options: banking | healthcare | finance | gdpr | sox | hipaa | generic. "
            "Defaults to value in config.yaml when blank."
        ),
    ),
    extra_domains: str = Form(
        "",
        description=(
            "Comma-separated list of additional regulatory domains to layer on top. "
            "Example: 'gdpr,sox' to add GDPR + SOX rules on top of the primary domain."
        ),
    ),
) -> Dict[str, Any]:
    """
    **Unified ingest + pipeline endpoint.**

    1. Uploads the file to a temp path
    2. Runs `UniversalIntake.ingest()` → ISSFSnapshot
    3. Optionally saves the snapshot to ``data/snapshots/``
    4. Runs `PipelineBridge.run(snapshot)` — all 13 stages
    5. Returns the full ``PipelineResult.summary()`` + snapshot metadata

    This endpoint unifies the previously disconnected UDIL ingestion
    (``POST /ingest/file``) and pipeline execution (``POST /api/run``)\
    flows into a single, atomic operation.

    The ``domain`` field tells the compliance engine which regulatory
    rule set to enforce (banking → AML/Basel III, healthcare → HIPAA/PHI,
    gdpr → data residency/consent, etc.). When blank, ``config.yaml`` is used.
    """
    config = _load_config()
    run_id = str(uuid.uuid4())
    dataset_id = dataset_id or os.path.splitext(file.filename or "upload")[0]
    suffix = os.path.splitext(file.filename or "")[1] or ".csv"

    # ── Per-request domain override ───────────────────────────────────────────
    # Rule: if the user explicitly chose a domain, enforce only those rules.
    #       if the user left domain blank, SKIP all regulatory/compliance checks.
    _VALID_DOMAINS = {"banking", "healthcare", "finance", "gdpr", "sox", "hipaa", "generic"}

    primary = domain.strip().lower() if domain.strip() else None
    extra_list = [d.strip().lower() for d in extra_domains.split(",") if d.strip()]

    if primary:
        # User chose a domain — build effective list: primary + any extras
        domain_list = [primary] if primary in _VALID_DOMAINS else []
        for d in extra_list:
            if d in _VALID_DOMAINS and d not in domain_list:
                domain_list.append(d)
        config.setdefault("pipeline", {})["domain"] = domain_list[0] if domain_list else "generic"
        config.setdefault("validation", {}).setdefault("regulatory", {})["domains"] = domain_list
    else:
        # No domain selected → disable regulatory/compliance checks for this run
        config.setdefault("pipeline", {})["domain"] = "generic"
        config.setdefault("validation", {}).setdefault("regulatory", {})["domains"] = []

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
    domain: str = Form("", description="Primary regulatory domain"),
    extra_domains: str = Form("", description="Comma-separated secondary domains"),
    col_range: str = Form("", description="Optional range of columns to analyze (e.g. 1-10)"),
    row_range: str = Form("", description="Optional range of rows to analyze (e.g. 1-100)"),
    skip_stages: str = Form("drift_detection,experience_memory,rl_update", description="Comma-separated stage names to skip"),
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

    # ── Per-request domain override ───────────────────────────────────────────
    _VALID_DOMAINS = {"banking", "healthcare", "finance", "gdpr", "sox", "hipaa", "generic"}
    primary = domain.strip().lower() if domain.strip() else None
    extra_list = [d.strip().lower() for d in extra_domains.split(",") if d.strip()]

    if primary:
        domain_list = [primary] if primary in _VALID_DOMAINS else []
        for d in extra_list:
            if d in _VALID_DOMAINS and d not in domain_list:
                domain_list.append(d)
        config.setdefault("pipeline", {})["domain"] = domain_list[0] if domain_list else "generic"
        config.setdefault("validation", {}).setdefault("regulatory", {})["domains"] = domain_list
        # If domain selected, ensure governance is ON
        config.setdefault("validation", {}).setdefault("governance", {})["pii_detection"] = True
        if config["validation"]["governance"].get("policy", "off") == "off":
            config["validation"]["governance"]["policy"] = "flag"
    else:
        config.setdefault("pipeline", {})["domain"] = "generic"
        config.setdefault("validation", {}).setdefault("regulatory", {})["domains"] = []

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
            if not dataset_id:
                import hashlib
                host = urlparse(api_cfg.get("url", "")).hostname or "unknown"
                url_hash = hashlib.md5(api_cfg.get("url", "").encode("utf-8")).hexdigest()[:6]
                dataset_id = f"api_{host.replace('.', '_')}_{url_hash}"
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
                window_config=WindowConfig(strategy="tumbling", window_size_s=30, watermark_delay_s=86400),
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

        # ── Apply column range slicing if requested ──────────────────────────────
        if col_range.strip():
            try:
                import re
                val = col_range.strip()
                if re.match(r"^\d+\s*-\s*\d+$", val):
                    # Numeric range
                    parts = [int(p.strip()) for p in val.split("-")]
                    start_idx = max(0, parts[0] - 1) # 1-indexed to 0-indexed
                    end_idx = parts[1]
                    if snapshot.data is not None and not snapshot.data.empty:
                        # Slice data by columns
                        snapshot.data = snapshot.data.iloc[:, start_idx:end_idx].copy()
                        # Sync column metadata
                        if hasattr(snapshot, "column_metadata") and snapshot.column_metadata:
                            snapshot.column_metadata = snapshot.column_metadata[start_idx:end_idx]
                else:
                    # Specific column names
                    col_names = [c.strip() for c in val.split(",") if c.strip()]
                    if snapshot.data is not None and not snapshot.data.empty:
                        valid_cols = []
                        df_cols_lower = {str(c).lower(): c for c in snapshot.data.columns}
                        for c in col_names:
                            if c in snapshot.data.columns:
                                valid_cols.append(c)
                            elif c.lower() in df_cols_lower:
                                valid_cols.append(df_cols_lower[c.lower()])
                        
                        if valid_cols:
                            snapshot.data = snapshot.data[valid_cols].copy()
                            if hasattr(snapshot, "column_metadata") and snapshot.column_metadata:
                                snapshot.column_metadata = [cm for cm in snapshot.column_metadata if cm.name in valid_cols]
                        else:
                            logger.warning(f"None of the specified columns {col_names} found in data.")
            except Exception as _slice_exc:
                logger.warning(f"Failed to slice data by col_range '{col_range}': {_slice_exc}")

        # ── Apply row range slicing if requested ──────────────────────────────
        if row_range.strip():
            try:
                parts = [p.strip() for p in row_range.strip().split("-")]
                if len(parts) == 2:
                    start_idx = max(0, int(parts[0]) - 1) 
                    end_idx = int(parts[1])
                    if snapshot.data is not None and not snapshot.data.empty:
                        # Slice data by rows
                        snapshot.data = snapshot.data.iloc[start_idx:end_idx].copy()
                        snapshot.row_count = len(snapshot.data)
            except Exception as _slice_exc:
                logger.warning(f"Failed to slice data by row_range '{row_range}': {_slice_exc}")

        # ── Always save snapshot so results.py can load sample_rows ──────────
        snap_dir = config.get("storage", {}).get("snapshot_dir", "data/snapshots")
        try:
            snapshot.save(directory=snap_dir)
        except Exception as _save_exc:
            logger.warning("Snapshot save failed: %s", _save_exc)

        effective_target = target_col.strip() or _guess_target_col(snapshot.data)

        # ── Auto-Detect Regulatory Columns if Domain is Set ─────────────────
        reg_cfg = config.get("validation", {}).get("regulatory", {})
        domains_set = reg_cfg.get("domains", [])
        if "banking" in domains_set:
            banking_cfg = reg_cfg.setdefault("banking", {})
            
            # Auto-detect amount_columns
            if "amount_columns" not in banking_cfg:
                amt_matches = [c for c in snapshot.data.columns if "amount" in str(c).lower() or "balance" in str(c).lower()]
                if amt_matches:
                    banking_cfg["amount_columns"] = amt_matches
                
            # Auto-detect aml_amount_column
            if "aml_amount_column" not in banking_cfg:
                amt_matches = [c for c in snapshot.data.columns if "amount" in str(c).lower()]
                if amt_matches:
                    banking_cfg["aml_amount_column"] = amt_matches[0]
                    
            # Auto-detect velocity columns
            vel_cfg = banking_cfg.setdefault("velocity", {})
            if "transaction_id_column" not in vel_cfg:
                id_matches = [c for c in snapshot.data.columns if "account" in str(c).lower() or "customer" in str(c).lower() or "id" in str(c).lower()]
                if id_matches:
                    vel_cfg["transaction_id_column"] = id_matches[0]
            if "timestamp_column" not in vel_cfg:
                time_matches = [c for c in snapshot.data.columns if "date" in str(c).lower() or "time" in str(c).lower()]
                if time_matches:
                    vel_cfg["timestamp_column"] = time_matches[0]

        if "healthcare" in domains_set:
            hc_cfg = reg_cfg.setdefault("healthcare", {})
            if "age_column" not in hc_cfg:
                age_matches = [c for c in snapshot.data.columns if "age" in str(c).lower()]
                if age_matches:
                    hc_cfg["age_column"] = age_matches[0]

        skip_list = [s.strip() for s in skip_stages.split(",") if s.strip()] if skip_stages else None

        bridge = PipelineBridge(config=config)
        bridge_result = bridge.run(
            snapshot,
            target_col=effective_target,
            run_id=run_id,
            skip_stages=skip_list,
        )

        summary = bridge_result.summary()

        # ── Extract sample rows for dashboard visualisation ───────────────────
        sample_rows = []
        col_count = 0
        try:
            if snapshot.data is not None and not snapshot.data.empty:
                col_count = len(snapshot.data.columns)
                sample_df = snapshot.data.head(500)
                sample_rows = sample_df.where(sample_df.notna(), None).to_dict(orient="records")
        except Exception as _sample_exc:
            logger.warning("Could not extract sample rows: %s", _sample_exc)

        # ── Persist to audit log so Reports page can find this run ───────────
        os.makedirs("audit", exist_ok=True)
        audit_entry = {
            "event":             "PIPELINE_RUN",
            "run_id":            run_id,
            "dataset_id":        snapshot.dataset_id,
            "source_kind":       source_kind,
            "target_column_used": effective_target,
            "gate_decision":     bridge_result.gate_decision,
            "gate1_decision":    bridge_result.gate1_decision,
            "gate2_decision":    bridge_result.gate2_decision,
            "confidence_score":  (bridge_result.confidence_vector or {}).get("confidence_score", 0.0),
            "confidence_vector": bridge_result.confidence_vector,
            "quality_score":     snapshot.quality_score,
            "row_count":         snapshot.row_count,
            "col_count":         col_count,
            "retry_count":       bridge_result.retry_count,
            "snapshot_id":       snapshot.snapshot_id,
            "model_metrics":     bridge_result.model_metrics or {},
            "stages": [
                {
                    "name":        s.get("stage", s.get("name", f"Stage {i+1}")),
                    "status":      s.get("status", "UNKNOWN"),
                    "duration_ms": s.get("elapsed_ms"),
                    "reason":      s.get("reason", ""),
                }
                for i, s in enumerate(summary.get("stages", []))
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open("audit/audit.jsonl", "a", encoding="utf-8") as af:
                af.write(json.dumps(audit_entry) + "\n")
        except Exception as _audit_exc:
            logger.warning("Audit log write failed: %s", _audit_exc)

        return {
            "status":      "ok",
            "run_id":      run_id,
            "source_kind": source_kind,
            "dataset_id":  snapshot.dataset_id,
            "snapshot_id": snapshot.snapshot_id,
            "row_count":   snapshot.row_count,
            "col_count":   col_count,
            "sample_rows": sample_rows,
            "target_column_used": effective_target,
            "final_result": {
                "gate_decision":      bridge_result.gate_decision,
                "gate1_decision":     bridge_result.gate1_decision,
                "gate2_decision":     bridge_result.gate2_decision,
                "quality_score":      snapshot.quality_score,
                "confidence_score":   (bridge_result.confidence_vector or {}).get("confidence_score", 0.0),
                "confidence_vector":  bridge_result.confidence_vector,
                "target_column_used": effective_target,
                "report_path":        bridge_result.report_path,
                "model_metrics":      bridge_result.model_metrics,
                "retry_count":        bridge_result.retry_count,
                "governance_report":  bridge_result.governance_report,
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


@router.post(
    "/pipeline/list-tables",
    summary="List available tables/collections from a database source",
    response_model=None,
)
async def pipeline_list_tables(
    source_kind: str = Form(..., description="database or graph_db"),
    source_input: str = Form(..., description="URI or JSON config"),
) -> Dict[str, Any]:
    from ingestion.universal_intake import SourceConfig
    from ingestion.readers.db_reader import DBReader

    source_kind = (source_kind or "database").strip().lower()
    
    if source_kind not in ("database", "graph_db"):
        raise HTTPException(status_code=400, detail="Only database and graph_db support listing tables")
        
    if not source_input.strip():
        raise HTTPException(status_code=400, detail="Provide source_input as connection JSON config")

    try:
        db_cfg_dict = _db_cfg_from_uri(source_input, source_kind)
        # Create a dummy SourceConfig object just to satisfy the API
        cfg = SourceConfig(
            source_type=source_kind,
            dataset_id="temp_list",
            data_mode="batch",
            db_config=db_cfg_dict,
        )
        
        # We need the inner DBSourceConfig that is parsed inside UniversalIntake.
        # DBReader expects the parsed dataclass, not the raw dict.
        from ingestion.readers.db_reader import DBSourceConfig
        db_cfg = DBSourceConfig(**db_cfg_dict)
        
        reader = DBReader()
        tables = reader.list_tables(db_cfg)
        
        return {
            "status": "ok",
            "source_kind": source_kind,
            "tables": tables
        }
    except Exception as exc:
        logger.exception("pipeline/list-tables failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

