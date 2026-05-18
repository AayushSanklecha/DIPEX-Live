"""
api/routes/pipeline_run.py
---------------------------
Pipeline execution endpoints:

  POST /api/pipeline/run           — Full pipeline run (upload file separately first)
  POST /api/pipeline/simple-run    — Unified: ingest + full 13-stage pipeline in one call
  POST /api/pipeline/preview-plan  — Schema-only scan, returns ops plan for UI approval (2-5s)
  POST /api/pipeline/list-tables   — List tables/collections from a database source

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
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict, List, Optional


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


# ── PREVIEW PLAN ──────────────────────────────────────────────────────────────

def _int_or_none(v) -> Optional[int]:
    """Safely convert a value to int; returns None for missing/non-numeric input."""
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class _PreviewPlanRequest:
    """Pydantic-free request model parsed from JSON body."""
    def __init__(self, data: Dict[str, Any]):
        self.domain:            str            = data.get("domain", "generic") or "generic"
        self.target_col:        Optional[str]  = data.get("target_col") or None
        self.mode:              str            = data.get("mode", "auto") or "auto"
        self.user_context:      str            = data.get("user_context", "") or ""
        self.user_instructions: str            = data.get("user_instructions", "") or ""
        self.column_names:      List[str]      = data.get("column_names") or []
        self.n_rows:            Optional[int]  = data.get("n_rows")
        self.n_cols:            Optional[int]  = data.get("n_cols")
        self.null_rate:         Optional[float]= data.get("null_rate")
        self.plan_rejection_count: int         = int(data.get("plan_rejection_count", 0))
        # Actual type distribution computed from real data values by the frontend.
        # When present these override the name-keyword heuristic in pipeline_preview_plan.
        self.numeric_cols_count:    Optional[int] = _int_or_none(data.get("numeric_cols_count"))
        self.categorical_cols_count:Optional[int] = _int_or_none(data.get("categorical_cols_count"))
        self.temporal_cols_count:   Optional[int] = _int_or_none(data.get("temporal_cols_count"))
        self.text_cols_count:       Optional[int] = _int_or_none(data.get("text_cols_count"))


def _build_operations_plan(
    req: _PreviewPlanRequest,
    null_pct: float,
    parsed_hints: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Decide which pipeline operations are planned vs skipped based on context and parsed hints."""
    ops = []

    # Extract skip list from parsed hints (instruction-driven skips)
    hint_skip_stages: List[str] = []
    if parsed_hints is not None and hasattr(parsed_hints, "hints"):
        hint_skip_stages = parsed_hints.hints.skip_stages or []

    def _status(stage_key: str, default: str = "planned") -> str:
        return "skipped (per instruction)" if stage_key in hint_skip_stages else default

    # Always-on
    ops.append({"op": "pii_scan",         "label": "PII Scan",          "detail": "Email, SSN, Credit Card, Phone, ICD-10, IBAN detection", "status": _status("pii_scan")})
    ops.append({"op": "schema_validation", "label": "Schema Validation",  "detail": "Null rates, cardinality, type inference",                "status": "planned"})

    # Null-dependent
    if null_pct > 0.01:
        ops.append({"op": "null_imputation", "label": "Null Imputation",   "detail": f"Median / KNN imputation (null rate: {null_pct*100:.1f}%)", "status": "planned"})
    else:
        ops.append({"op": "null_imputation", "label": "Null Imputation",   "detail": "Skipped — dataset is clean (<1% nulls)",                 "status": "skipped"})

    # Outlier detection — label adjusted for hint
    outlier_hint = parsed_hints.hints.outlier_policy if parsed_hints and hasattr(parsed_hints, "hints") else None
    outlier_detail = {
        "preserve":   "Outliers preserved (per your instruction)",
        "quarantine": "Outliers quarantined / removed (per your instruction)",
        "flag":       "Outliers flagged for review (per your instruction)",
        "winsorize":  "Outliers capped via winsorization",
    }.get(outlier_hint or "", "IQR winsorize + IsolationForest anomaly score")
    ops.append({"op": "outlier_detection",   "label": "Outlier Detection",  "detail": outlier_detail,                                             "status": _status("anomaly_detection")})
    ops.append({"op": "feature_engineering", "label": "Feature Engineering", "detail": "DFS ratio/product pairs, polynomial, calendar, binning",   "status": _status("feature_engineering")})

    # Domain-specific
    effective_domain = req.domain
    if parsed_hints and hasattr(parsed_hints, "hints") and parsed_hints.hints.domain:
        effective_domain = parsed_hints.hints.domain
    if effective_domain and effective_domain not in ("generic", ""):
        ops.append({"op": "regulatory_compliance", "label": "Regulatory Compliance",
                    "detail": f"Domain: {effective_domain.upper()} rules enforced", "status": _status("regulatory_compliance")})
    else:
        ops.append({"op": "regulatory_compliance", "label": "Regulatory Compliance",
                    "detail": "Skipped — no domain selected", "status": "skipped"})

    # Supervised vs unsupervised (target from hint takes priority)
    effective_target = req.target_col
    if parsed_hints and hasattr(parsed_hints, "hints") and parsed_hints.hints.target_col:
        effective_target = parsed_hints.hints.target_col
    if effective_target:
        cv_folds = (parsed_hints.hints.cv_folds or 5) if parsed_hints and hasattr(parsed_hints, "hints") else 5
        ops.append({"op": "automl", "label": "AutoML",
                    "detail": f"XGBoost + LightGBM {cv_folds}-fold CV → predict '{effective_target}'",
                    "status": _status("modeling")})
    else:
        ops.append({"op": "unsupervised", "label": "Unsupervised Analysis",
                    "detail": "IsolationForest anomaly + K-Means clustering", "status": _status("modeling")})

    return ops


def _build_warnings(req: _PreviewPlanRequest, n_rows: int, n_cols: int, null_pct: float) -> List[Dict[str, str]]:
    """Generate contextual warnings from dataset metadata."""
    warns = []

    if n_rows < 100:
        warns.append({"level": "warning", "message": f"Very small dataset ({n_rows} rows) — model training quality may be limited."})
    elif n_rows > 5_000_000:
        warns.append({"level": "info", "message": f"Large dataset ({n_rows:,} rows) — chunked streaming and DuckDB merge will be used."})

    if null_pct > 0.40:
        warns.append({"level": "warning", "message": f"High null rate ({null_pct*100:.1f}%) — many rows may be quarantined."})

    if n_cols > 200:
        warns.append({"level": "info", "message": f"Wide dataset ({n_cols} cols) — DFS feature explosion is capped at 50 new features."})

    if req.target_col and req.target_col not in (req.column_names or []):
        if req.column_names:
            warns.append({"level": "warning", "message": f"Target column '{req.target_col}' not found in declared column list — will auto-detect at runtime."})

    if not req.domain or req.domain == "generic":
        warns.append({"level": "info", "message": "No regulatory domain selected — compliance checks will be skipped."})

    return warns


@router.post(
    "/pipeline/preview-plan",
    summary="Schema-only scan to preview what the pipeline will do before running",
    response_model=None,
)
async def pipeline_preview_plan(request_body: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    """
    **Pre-Analysis Plan endpoint** — returns a plan JSON the UI uses in the
    AnalysisPlanModal before the user approves a run.

    - Accepts column names, row/col counts, null rate — NO data uploaded.
    - Does NOT run any ML or preprocessing.
    - Returns in 2-5s (pure Python metadata logic).

    Response shape:
    ```json
    {
      "data_summary": { ... },
      "domain": { "active": "banking", "rules_count": 7, "rules": [...] },
      "operations": [ { "op": "pii_scan", "label": "...", "status": "planned" }, ... ],
      "warnings": [ { "level": "warning", "message": "..." } ],
      "plan_elapsed_ms": 42
    }
    ```
    """
    t0 = time.monotonic()
    req = _PreviewPlanRequest(request_body)

    # ── Parse analyst instructions ────────────────────────────────────────────
    parsed_hints = None
    instruction_summary: List[str] = []
    if req.user_instructions.strip():
        try:
            from api.routes.instruction_parser import parse_instructions
            parsed_hints = parse_instructions(req.user_instructions)
            instruction_summary = parsed_hints.summary
            # Override request fields with parsed hints where applicable
            if parsed_hints.hints.target_col and not req.target_col:
                req.target_col = parsed_hints.hints.target_col
            if parsed_hints.hints.domain and req.domain in ("", "generic"):
                req.domain = parsed_hints.hints.domain
        except Exception as _pe:
            logger.warning("Instruction parsing failed: %s", _pe)

    n_rows    = int(req.n_rows or 0)
    n_cols    = int(req.n_cols or len(req.column_names) or 0)
    null_rate = float(req.null_rate or 0.0)

    # Estimate drop/quarantine columns
    columns_to_drop: List[str] = []
    rows_quarantine_est = 0
    numeric_cols = 0
    categorical_cols = 0

    # ── Use actual type distribution from frontend's data analysis when available ──
    # The frontend runs inferType() on real column values and sends exact counts.
    # Only fall back to the name-keyword heuristic when no real data is provided
    # (e.g. database / Kafka / API sources where preview rows aren't available).
    if req.numeric_cols_count is not None:
        numeric_cols     = int(req.numeric_cols_count)
        categorical_cols = int(req.categorical_cols_count or 0)
    elif req.column_names:
        # Name-keyword heuristic (fallback for non-file sources only)
        date_kws = {"date", "time", "year", "month", "day", "created", "updated"}
        num_kws  = {"amount", "age", "price", "count", "total", "value", "score",
                    "rate", "qty", "revenue", "distance", "delay", "minutes", "id"}
        for col in req.column_names:
            cl = col.lower()
            if any(k in cl for k in date_kws):
                pass  # temporal — not counted as numeric or categorical
            elif any(k in cl for k in num_kws) or any(c.isdigit() for c in col):
                numeric_cols += 1
            else:
                categorical_cols += 1

    if null_rate > 0.9:
        rows_quarantine_est = int(n_rows * null_rate * 0.5)

    # Auto-detect domain from column names if not specified
    detected_domains: List[str] = []
    if req.column_names:
        banking_kws    = {"iban", "swift", "aml", "basel", "ltv", "collateral", "account_balance"}
        healthcare_kws = {"icd", "diagnosis", "patient", "clinical", "phi", "lab_result", "medication"}
        finance_kws    = {"portfolio", "nav", "volatility", "sharpe", "drawdown", "var", "mifid"}
        gdpr_kws       = {"consent", "data_subject", "residency", "retention", "erasure"}
        all_cols_lower = {c.lower() for c in req.column_names}

        if all_cols_lower & banking_kws:    detected_domains.append("banking")
        if all_cols_lower & healthcare_kws: detected_domains.append("healthcare")
        if all_cols_lower & finance_kws:    detected_domains.append("finance")
        if all_cols_lower & gdpr_kws:       detected_domains.append("gdpr")

    active_domain = req.domain if req.domain and req.domain != "generic" else (detected_domains[0] if detected_domains else "generic")

    # Build domain rule list
    domain_rules_map = {
        "banking":    ["AML Threshold", "Transaction Velocity", "LTV Ratio", "Basel III Capital", "Know Your Customer (KYC)"],
        "healthcare": ["HIPAA PHI Scan", "Patient Age Validation", "ICD-10 Code Integrity", "Data De-identification"],
        "finance":    ["SEC Reporting", "VaR Limits", "Portfolio Concentration", "MiFID II Trade Reporting"],
        "gdpr":       ["Consent Validation", "Data Residency", "Retention Period", "Right to Erasure"],
        "sox":        ["Audit Trail Completeness", "Segregation of Duties", "Change Management"],
        "hipaa":      ["PHI Encryption", "Access Control", "Audit Logging", "Backup Procedures"],
        "generic":    [],
    }
    active_rules = domain_rules_map.get(active_domain, [])

    ops      = _build_operations_plan(req, null_rate, parsed_hints)
    warnings = _build_warnings(req, n_rows, n_cols, null_rate)

    plan = {
        "data_summary": {
            "n_rows":                  n_rows,
            "n_cols":                  n_cols,
            "overall_null_pct":        null_rate * 100,
            "numeric_cols":            numeric_cols,
            "categorical_cols":        categorical_cols,
            "duplicate_rows":          0,   # cannot know without data
            "columns_to_drop":         columns_to_drop,
            "rows_to_quarantine_est":  rows_quarantine_est,
            "target_col":              req.target_col,
        },
        "domain": {
            "selected":    req.domain,
            "detected":    detected_domains,
            "active":      active_domain,
            "rules_count": len(active_rules),
            "rules":       active_rules,
        },
        "operations": ops,
        "warnings":   warnings,
        # ── Instruction intelligence ──────────────────────────────────────────
        "instruction_summary":      instruction_summary,
        "instruction_confidence":   round(parsed_hints.confidence, 3) if parsed_hints else 0.0,
        "instruction_hints":        parsed_hints.hints.to_dict() if parsed_hints else {},
        "plan_rejection_count":     req.plan_rejection_count,
        "plan_elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
    }
    return plan


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
    skip_stages: str = Form("", description="Comma-separated stage names to skip"),
    plan_approved: str = Form("false", description="Set 'true' if user approved the pre-analysis plan in the UI"),
    user_instructions: str = Form("", description="Free-text analyst instructions to guide the pipeline (max 500 chars)"),
    plan_rejection_count: int = Form(0, description="Number of times user rejected the plan before approving"),
) -> Dict[str, Any]:
    """
    Minimal one-click endpoint used by simplified UI.

    Accepts a single source definition and automatically runs:
      intake -> ISSF snapshot -> full pipeline bridge -> formatted final result

    When `plan_approved=true` is passed the audit log records that the user
    explicitly reviewed and approved the pre-analysis plan before execution.
    """
    from ingestion.pipeline_bridge import PipelineBridge
    from ingestion.universal_intake import SourceConfig, UniversalIntake
    from ingestion.readers.stream_reader import KafkaSourceConfig, WindowConfig

    config = _load_config()
    run_id = str(uuid.uuid4())
    source_kind = (source_kind or "file").strip().lower()
    dataset_id = (dataset_id or "").strip()
    _plan_approved = (plan_approved or "").strip().lower() == "true"

    # ── Parse analyst instructions ────────────────────────────────────────────
    _parsed_hints = None
    _instruction_summary: List[str] = []
    _user_instructions = (user_instructions or "").strip()[:500]
    if _user_instructions:
        try:
            from api.routes.instruction_parser import parse_instructions
            _parsed_hints = parse_instructions(_user_instructions)
            _instruction_summary = _parsed_hints.summary
            # Override form fields with instruction-parsed values (non-destructive)
            if not target_col.strip() and _parsed_hints.hints.target_col:
                target_col = _parsed_hints.hints.target_col
            if not domain.strip() and _parsed_hints.hints.domain:
                domain = _parsed_hints.hints.domain
            if not row_range.strip() and _parsed_hints.hints.row_range:
                row_range = _parsed_hints.hints.row_range
            # Merge instruction-level skip stages into skip_stages
            if _parsed_hints.hints.skip_stages:
                existing_skips = {s.strip() for s in skip_stages.split(",") if s.strip()}
                merged_skips = existing_skips | set(_parsed_hints.hints.skip_stages)
                skip_stages = ",".join(merged_skips)
            # Inject pipeline hints into config
            config = _parsed_hints.hints.apply_to_config(config)
            logger.info(
                "Instructions parsed — confidence=%.2f summary=%s",
                _parsed_hints.confidence, _instruction_summary,
            )
        except Exception as _pe:
            logger.warning("Instruction parsing skipped: %s", _pe)

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
            db_table = db_cfg.get("table_or_collection", "")
            safe_table = db_table if db_table else "dataset"
            dataset_id = dataset_id or f"{db_cfg.get('backend', 'database')}_{safe_table}"
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

        # ──────────────────────────────────────────────────────────────────────
        # Apply optional filters  (all three fields are fully independent;
        # any combination of 0 / 1 / 2 / 3 filled values is valid in every
        # source mode: file, database, kafka, api)
        # Order: row_range first (fewer rows to copy), then col_range.
        # target_col is ALWAYS preserved even if omitted from col_range.
        # ──────────────────────────────────────────────────────────────────────
        import re as _re

        # ── Row-range filter ──────────────────────────────────────────────────
        _rr = (row_range or "").strip().replace(" ", "")
        if _rr:
            try:
                _rr_match = _re.match(r"^(\d+)[\-\u2013](\d+)$", _rr)
                if _rr_match:
                    _r_start = max(0, int(_rr_match.group(1)) - 1)  # 1-indexed
                    _r_end   = int(_rr_match.group(2))
                    if snapshot.data is not None and not snapshot.data.empty:
                        snapshot.data = snapshot.data.iloc[_r_start:_r_end].reset_index(drop=True).copy()
                        snapshot.row_count = len(snapshot.data)
                        logger.info("[filter] row_range '%s' -> rows %d-%d (%d rows)",
                                    _rr, _r_start + 1, _r_end, snapshot.row_count)
                elif _rr.isdigit():
                    # Single number: treat as head(N)
                    if snapshot.data is not None and not snapshot.data.empty:
                        snapshot.data = snapshot.data.head(int(_rr)).copy()
                        snapshot.row_count = len(snapshot.data)
                else:
                    logger.warning("[filter] row_range '%s' not recognized — ignored (use '10-500')", _rr)
            except Exception as _rr_exc:
                logger.warning("[filter] row_range '%s' failed: %s — skipped", _rr, _rr_exc)

        # ── Column-range / column-name filter ─────────────────────────────────
        _cr = (col_range or "").strip()
        if _cr and snapshot.data is not None and not snapshot.data.empty:
            try:
                all_cols = list(snapshot.data.columns)
                keep_cols: List[str] = []

                _cr_clean = _cr.replace(" ", "")
                if _re.match(r"^\d+[\-\u2013]\d+$", _cr_clean):
                    # Numeric index range (1-indexed)
                    _cr_m = _re.match(r"^(\d+)[\-\u2013](\d+)$", _cr_clean)
                    _c_start = max(0, int(_cr_m.group(1)) - 1)
                    _c_end   = int(_cr_m.group(2))
                    keep_cols = all_cols[_c_start:_c_end]
                    logger.info("[filter] col_range '%s' -> cols %d-%d (%d cols)",
                                _cr, _c_start + 1, _c_end, len(keep_cols))
                else:
                    # Comma-separated column names (case-insensitive)
                    _col_names = [c.strip() for c in _cr.split(",") if c.strip()]
                    _cols_lower = {str(c).lower(): c for c in all_cols}
                    for _cn in _col_names:
                        if _cn in all_cols:
                            keep_cols.append(_cn)
                        elif _cn.lower() in _cols_lower:
                            keep_cols.append(_cols_lower[_cn.lower()])
                    if keep_cols:
                        logger.info("[filter] col_range '%s' -> named cols: %s", _cr, keep_cols)
                    else:
                        logger.warning("[filter] col_range '%s' — no matching columns, filter skipped", _cr)

                if keep_cols:
                    # Always re-inject target_col if it was excluded
                    _tc = target_col.strip() if target_col else None
                    if _tc and _tc in all_cols and _tc not in keep_cols:
                        keep_cols.append(_tc)
                        logger.info("[filter] target_col '%s' re-injected into col_range slice", _tc)
                    # Preserve original column order
                    keep_cols = [c for c in all_cols if c in keep_cols]
                    snapshot.data = snapshot.data[keep_cols].copy()
                    if hasattr(snapshot, "column_metadata") and snapshot.column_metadata:
                        snapshot.column_metadata = [
                            cm for cm in snapshot.column_metadata if cm.name in keep_cols
                        ]
            except Exception as _cr_exc:
                logger.warning("[filter] col_range '%s' failed: %s — skipped", _cr, _cr_exc)

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
            configured_amts = banking_cfg.get("amount_columns", [])
            if not any(c in snapshot.data.columns for c in configured_amts):
                amt_matches = [c for c in snapshot.data.columns if "amount" in str(c).lower() or "balance" in str(c).lower()]
                if amt_matches:
                    banking_cfg["amount_columns"] = amt_matches
                
            # Auto-detect aml_amount_column
            if banking_cfg.get("aml_amount_column") not in snapshot.data.columns:
                amt_matches = [c for c in snapshot.data.columns if "amount" in str(c).lower()]
                if amt_matches:
                    banking_cfg["aml_amount_column"] = amt_matches[0]
                    
            # Auto-detect velocity columns
            vel_cfg = banking_cfg.setdefault("velocity", {})
            if vel_cfg.get("transaction_id_column") not in snapshot.data.columns:
                id_matches = [c for c in snapshot.data.columns if "account" in str(c).lower() or "customer" in str(c).lower() or "id" in str(c).lower()]
                if id_matches:
                    vel_cfg["transaction_id_column"] = id_matches[0]
            if vel_cfg.get("timestamp_column") not in snapshot.data.columns:
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

        # ── Collect rich analytics data from bridge_result ────────────────────
        _analytics_result = bridge_result.analytics_result or {}
        _governance_report = bridge_result.governance_report or {}
        _regulatory_report_raw = bridge_result.regulatory_report or []

        # Feature importances — from model_metrics or analytics_result
        _feature_importances = {}
        mm = bridge_result.model_metrics or {}
        if mm.get("feature_importances"):
            _feature_importances = mm["feature_importances"]
        elif mm.get("feature_importance"):
            _feature_importances = mm["feature_importance"]
        elif _analytics_result.get("feature_importances"):
            _feature_importances = _analytics_result["feature_importances"]

        # Statistical tests — from analytics_result
        _statistical_tests = _analytics_result.get("statistical_tests", {})
        if not _statistical_tests:
            _statistical_tests = _analytics_result.get("stats", {})

        # Bias & fairness report — from analytics_result
        _bias_report = _analytics_result.get("bias_report", {})
        if not _bias_report:
            _bias_report = _analytics_result.get("bias_fairness", {})

        # Anomaly report — from analytics_result
        _anomaly_report = _analytics_result.get("anomaly_report", {})
        if not _anomaly_report:
            _anomaly_report = _analytics_result.get("anomaly_deep_dive", {})

        # Regulatory report — normalise from list → dict keyed by domain
        _regulatory_report: dict = {}
        if isinstance(_regulatory_report_raw, list):
            for item in _regulatory_report_raw:
                if isinstance(item, dict):
                    d = item.get("domain", "generic")
                    _regulatory_report.setdefault(d, {"violations": []})
                    _regulatory_report[d]["violations"].append(item)
        elif isinstance(_regulatory_report_raw, dict):
            _regulatory_report = _regulatory_report_raw
        # Also try analytics_result
        if not _regulatory_report:
            _regulatory_report = _analytics_result.get("regulatory_report", {})

        # RL agent summary — try to read from PPO agent checkpoint
        _rl_agent_summary: dict = {}
        try:
            from learning.rl_agent.agent import PPOAgent
            _ppo = PPOAgent.from_config(_load_config())
            _rl_agent_summary = {
                "episode_count":  _ppo._episode_count,
                "in_shadow_mode": _ppo.in_shadow_mode,
                "last_reward":    getattr(_ppo, "_last_reward", None),
                "recommended_action": None,
                "reward_components":  None,
            }
        except Exception:
            pass

        # ── Determine effective domain list for audit ─────────────────────────
        _domain_used = primary or ""           # primary domain string (e.g. "banking")
        _domain_list_used = domain_list if primary else []  # full list incl. extra domains

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
            "model_metrics":     mm,
            "plan_approved":     _plan_approved,
            # ── Regulatory domain tracking (always saved, even if 0 violations) ─
            "domain_used":       _domain_used,
            "domain_list_used":  _domain_list_used,
            # ── Rich analytics data (powers /api/analytics/{run_id}) ───────────
            "feature_importances": _feature_importances,
            "statistical_tests":   _statistical_tests,
            "bias_report":         _bias_report,
            "anomaly_report":      _anomaly_report,
            "regulatory_report":   _regulatory_report,
            "governance_report":   _governance_report,
            "rl_agent_summary":    _rl_agent_summary,
            # ── Instruction intelligence tracking ─────────────────────────────
            "user_instructions":      _user_instructions,
            "instruction_summary":    _instruction_summary,
            "instruction_confidence": round(_parsed_hints.confidence, 3) if _parsed_hints else 0.0,
            "plan_rejection_count":   plan_rejection_count,
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
                af.write(json.dumps(audit_entry, default=str) + "\n")
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
            "plan_approved": _plan_approved,
            # ── Instruction intelligence ──────────────────────────────────────
            "user_instructions":      _user_instructions,
            "instruction_summary":    _instruction_summary,
            "instruction_confidence": round(_parsed_hints.confidence, 3) if _parsed_hints else 0.0,
            "plan_rejection_count":   plan_rejection_count,
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


@router.post(
    "/pipeline/preview-source",
    summary="Generic preview to extract sample rows and schema metadata for DB, Kafka, and API without running the full pipeline",
    response_model=None,
)
async def pipeline_preview_source(
    source_kind: str = Form("database", description="database|graph_db|api|live"),
    source_input: str = Form("", description="URI/topic/url or optional JSON config"),
) -> Dict[str, Any]:
    """
    Unified preview endpoint that safely connects to a Database, API, or Kafka stream,
    fetches a maximum of 500 rows, and returns the real rows + metadata (total row count, null rate).
    """
    from ingestion.universal_intake import SourceConfig
    from ingestion.readers.db_reader import DBReader, DBSourceConfig
    from ingestion.readers.api_reader import APIReader, APISourceConfig
    from ingestion.readers.stream_reader import StreamReader, KafkaSourceConfig, WindowConfig
    
    source_kind = (source_kind or "database").strip().lower()
    if not source_input.strip():
        raise HTTPException(status_code=400, detail="Provide source_input")

    config = _load_config()

    df = None
    errors = []
    
    try:
        if source_kind in ("database", "graph_db"):
            db_cfg_dict = _db_cfg_from_uri(source_input, source_kind)
            db_cfg = DBSourceConfig(**db_cfg_dict)
            reader = DBReader()
            result = reader.read(db_cfg)
            if result and result.data is not None:
                df = result.data.head(500)
                errors = result.errors
                
        elif source_kind == "api":
            api_cfg_dict = _api_cfg_from_input(source_input)
            api_cfg = APISourceConfig(**api_cfg_dict)
            reader = APIReader()
            result = reader.read(api_cfg)
            if result and result.data is not None:
                df = result.data.head(500)
                errors = result.errors
                
        elif source_kind == "live":
            stream_cfg_dict = _stream_cfg_from_input(source_input, config)
            stream_cfg = KafkaSourceConfig(**stream_cfg_dict)
            window_cfg = WindowConfig(strategy="tumbling", window_size_s=5, watermark_delay_s=5)
            reader = StreamReader()
            
            # Fetch exactly 1 window for preview
            snapshots = []
            for snap in reader.read_kafka(stream_cfg, window_cfg, max_windows=1):
                if snap and snap.data is not None and not snap.data.empty:
                    snapshots.append(snap.data)
            
            if snapshots:
                import pandas as pd
                df = pd.concat(snapshots, ignore_index=True).head(500)
                
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source_kind for preview: {source_kind}")

        if df is None:
            return {
                "status": "empty",
                "rows": [],
                "n_rows": 0,
                "null_rate": 0,
                "errors": [str(e) for e in errors]
            }
            
        # Compute exact metadata needed for Pre-Analysis Plan
        total_rows = len(df)
        total_nulls = df.isna().sum().sum()
        total_cells = df.size
        null_rate = float(total_nulls / total_cells) if total_cells > 0 else 0.0

        sample_rows = df.where(df.notna(), None).to_dict(orient="records")

        return {
            "status": "ok",
            "rows": sample_rows,
            "n_rows": total_rows, # Extrapolating beyond 500 would require counting the full source, we return sample count here
            "null_rate": null_rate,
            "errors": [str(e) for e in errors]
        }
        
    except Exception as exc:
        logger.exception("pipeline/preview-source failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


