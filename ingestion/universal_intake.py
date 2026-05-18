"""
ingestion/universal_intake.py
-------------------------------
Universal Data Intake & Processing Layer — main orchestrator.

Single entry point for ALL ingestion modes.
Routes to correct reader, normalises, runs schema registry,
applies quality gate, and returns an ISSF snapshot.

Usage::

    from ingestion.universal_intake import UniversalIntake, SourceConfig

    intake = UniversalIntake(config)
    snapshot = intake.ingest(SourceConfig(
        source_type="file",
        dataset_id="sales",
        path="data/sales.csv",
    ))
    assert snapshot.is_compliant
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ingestion.error_handler import (
    ErrorAggregator, QualityGateError, SafeExecutor, SchemaError,
)
from ingestion.issf import ColumnMeta, ISSFSnapshot
from ingestion.normaliser import Normaliser
from ingestion.quality_gate import QualityGate, QualityReport
from ingestion.schema_registry import SchemaRegistry
from ingestion.adaptive_learner import AdaptiveLearner, IngestionOutcome
from ingestion.schema_infer import SmartSchemaInferer
from validation.governance.governor import DataGovernor, GovernanceError

logger = logging.getLogger("dipex.ingestion.universal_intake")


# ── Source Config ─────────────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    """
    Unified source config that routes to the correct reader.

    source_type : 'file' | 'api' | 'database' | 'stream'
    dataset_id  : stable human-readable identifier for this dataset
    data_mode   : 'batch' | 'live' | 'stream'
    """
    source_type: str                   = "file"
    dataset_id: str                    = ""
    data_mode: str                     = "batch"

    # File source
    path: Optional[str]                = None
    file_format: Optional[str]         = None
    sheet_name: Optional[str]          = None

    # API source (APISourceConfig dict or object)
    api_config: Optional[Any]          = None

    # DB source (DBSourceConfig dict or object)
    db_config: Optional[Any]           = None

    # Stream source (KafkaSourceConfig + WindowConfig dicts or objects)
    stream_config: Optional[Any]       = None
    window_config: Optional[Any]       = None
    max_stream_windows: int            = 10

    # Quality gate overrides
    range_rules: Optional[Dict[str, Tuple[float, float]]] = None
    allowed_categories: Optional[Dict[str, List[str]]]    = None
    expected_dtypes: Optional[Dict[str, str]]             = None
    fk_rules: Optional[Dict[str, Set]]                    = None
    baseline_snapshot_id: Optional[str]                   = None

    # Governance
    require_quality_pass: bool         = False  # Data always flows; quality issues logged as warnings
    block_on_schema_break: bool        = False  # Schema breaks are advisory — data still flows through

    extra: Dict[str, Any]              = field(default_factory=dict)


# ── UniversalIntake ───────────────────────────────────────────────────────────

class UniversalIntake:
    """
    Orchestrates the full ingestion pipeline:

    Reader → Normaliser → SchemaRegistry → QualityGate → ISSFSnapshot

    Parameters
    ----------
    config : dict loaded from config.yaml (or empty dict for defaults)
    snapshot_dir : directory to persist ISSF snapshots
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        snapshot_dir: str = "data/snapshots",
        registry_dir: str = "data/schema_registry",
        kb_path: str = "data/adaptive_kb.json",
        layer_base_dir: str = "data",
    ) -> None:
        self.config        = config or {}
        self.snapshot_dir  = snapshot_dir
        self.normaliser    = Normaliser()
        self.schema_reg    = SchemaRegistry(registry_dir=registry_dir)
        self.quality_gate  = QualityGate(config=self.config)
        self.governor      = DataGovernor(config=self.config)
        self.learner       = AdaptiveLearner(kb_path=kb_path)
        self.schema_inferer = SmartSchemaInferer()   # [ML] semantic type classifier
        intake_cfg         = self.config.get("universal_intake", {})
        self.chunk_size    = int(intake_cfg.get("chunk_size", 50_000))
        os.makedirs(snapshot_dir, exist_ok=True)

        # ── Layer isolation: Bronze/Silver/Gold manager ───────────────────────
        dl_cfg = self.config.get("data_layers", {})
        _base = dl_cfg.get("bronze_dir", layer_base_dir).replace("/bronze", "") or layer_base_dir
        try:
            from ingestion.data_layers import LayerManager
            self.layer_manager: Optional[Any] = LayerManager(base_dir=_base)
        except Exception:  # noqa: BLE001
            self.layer_manager = None
            logger.warning("LayerManager unavailable — layer isolation disabled")

    # ── Public entry point ────────────────────────────────────────────────────

    def ingest(self, cfg: SourceConfig) -> ISSFSnapshot:
        """
        Run the full ingestion pipeline and return an ISSF snapshot.
        Raises on BREAKING schema drift (configurable) or quality FAILED.
        Before starting: consults AdaptiveLearner for recommended strategy.
        After finishing: records outcome so learner improves for next time.
        """
        correlation_id = str(uuid.uuid4())
        executor = SafeExecutor(dataset_id=cfg.dataset_id, source_type=cfg.source_type)
        t0 = time.perf_counter()

        # ── Pre-flight: get strategy recommendation from past learning ─────────
        recommendation = self.learner.recommend(
            dataset_id=cfg.dataset_id,
            source_type=cfg.source_type,
            hints={"path": cfg.path, "format": cfg.file_format},
        )
        if recommendation.recommended_format and not cfg.file_format:
            cfg.file_format = recommendation.recommended_format
            logger.info("[AdaptiveLearner] Using learnt format: %s", cfg.file_format)
        if recommendation.warnings:
            for w in recommendation.warnings:
                logger.warning("[AdaptiveLearner] Pre-flight warning: %s", w)

        logger.info(
            "[%s] Ingestion started — dataset=%s source_type=%s mode=%s (learner_confidence=%.0f%%)",
            correlation_id[:8], cfg.dataset_id, cfg.source_type, cfg.data_mode,
            recommendation.confidence * 100,
        )

        # ── Step 1: Read ──────────────────────────────────────────────────────
        df, read_errors, source_uri = self._read(cfg, executor)
        if df is None:
            return self._failure_snapshot(cfg, read_errors, correlation_id)

        ingestion_errors = list(read_errors)

        # ── Step 1b: Bronze Layer — lock raw data immediately ─────────────────
        _bronze_id: Optional[str] = None
        if self.layer_manager is not None:
            try:
                _bronze = self.layer_manager.store_bronze(
                    df, cfg.dataset_id, correlation_id, component="ingestion"
                )
                _bronze_id = _bronze.checksum[:16]
                logger.info(
                    "[%s] Bronze locked — dataset=%s checksum=%s…",
                    correlation_id[:8], cfg.dataset_id, _bronze_id,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning("Bronze layer storage failed (non-fatal): %s", _e)

        # ── Step 2: Normalise ─────────────────────────────────────────────────
        try:
            df, column_meta = self.normaliser.normalise(df, dataset_id=cfg.dataset_id)
        except Exception as _norm_exc:  # noqa: BLE001
            logger.error(
                "[%s] Normaliser crashed (non-fatal, using raw data): %s",
                correlation_id[:8], _norm_exc,
            )
            column_meta = []
            from ingestion.issf import IngestionError as IE
            ingestion_errors.append(IE(
                error_type="NORMALISATION_ERROR",
                message=f"Normaliser failed: {_norm_exc}",
                severity="ERROR",
                correlation_id=correlation_id,
            ))

        # ── [ML] Semantic Schema Enrichment ──────────────────────────────────
        try:
            _raw_schema = {c.name: c.dtype for c in column_meta}
            _enriched   = self.schema_inferer.enrich_schema(df, _raw_schema)
            for cm in column_meta:
                _e = _enriched.get(cm.name, {})
                cm.extra_meta = getattr(cm, "extra_meta", {}) or {}
                cm.extra_meta["semantic_type"] = _e.get("semantic_type", "unknown")
                cm.extra_meta["nlp_tags"]      = _e.get("nlp_tags", [])
                cm.extra_meta["ml_confidence"] = _e.get("confidence", 0.0)
        except Exception as _ml_exc:  # noqa: BLE001
            logger.warning(
                "[%s] Schema enrichment failed (non-fatal): %s",
                correlation_id[:8], _ml_exc,
            )

        schema = {c.name: c.dtype for c in column_meta}

        # ── Step 2b: Silver Layer — promote normalised snapshot ───────────────
        _silver_id: Optional[str] = None
        if self.layer_manager is not None:
            try:
                from ingestion.data_layers import ImmutableDataFrame
                _bronze_imm = ImmutableDataFrame(
                    df.copy(), layer="bronze", dataset_id=cfg.dataset_id
                )
                _silver = self.layer_manager.promote_to_silver(
                    _bronze_imm, df, cfg.dataset_id, correlation_id,
                    component="normaliser",
                    tags={"schema_version": "", "source_type": cfg.source_type},
                )
                _silver_id = _silver.checksum[:16]
                logger.info(
                    "[%s] Silver promoted — dataset=%s checksum=%s…",
                    correlation_id[:8], cfg.dataset_id, _silver_id,
                )
            except Exception as _e:  # noqa: BLE001
                logger.warning("Silver layer promotion failed (non-fatal): %s", _e)

        # ── Step 3: Schema Registry ───────────────────────────────────────────
        drift_report = None
        schema_version = "0.0.0"
        try:
            drift_report = self.schema_reg.register(
                dataset_id=cfg.dataset_id,
                schema=schema,
                row_count=len(df),
                source_uri=source_uri or "",
            )
            schema_version = drift_report.new_version

            if drift_report.is_breaking and cfg.block_on_schema_break:
                raise SchemaError(
                    f"BREAKING schema drift in dataset '{cfg.dataset_id}': {drift_report.summary}",
                    correlation_id=correlation_id,
                )
            if drift_report.is_breaking:
                from ingestion.issf import IngestionError as IE
                ingestion_errors.append(IE(
                    error_type="SCHEMA_ERROR",
                    message=drift_report.summary,
                    severity="ERROR",
                    correlation_id=correlation_id,
                ))
        except SchemaError:
            raise  # re-raise if user wants to block on schema break
        except Exception as _reg_exc:  # noqa: BLE001
            logger.error(
                "[%s] Schema registry crashed (non-fatal, using v0.0.0): %s",
                correlation_id[:8], _reg_exc,
            )
            from ingestion.issf import IngestionError as IE
            ingestion_errors.append(IE(
                error_type="SCHEMA_ERROR",
                message=f"Schema registry failed: {_reg_exc}",
                severity="ERROR",
                correlation_id=correlation_id,
            ))

        # ── Step 3b: Active Data Governance (PII Scan) ───────────────────────
        try:
            df, gov_report = self.governor.enforce(df, dataset_id=cfg.dataset_id)
            if gov_report.get("status") == "redacted":
                logger.warning("[%s] Governance redaction applied — %d PII elements stripped.",
                               correlation_id[:8], gov_report.get("total_redactions", 0))
                from ingestion.issf import IngestionError as IE
                ingestion_errors.append(IE(
                    error_type="GOVERNANCE_REDACT",
                    message=f"PII Redaction Applied. Stripped {gov_report.get('total_redactions', 0)} entities.",
                    severity="WARN",
                    correlation_id=correlation_id,
                ))
            elif gov_report.get("status") == "flagged":
                # Add warning logs but don't halt
                from ingestion.issf import IngestionError as IE
                for pii_type, cols in gov_report.get("pii_hits", {}).items():
                    ingestion_errors.append(IE(
                        error_type="GOVERNANCE_FLAG",
                        message=f"PII Flagged: Column '{pii_type}' contains {cols} entities.",
                        severity="WARN",
                        correlation_id=correlation_id,
                    ))

        except GovernanceError as gov_exc:
            from ingestion.issf import IngestionError as IE
            ingestion_errors.append(IE(
                error_type="GOVERNANCE_ERROR",
                message=str(gov_exc),
                severity="CRITICAL",
                correlation_id=correlation_id,
            ))
            # Wrap as a fatal error returning failure snapshot
            logger.error("[%s] Governance policy rejected dataset: %s", correlation_id[:8], gov_exc)
            return self._failure_snapshot(cfg, ingestion_errors, correlation_id)

        # ── Step 4: Quality Gate ──────────────────────────────────────────────
        try:
            baseline_df = self._load_baseline(cfg.baseline_snapshot_id)
            quality_report = self.quality_gate.check(
                df=df,
                dataset_id=cfg.dataset_id,
                snapshot_id=correlation_id,
                range_rules=cfg.range_rules,
                allowed_categories=cfg.allowed_categories,
                expected_dtypes=cfg.expected_dtypes,
                baseline_df=baseline_df,
                fk_rules=cfg.fk_rules,
            )
        except Exception as _qg_exc:  # noqa: BLE001
            logger.error(
                "[%s] Quality gate crashed (non-fatal, defaulting to WARN): %s",
                correlation_id[:8], _qg_exc,
            )
            quality_report = QualityReport(
                dataset_id=cfg.dataset_id, snapshot_id=correlation_id,
                quality_score=0.5, validation_status="WARN",
                row_count=len(df), duplicate_count=0, duplicate_rate=0.0,
                overall_null_rate=0.0, column_quality=[], distribution_drift={},
                violations=[], warnings=[f"Quality gate error: {_qg_exc}"],
            )
            from ingestion.issf import IngestionError as IE
            ingestion_errors.append(IE(
                error_type="QUALITY_GATE_ERROR",
                message=f"Quality gate failed: {_qg_exc}",
                severity="ERROR",
                correlation_id=correlation_id,
            ))

        if quality_report.validation_status == "FAILED" and cfg.require_quality_pass:
            raise QualityGateError(
                f"Quality gate FAILED for dataset '{cfg.dataset_id}' "
                f"(score={quality_report.quality_score:.2f}): "
                + "; ".join(quality_report.violations[:3]),
                correlation_id=correlation_id,
            )

        # Add quality warnings to error log
        for w in quality_report.warnings:
            from ingestion.issf import IngestionError as IE
            ingestion_errors.append(IE(
                error_type="QUALITY_WARN", message=w, severity="WARN",
                correlation_id=correlation_id,
            ))

        # ── Step 5: Build ISSF snapshot ───────────────────────────────────────
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        snapshot = ISSFSnapshot(
            dataset_id=cfg.dataset_id,
            snapshot_id=correlation_id,
            schema_version=schema_version,
            data_mode=cfg.data_mode,
            source_type=cfg.source_type,
            source_uri=source_uri or cfg.path or "",
            column_metadata=column_meta,
            row_count=len(df),
            quality_score=quality_report.quality_score,
            validation_status=quality_report.validation_status,
            error_logs=ingestion_errors,
            data=df,
            extra_meta={
                "correlation_id": correlation_id,
                "read_time_ms": elapsed_ms,
                "schema_drift_summary": drift_report.summary,
                "schema_drift_breaking": drift_report.is_breaking,
                "duplicate_count": quality_report.duplicate_count,
                "overall_null_rate": quality_report.overall_null_rate,
                # ── Layer isolation metadata ──────────────────────────────────
                "bronze_checksum": _bronze_id,
                "silver_checksum": _silver_id,
                "layer_isolation_enabled": self.layer_manager is not None,
            },
        )

        try:
            snapshot.save(self.snapshot_dir)
        except Exception as _save_exc:  # noqa: BLE001
            logger.error(
                "[%s] Snapshot save failed (non-fatal): %s",
                correlation_id[:8], _save_exc,
            )

        logger.info(
            "[%s] Ingestion complete — %d rows, schema v%s, quality=%.2f, status=%s, elapsed=%.0fms",
            correlation_id[:8], len(df), schema_version,
            quality_report.quality_score, quality_report.validation_status, elapsed_ms,
        )

        # ── Post-flight: teach the learner ────────────────────────────────────
        try:
            _drift_breaking = drift_report.is_breaking if drift_report else False
            _drift_type = (
                drift_report.changes[0].change_type
                if drift_report and drift_report.changes else None
            )
            outcome = IngestionOutcome(
                dataset_id=cfg.dataset_id,
                source_type=cfg.source_type,
                success=True,
                quality_score=quality_report.quality_score,
                validation_status=quality_report.validation_status,
                row_count=len(df),
                schema_version=schema_version,
                elapsed_ms=elapsed_ms,
                strategy_used={
                    "format": cfg.file_format or "auto",
                    "encoding": "auto",
                    "delimiter": "auto",
                },
                quality_issues=[v for v in quality_report.violations],
                schema_drift=_drift_breaking,
                schema_drift_type=_drift_type,
                source_uri=cfg.path or "",
            )
            self.learner.record(outcome)
        except Exception as _learn_exc:  # noqa: BLE001
            logger.warning(
                "[%s] Learner record failed (non-fatal): %s",
                correlation_id[:8], _learn_exc,
            )

        return snapshot

    # ── Batch mode (multiple files or DB tables) ──────────────────────────────

    def ingest_batch(self, configs: List[SourceConfig]) -> List[ISSFSnapshot]:
        """Ingest multiple sources. Failures are isolated — other sources continue."""
        results: List[ISSFSnapshot] = []
        for i, cfg in enumerate(configs):
            try:
                snap = self.ingest(cfg)
                results.append(snap)
                logger.info("Batch [%d/%d] completed: %s", i + 1, len(configs), cfg.dataset_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch [%d/%d] failed: %s — %s", i + 1, len(configs), cfg.dataset_id, exc)
        return results

    # ── Internal: Reader dispatch ─────────────────────────────────────────────

    def _read(
        self, cfg: SourceConfig, executor: SafeExecutor
    ) -> Tuple[Optional[pd.DataFrame], List, str]:
        """Route to the correct reader and return (df, errors, source_uri)."""
        source_type = cfg.source_type.lower()

        if source_type == "file":
            df, errors, uri = self._read_file(cfg, executor)
        elif source_type == "api":
            df, errors, uri = self._read_api(cfg, executor)
        elif source_type == "database":
            df, errors, uri = self._read_db(cfg, executor)
        elif source_type == "stream":
            df, errors, uri = self._read_stream(cfg, executor)
        else:
            raise ValueError(f"Unknown source_type: {cfg.source_type!r}")

        return df, errors, uri

    def _read_file(self, cfg: SourceConfig, executor: SafeExecutor):
        from ingestion.readers.file_reader import FileReader
        from ingestion.data_rescue import rescue_dataframe
        reader = FileReader(chunk_size=self.chunk_size)

        def _do_read():
            return reader.read(cfg.path or "", fmt=cfg.file_format, sheet_name=cfg.sheet_name)

        result, errors = executor.run(_do_read)
        df_raw = result.data if result is not None else None
        read_errors = (errors + result.errors) if result is not None else errors

        if df_raw is not None and not df_raw.empty:
            # ── DataRescue pass: fix any structural issues ───────────────────
            df_rescued, rescue_rpt = rescue_dataframe(
                df_raw, source_type="file", context={"path": cfg.path}
            )
            if rescue_rpt.was_rescued:
                from ingestion.issf import IngestionError as IE
                for w in rescue_rpt.warnings:
                    read_errors.append(IE(error_type="QUALITY_WARN", message=f"[DataRescue] {w}",
                                          severity="WARN", correlation_id="rescue"))
            return df_rescued, read_errors, cfg.path or ""

        # ── Primary reader failed or empty: invoke universal fallback cascade ─
        logger.warning(
            "Primary FileReader failed or returned empty — activating UniversalFallbackReader cascade"
        )
        try:
            from ingestion.readers.universal_fallback import UniversalFallbackReader
            fallback = UniversalFallbackReader(max_fallback_rows=self.chunk_size * 10)
            rec = self.learner.recommend(cfg.dataset_id, "file", hints={"path": cfg.path})
            fb_result = fallback.read(
                cfg.path or "",
                hint_format=cfg.file_format or rec.recommended_format,
                hint_encoding=rec.recommended_encoding,
            )
            logger.info(
                "UniversalFallbackReader succeeded — format=%s rows=%d partial=%s",
                fb_result.format_detected, fb_result.row_count, fb_result.is_partial,
            )
            if fb_result.strategy_used:
                cfg.file_format = fb_result.format_detected

            from ingestion.issf import IngestionError as IE
            fallback_errs = read_errors + [
                IE(error_type="QUALITY_WARN", message=w,
                   severity="WARN", correlation_id="fallback")
                for w in fb_result.warnings
            ]
            # ── DataRescue pass on fallback result ───────────────────────────
            df_rescued, rescue_rpt = rescue_dataframe(
                fb_result.data, source_type="file", context={"path": cfg.path}
            )
            for w in rescue_rpt.warnings:
                fallback_errs.append(IE(error_type="QUALITY_WARN", message=f"[DataRescue] {w}",
                                        severity="WARN", correlation_id="rescue"))
            return df_rescued, fallback_errs, cfg.path or ""
        except Exception as fb_exc:  # noqa: BLE001
            logger.error("UniversalFallbackReader also failed: %s", fb_exc)
            self.learner.record(IngestionOutcome(
                dataset_id=cfg.dataset_id,
                source_type="file",
                success=False,
                error_type="DATA_FORMAT_ERROR",
                error_message=str(fb_exc),
                source_uri=cfg.path or "",
            ))
            # ── Last resort: return placeholder via DataRescue ───────────────
            df_placeholder, _ = rescue_dataframe(
                None, source_type="file", context={"path": cfg.path, "error": str(fb_exc)}
            )
            return df_placeholder, read_errors, cfg.path or ""

    def _read_api(self, cfg: SourceConfig, executor: SafeExecutor):
        from ingestion.readers.api_reader import APIReader, APISourceConfig
        from ingestion.data_rescue import rescue_dataframe
        reader = APIReader()
        api_cfg = cfg.api_config
        if isinstance(api_cfg, dict):
            api_cfg = APISourceConfig(**api_cfg)

        result, errors = executor.run(reader.read, api_cfg)
        url = getattr(api_cfg, "url", "")
        raw_df = result.data if result is not None else None
        raw_errors = (errors + result.errors) if result is not None else errors
        df_rescued, rescue_rpt = rescue_dataframe(
            raw_df, source_type="api", context={"url": url}
        )
        if rescue_rpt.was_rescued:
            from ingestion.issf import IngestionError as IE
            for w in rescue_rpt.warnings:
                raw_errors.append(IE(error_type="QUALITY_WARN", message=f"[DataRescue] {w}",
                                     severity="WARN", correlation_id="rescue"))
        return df_rescued, raw_errors, url

    def _read_db(self, cfg: SourceConfig, executor: SafeExecutor):
        from ingestion.readers.db_reader import DBReader, DBSourceConfig
        from ingestion.data_rescue import rescue_dataframe
        reader = DBReader()
        db_cfg = cfg.db_config
        if isinstance(db_cfg, dict):
            db_cfg = DBSourceConfig(**db_cfg)

        uri = f"{getattr(db_cfg, 'backend', 'db')}://{getattr(db_cfg, 'host', '')}/{getattr(db_cfg, 'database', '')}/{getattr(db_cfg, 'table_or_collection', '')}"
        result, errors = executor.run(reader.read, db_cfg)
        raw_df = result.data if result is not None else None
        raw_errors = (errors + result.errors) if result is not None else errors
        df_rescued, rescue_rpt = rescue_dataframe(
            raw_df, source_type="database", context={"uri": uri}
        )
        if rescue_rpt.was_rescued:
            from ingestion.issf import IngestionError as IE
            for w in rescue_rpt.warnings:
                raw_errors.append(IE(error_type="QUALITY_WARN", message=f"[DataRescue] {w}",
                                     severity="WARN", correlation_id="rescue"))
        return df_rescued, raw_errors, uri

    def _read_stream(self, cfg: SourceConfig, executor: SafeExecutor):
        from ingestion.readers.stream_reader import StreamReader, WindowConfig, KafkaSourceConfig
        from ingestion.data_rescue import rescue_dataframe
        reader = StreamReader()
        stream_cfg = cfg.stream_config
        window_cfg = cfg.window_config
        if isinstance(window_cfg, dict):
            window_cfg = WindowConfig(**window_cfg)
        if window_cfg is None:
            window_cfg = WindowConfig()

        topic = getattr(stream_cfg, "topic", "kafka")
        # Collect first N windows and concatenate into one DataFrame
        snapshots = []
        stream_errors: list = []
        try:
            for snap in reader.read_kafka(stream_cfg, window_cfg, max_windows=cfg.max_stream_windows):
                if snap is not None and snap.data is not None and not snap.data.empty:
                    snapshots.append(snap.data)
                if len(snapshots) >= cfg.max_stream_windows:
                    break
        except Exception as exc:  # noqa: BLE001
            from ingestion.error_handler import StreamLagError
            stream_errors.append(StreamLagError(f"Stream read error: {exc}"))
            logger.warning("[UniversalIntake] Stream read error (non-fatal): %s", exc)

        raw_df = pd.concat(snapshots, ignore_index=True) if snapshots else None
        df_rescued, rescue_rpt = rescue_dataframe(
            raw_df, source_type="stream", context={"topic": topic}
        )
        if rescue_rpt.was_rescued:
            from ingestion.issf import IngestionError as IE
            for w in rescue_rpt.warnings:
                stream_errors.append(IE(error_type="QUALITY_WARN", message=f"[DataRescue] {w}",
                                        severity="WARN", correlation_id="rescue"))
        return df_rescued, stream_errors, topic

    def _load_baseline(self, baseline_snapshot_id: Optional[str]) -> Optional[pd.DataFrame]:
        """Load a previous snapshot's data as baseline for PSI comparison.

        Checks ``_issf.parquet`` first (current naming convention), then falls
        back to ``_data.parquet`` so baselines saved by older versions of
        DIPEX are still found correctly.
        """
        if not baseline_snapshot_id:
            return None
        for suffix in ("_issf.parquet", "_data.parquet"):
            parquet_path = os.path.join(self.snapshot_dir, f"{baseline_snapshot_id}{suffix}")
            try:
                if os.path.exists(parquet_path):
                    return pd.read_parquet(parquet_path)
            except Exception:  # noqa: BLE001
                logger.debug("Could not load baseline parquet %s", parquet_path)
        return None

    def _failure_snapshot(
        self, cfg: SourceConfig, errors: List, correlation_id: str
    ) -> ISSFSnapshot:
        """Return a compliant but FAILED ISSF snapshot when reading fails."""
        # Teach learner about this failure
        err_type = errors[0].error_type if errors else "DATA_FORMAT_ERROR"
        err_msg  = errors[0].message    if errors else "Read returned no data"
        try:
            self.learner.record(IngestionOutcome(
                dataset_id=cfg.dataset_id,
                source_type=cfg.source_type,
                success=False,
                quality_score=0.0,
                validation_status="FAILED",
                error_type=err_type,
                error_message=str(err_msg),
                source_uri=cfg.path or "",
                strategy_used={"format": cfg.file_format or "unknown"},
            ))
        except Exception as _learn_exc:  # noqa: BLE001
            logger.warning(
                "Learner record failed in failure_snapshot (non-fatal): %s", _learn_exc
            )
        return ISSFSnapshot(
            dataset_id=cfg.dataset_id,
            snapshot_id=correlation_id,
            schema_version="0.0.0",
            data_mode=cfg.data_mode,
            source_type=cfg.source_type,
            source_uri=cfg.path or "",
            column_metadata=[],
            row_count=0,
            quality_score=0.0,
            validation_status="FAILED",
            error_logs=errors,
            data=None,
        )

    # ── Internal: Large Data Helpers ──────────────────────────────────────────

    def _memory_guard(self) -> None:
        """Pause ingestion if process RSS memory exceeds max_mem_rss_gb."""
        try:
            import psutil
            large_cfg = self.config.get("large_data", {})
            max_rss_gb = float(large_cfg.get("max_mem_rss_gb", 8.0))
            process = psutil.Process(os.getpid())
            rss_gb = process.memory_info().rss / (1024 ** 3)
            if rss_gb > max_rss_gb:
                logger.warning(
                    "Memory guard triggered (RSS: %.1f GB > %s GB). Pausing ingestion...",
                    rss_gb, max_rss_gb
                )
                time.sleep(2.0)
                # optionally call gc.collect() here if we want tighter management
                import gc
                gc.collect()
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Memory guard failed: %s", exc)

    def _union_parquet_files(self, paths: List[str]) -> pd.DataFrame:
        """Merge multiple Parquet files via DuckDB if enabled, else Pandas concat."""
        if not paths:
            return pd.DataFrame()
        large_cfg = self.config.get("large_data", {})
        use_duckdb = large_cfg.get("use_duckdb_merge", True)

        if use_duckdb:
            try:
                import duckdb
                path_list = ", ".join(f"'{p}'" for p in paths)
                sql = f"SELECT * FROM read_parquet([{path_list}])"
                return duckdb.query(sql).df()
            except ImportError:
                logger.debug("DuckDB not available for _union_parquet_files")
            except Exception as exc:
                logger.warning("DuckDB merge failed, falling back: %s", exc)

        # Fallback
        dfs = []
        for path in paths:
            try:
                if path.endswith(".parquet"):
                    dfs.append(pd.read_parquet(path))
                else:
                    dfs.append(pd.read_csv(path))
            except Exception as exc:
                logger.warning("Could not read chunk %s: %s", path, exc)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # ── Convenience factory ───────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "UniversalIntake":
        """Build UniversalIntake from config.yaml."""
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        intake_cfg = config.get("universal_intake", {})
        return cls(
            config=config,
            snapshot_dir=intake_cfg.get("snapshot_dir", "data/snapshots"),
            registry_dir=intake_cfg.get("schema_registry_path", "data/schema_registry"),
        )
