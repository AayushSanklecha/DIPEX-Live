"""
ingestion/batch_processor.py — ENHANCED
-----------------------------------------
Production-grade batch ingestion orchestrator.

Modes
-----
full_refresh    : Drop + reload entire dataset
incremental     : Only load rows newer than last watermark
partition_aware : Load specific partition(s) only
scheduled       : Run on a cron-like schedule

Features
--------
- Historical archive retention (snapshots older than N days compressed to .gz)
- CDC (Change Data Capture) position tracking per source
- Configurable parallelism
- Per-partition isolated ISSF snapshots
- Full PipelineBridge integration after each ingest
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.ingestion.batch_processor")

# ── Batch Config ──────────────────────────────────────────────────────────────

@dataclass
class BatchConfig:
    """Configuration for a batch ingestion job."""
    job_id: str                        = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mode: str                          = "full_refresh"  # full_refresh|incremental|partition|scheduled
    source_configs: List[Any]          = field(default_factory=list)  # List[SourceConfig]
    target_col: Optional[str]          = None
    skip_pipeline_stages: List[str]    = field(default_factory=list)
    partition_col: Optional[str]       = None
    partition_values: Optional[List]   = None
    watermark_store_path: str          = "data/watermarks.json"
    archive_after_days: int            = 30
    snapshot_dir: str                  = "data/snapshots"
    registry_dir: str                  = "data/schema_registry"
    max_parallel: int                  = 1              # Reserved: parallel execution
    cron_expr: Optional[str]           = None           # e.g. "0 2 * * *" (2am daily)
    notify_on_failure: bool            = True
    run_pipeline_bridge: bool          = True           # Run downstream stages after ingest


@dataclass
class BatchResult:
    job_id: str
    mode: str
    started_at: str
    completed_at: Optional[str] = None
    total_sources: int = 0
    succeeded: int = 0
    failed: int = 0
    total_rows_ingested: int = 0
    snapshots: List[str] = field(default_factory=list)  # snapshot_ids
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id, "mode": self.mode,
            "started_at": self.started_at, "completed_at": self.completed_at,
            "total_sources": self.total_sources, "succeeded": self.succeeded,
            "failed": self.failed, "total_rows_ingested": self.total_rows_ingested,
            "snapshots": self.snapshots, "errors": self.errors,
        }


# ── Watermark Store ───────────────────────────────────────────────────────────

class WatermarkStore:
    """
    Persist watermark (last sync position) per dataset_id.
    Supports any comparable value: datetime, integer, string.
    Also stores CDC binlog/WAL positions.
    """

    def __init__(self, path: str = "data/watermarks.json") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:  # noqa: BLE001
                self._data = {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def get_cdc_position(self, dataset_id: str) -> Optional[Dict]:
        """Get stored CDC binlog/WAL position."""
        return self._data.get(f"cdc_{dataset_id}")

    def set_cdc_position(self, dataset_id: str, position: Dict) -> None:
        """Store CDC position after a successful sync."""
        self._data[f"cdc_{dataset_id}"] = {**position,
                                             "updated_at": datetime.now(timezone.utc).isoformat()}
        self._save()


# ── Archive Manager ───────────────────────────────────────────────────────────

class ArchiveManager:
    """
    Manages historical archive retention for ISSF snapshots.
    Snapshots older than `archive_after_days` are gzip-compressed and moved
    to data/archive/ to free up active snapshot directory space.
    """

    def __init__(self, snapshot_dir: str = "data/snapshots",
                 archive_dir: str = "data/archive") -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def archive_old_snapshots(self, days: int = 30) -> int:
        """Compress and archive snapshots older than `days`. Returns count archived."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        archived = 0
        for meta_file in self.snapshot_dir.glob("*_issf.json"):
            try:
                mtime = datetime.fromtimestamp(meta_file.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    snap_id = meta_file.stem.replace("_issf", "")
                    self._archive_snapshot(snap_id)
                    archived += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Archive failed for %s: %s", meta_file.name, exc)
        if archived:
            logger.info("Archived %d old snapshots (>%d days)", archived, days)
        return archived

    def _archive_snapshot(self, snapshot_id: str) -> None:
        for suffix in ("_issf.json", "_data.parquet", "_data.csv"):
            src = self.snapshot_dir / f"{snapshot_id}{suffix}"
            if src.exists():
                dst = self.archive_dir / f"{snapshot_id}{suffix}.gz"
                with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                src.unlink()

    def restore_snapshot(self, snapshot_id: str) -> None:
        """Restore an archived snapshot back to the active snapshot directory."""
        for suffix in ("_issf.json", "_data.parquet", "_data.csv"):
            src = self.archive_dir / f"{snapshot_id}{suffix}.gz"
            if src.exists():
                dst = self.snapshot_dir / f"{snapshot_id}{suffix}"
                with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                logger.info("Restored %s from archive.", dst.name)


# ── CDC Tracker ───────────────────────────────────────────────────────────────

class CDCTracker:
    """
    Change Data Capture (CDC) awareness layer.

    Tracks the last-read position for each source so that subsequent
    incremental reads fetch only NEW or CHANGED rows.

    Supported CDC methods
    ---------------------
    watermark : Compare a timestamp/integer column against last known max value
    binlog    : MySQL binlog file + position (stored, not actively consumed)
    wal_lsn   : PostgreSQL LSN (stored, for future replication slot integration)
    offset    : Kafka consumer offset per partition
    """

    def __init__(self, watermark_store: WatermarkStore) -> None:
        self.store = watermark_store

    def get_incremental_filter(self, dataset_id: str, watermark_col: str) -> Optional[Any]:
        """Return last watermark value for incremental reads."""
        key = f"wm_{dataset_id}_{watermark_col}"
        return self.store.get(key)

    def update_watermark(self, dataset_id: str, watermark_col: str, new_value: Any) -> None:
        """Update watermark after a successful incremental read."""
        key = f"wm_{dataset_id}_{watermark_col}"
        self.store.set(key, new_value)
        logger.debug("Watermark updated: %s → %s", key, new_value)

    def record_mysql_binlog(self, dataset_id: str, log_file: str, log_pos: int) -> None:
        self.store.set_cdc_position(dataset_id, {"type": "mysql_binlog",
                                                     "log_file": log_file, "log_pos": log_pos})

    def record_pg_lsn(self, dataset_id: str, lsn: str) -> None:
        self.store.set_cdc_position(dataset_id, {"type": "pg_wal_lsn", "lsn": lsn})

    def record_kafka_offset(self, dataset_id: str, topic: str,
                             partition_offsets: Dict[int, int]) -> None:
        self.store.set_cdc_position(dataset_id, {"type": "kafka_offset",
                                                     "topic": topic, "offsets": partition_offsets})

    def get_cdc_position(self, dataset_id: str) -> Optional[Dict]:
        return self.store.get_cdc_position(dataset_id)


# ── Batch Processor ───────────────────────────────────────────────────────────

class BatchProcessor:
    """
    Orchestrates batch ingestion across multiple sources with full downstream
    pipeline integration.

    Usage::

        bp = BatchProcessor(config)
        batch_cfg = BatchConfig(
            mode="incremental",
            source_configs=[SourceConfig(source_type="file", dataset_id="sales", path="data.csv")],
            target_col="churn",
        )
        result = bp.run(batch_cfg)
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self._watermark_store: Optional[WatermarkStore] = None

    def _get_watermarks(self, path: str) -> WatermarkStore:
        if self._watermark_store is None:
            self._watermark_store = WatermarkStore(path)
        return self._watermark_store

    def run(self, batch_cfg: BatchConfig) -> BatchResult:
        t0 = time.perf_counter()
        result = BatchResult(
            job_id=batch_cfg.job_id,
            mode=batch_cfg.mode,
            started_at=datetime.now(timezone.utc).isoformat(),
            total_sources=len(batch_cfg.source_configs),
        )
        watermarks = self._get_watermarks(batch_cfg.watermark_store_path)
        cdc = CDCTracker(watermarks)
        archiver = ArchiveManager(snapshot_dir=batch_cfg.snapshot_dir)

        from ingestion.universal_intake import UniversalIntake
        intake = UniversalIntake(
            config=self.config,
            snapshot_dir=batch_cfg.snapshot_dir,
            registry_dir=batch_cfg.registry_dir,
        )

        from ingestion.pipeline_bridge import PipelineBridge
        bridge = PipelineBridge(config=self.config) if batch_cfg.run_pipeline_bridge else None

        for src_cfg in batch_cfg.source_configs:
            try:
                # Apply incremental watermark if available
                if batch_cfg.mode == "incremental":
                    wm_col = getattr(src_cfg, "db_config", None) and \
                             getattr(src_cfg.db_config, "watermark_column", "")
                    if wm_col:
                        last = cdc.get_incremental_filter(src_cfg.dataset_id, wm_col)
                        if hasattr(src_cfg.db_config, "watermark_last_value"):
                            src_cfg.db_config.watermark_last_value = last

                # Apply partition filter
                if batch_cfg.mode == "partition" and batch_cfg.partition_col:
                    src_cfg = self._apply_partition_filter(
                        src_cfg, batch_cfg.partition_col, batch_cfg.partition_values
                    )

                # Ingest
                snapshot = intake.ingest(src_cfg)
                result.snapshots.append(snapshot.snapshot_id)
                result.total_rows_ingested += snapshot.row_count
                result.succeeded += 1

                # Update watermark
                if batch_cfg.mode == "incremental" and snapshot.extra_meta:
                    new_wm = snapshot.extra_meta.get("watermark_new_value")
                    wm_col2 = getattr(getattr(src_cfg, "db_config", None),
                                      "watermark_column", None)
                    if new_wm is not None and wm_col2:
                        cdc.update_watermark(src_cfg.dataset_id, wm_col2, new_wm)

                # Run downstream pipeline
                if bridge and snapshot.data is not None:
                    pipe_result = bridge.run(
                        snapshot,
                        target_col=batch_cfg.target_col,
                        skip_stages=batch_cfg.skip_pipeline_stages,
                    )
                    logger.info(
                        "[%s] Pipeline gate: %s", batch_cfg.job_id, pipe_result.gate_decision
                    )

            except Exception as exc:  # noqa: BLE001
                result.failed += 1
                msg = f"{src_cfg.dataset_id}: {type(exc).__name__}: {exc}"
                result.errors.append(msg)
                logger.error("[Batch %s] Source failed: %s", batch_cfg.job_id, msg)

        # Archive old snapshots
        if batch_cfg.archive_after_days > 0:
            archiver.archive_old_snapshots(days=batch_cfg.archive_after_days)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        result.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "[Batch %s] Done — %d/%d sources succeeded, %d rows, %.0fms",
            batch_cfg.job_id, result.succeeded, result.total_sources,
            result.total_rows_ingested, elapsed,
        )
        self._write_job_log(result, batch_cfg)
        return result

    def run_scheduled(
        self,
        batch_cfg: BatchConfig,
        check_interval_s: float = 60.0,
        max_runs: int = 1_000_000,
    ) -> None:
        """
        Run batch ingestion on a schedule (blocking loop).
        Uses simple interval polling (production: replace with APScheduler/Airflow).
        """
        import fnmatch
        runs = 0
        interval = batch_cfg.extra.get("interval_s", check_interval_s) \
            if hasattr(batch_cfg, "extra") else check_interval_s
        logger.info("Scheduled batch started — interval=%.0fs job_id=%s",
                    interval, batch_cfg.job_id)
        while runs < max_runs:
            try:
                self.run(batch_cfg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Scheduled run failed: %s", exc)
            runs += 1
            logger.info("Next scheduled run in %.0fs (run %d/%d)", interval, runs, max_runs)
            time.sleep(interval)

    @staticmethod
    def _apply_partition_filter(src_cfg: Any, partition_col: str,
                                partition_values: Optional[List]) -> Any:
        """Inject partition filter into SQL/query for partition-aware reads."""
        import copy
        src = copy.deepcopy(src_cfg)
        if hasattr(src, "db_config") and src.db_config:
            if not src.db_config.query and src.db_config.table_or_collection and partition_values:
                vals = ", ".join(f"'{v}'" for v in partition_values)
                src.db_config.query = (
                    f'SELECT * FROM "{src.db_config.table_or_collection}" '
                    f'WHERE "{partition_col}" IN ({vals})'
                )
        return src

    @staticmethod
    def _write_job_log(result: BatchResult, cfg: BatchConfig) -> None:
        import json
        os.makedirs("audit", exist_ok=True)
        with open("audit/batch_jobs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
