"""
verify_pipeline.py
------------------
Canonical entry point for single-run DIPEX pipeline execution.

ARCHITECTURE NOTE (v3.1+)
--------------------------
This module is now a **thin façade** over ``ingestion.pipeline_bridge.PipelineBridge``,
which is the single canonical 13-stage pipeline implementation.

All business logic (preprocessing, validation, profiling, proposal, governance,
statistical analysis, confidence vector, Hard Gate 2, modeling, RL update,
adaptive learning, reporting, experience memory) lives in PipelineBridge.

This module exists to:
1. Preserve backward compatibility for any ``verify_pipeline.orchestrate_pipeline()``
   callers (CLI ``python main.py run``, legacy tests, etc.)
2. Accept the file-based interface (``run_id`` → ``data/uploads/{run_id}_sample.csv``)
   expected by ``POST /api/run``
3. Translate that file-based interface into an ISSFSnapshot and hand off to PipelineBridge

Usage
-----
    from verify_pipeline import orchestrate_pipeline
    result = orchestrate_pipeline(run_id, config, target_col="label")

Or via CLI:
    python main.py run --run-id <uuid> --target label
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger("dipex.verify_pipeline")


# ── Public API (backward-compatible) ─────────────────────────────────────────

def orchestrate_pipeline(
    run_id: str,
    config: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    skip_stages: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Orchestrate the full 13-stage DIPEX pipeline for *run_id*.

    1. Locates ``data/uploads/{run_id}_sample.csv`` (uploaded via POST /api/ingest)
    2. Creates an ISSFSnapshot via UniversalIntake
    3. Delegates to PipelineBridge.run() — the single canonical pipeline implementation
    4. Returns PipelineResult.summary()

    Parameters
    ----------
    run_id : str
        UUID that identifies the upload (from POST /api/ingest).
    config : dict, optional
        Pipeline configuration dict. Loaded from config.yaml if not provided.
    target_col : str, optional
        Supervised ML target column. None → unsupervised mode.
    skip_stages : list, optional
        Stage names to skip, e.g. ['modeling', 'rl_update'].

    Returns
    -------
    dict
        PipelineResult.summary() — gate decisions, confidence vector,
        per-stage results, model metrics, report path.
    """
    if config is None:
        config = _load_config()

    # ── Locate upload ───────────────────────────────────────────────────────
    upload_dir = config.get("storage", {}).get("upload_dir", "data/uploads")
    csv_path = os.path.join(upload_dir, f"{run_id}_sample.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Upload not found: {csv_path}. "
            f"Did you POST the file to /api/ingest first?"
        )

    logger.info("[%s] verify_pipeline: loading %s", run_id[:8], csv_path)

    # ── Read file → DataFrame ───────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Uploaded file is empty: {csv_path}")

    # ── UDIL ingest → ISSFSnapshot ──────────────────────────────────────────
    try:
        from ingestion.universal_intake import UniversalIntake, SourceConfig
        intake = UniversalIntake(config=config)
        snapshot = intake.ingest(SourceConfig(
            source_type="file",
            dataset_id=run_id,
            path=csv_path
        ))
    except ImportError:
        # Graceful fallback: build a minimal snapshot from the DataFrame
        logger.warning("UniversalIntake unavailable — using raw DataFrame snapshot")
        snapshot = _make_minimal_snapshot(df, run_id)

    # ── Delegate to canonical PipelineBridge ────────────────────────────────
    from ingestion.pipeline_bridge import PipelineBridge

    bridge = PipelineBridge(config=config)
    result = bridge.run(
        snapshot=snapshot,
        target_col=target_col,
        run_id=run_id,
        skip_stages=skip_stages or [],
    )

    summary = result.summary()
    logger.info(
        "[%s] verify_pipeline: gate=%s confidence=%.3f",
        run_id[:8],
        summary.get("gate_decision", "?"),
        (summary.get("confidence_vector") or {}).get("confidence_score", 0.0),
    )
    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    if os.path.exists("config.yaml"):
        with open("config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _make_minimal_snapshot(df: pd.DataFrame, run_id: str):
    """
    Build the lightest possible snapshot object that PipelineBridge.run()
    accepts when UniversalIntake is not available.
    """
    try:
        from ingestion.issf import ISSFSnapshot
        snap = ISSFSnapshot(
            snapshot_id=run_id,
            dataset_id=run_id,
            data=df,
            source_type="file",
            schema_version="1.0",
        )
        return snap
    except ImportError:
        # Return a duck-typed object as absolute last resort
        class _MinimalSnap:
            snapshot_id = run_id
            dataset_id  = run_id
            data        = df
            source_type = "file"
            schema_version = "1.0"
            row_count   = len(df)
            col_count   = len(df.columns)
            quality_score = 1.0
            validation_status = "PASSED"
            error_logs  = []
            domain_metadata = {}
            bronze_ref  = None
        return _MinimalSnap()
