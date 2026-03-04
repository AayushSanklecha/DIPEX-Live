"""
profiling/profile_report.py
-----------------------------
Step 3 — Data Profiling Engine: Aggregated Profile Report.

``ProfileReport`` is the single entry point for Step 3.  It orchestrates
all sub-profilers, aggregates their outputs, builds a unified ``analyst_flags``
list sorted by severity, and persists the result as a JSON file.

The JSON schema is fully self-describing and can be consumed by a BI dashboard,
a downstream audit system, or a retrieval-augmented generation (RAG) pipeline.

Usage::

    report = ProfileReport(config)
    result = report.generate(df, run_id="abc123")
    # → saves reports/{run_id}_profile.json
    # → returns the profile dict
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .profiler import Profiler
from .correlation_engine import CorrelationEngine
from .missingness_analyzer import MissingnessAnalyzer
from .drift_detector import DriftDetector

logger = logging.getLogger(__name__)

# Flag severity ordering for sort stability (highest severity first)
_FLAG_PRIORITY: Dict[str, int] = {
    "NEAR_DUPLICATE_COLUMNS":      0,
    "HIGH_CARDINALITY":            1,
    "STRONG_CORRELATION":          2,
    "STRONG_CATEGORICAL_ASSOCIATION": 3,
    "DRIFT_DETECTED":              4,
    "MNAR_SUSPECTED":              5,
    "MAR_PATTERN_DETECTED":        6,
    "HIGH_OVERALL_MISSINGNESS":    7,
    "CORRELATED_NULLS":            8,
    "HIGH_OUTLIER_RATE_IQR":       9,
    "HIGH_SKEW":                   10,
    "HEAVY_TAILS":                 11,
    "NON_NORMAL_DISTRIBUTION":     12,
    "HIGH_CARDINALITY_COLUMN":     13,
}


class ProfileReport:
    """
    Orchestrates all Step 3 sub-profilers and produces a unified report.

    Args:
        config:      Project config dict.
        report_dir:  Directory where JSON reports are persisted.
                     Falls back to ``config.storage.report_dir``, then ``"reports"``.
        baseline_df: Optional DataFrame holding baseline data for drift computation.
                     If ``None``, drift analysis section is omitted.
    """

    def __init__(
        self,
        config:      Optional[Dict[str, Any]] = None,
        baseline_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self._config      = config or {}
        self._baseline_df = baseline_df

        report_dir_cfg = self._config.get("storage", {}).get("report_dir", "reports")
        self._report_dir = report_dir_cfg

        self._profiler   = Profiler(self._config)
        self._corr_eng   = CorrelationEngine(self._config)
        self._miss_anal  = MissingnessAnalyzer(self._config)
        self._drift_det  = DriftDetector(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, df: pd.DataFrame, run_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Generates and persists the full data profile.

        Args:
            df:     The ingested DataFrame to profile.
            run_id: Pipeline run identifier — used for file naming and logging.

        Returns:
            Full profile dict.
        """
        logger.info(
            "ProfileReport: starting profiling for run_id=%s  shape=%s",
            run_id, df.shape,
        )

        # [RL] RLProfilingStrategy
        try:
            from profiling.rl_profiling_strategy import get_rl_profiling_strategy
            rl_strat = get_rl_profiling_strategy()
            ds_id = df.attrs.get("dataset_id", "unknown")
            skip_deep = rl_strat.should_skip_deep_profiling(dataset_id=ds_id, current_rows=len(df), current_cols=len(df.columns))
            logger.info("[RL] Profile Strategy decision: skip_deep_profiling=%s", skip_deep)
        except Exception:
            skip_deep = False

        generated_at = datetime.now(timezone.utc).isoformat()

        # ── Sub-profile execution ──────────────────────────────────────
        col_profile  = self._profiler.profile(df)

        if skip_deep:
            correlation, missingness, drift = {}, {}, {}
            logger.info("ProfileReport: Deep profiling skipped by RL policy.")
        else:
            correlation  = self._corr_eng.compute(df)
            missingness  = self._miss_anal.analyze(df)

            drift: Dict[str, Any] = {}
            if self._baseline_df is not None:
                drift = self._drift_det.detect(self._baseline_df, df)
            else:
                logger.debug("ProfileReport: no baseline_df provided — drift section skipped.")

        # ── Merge and prioritise analyst flags ─────────────────────────
        all_flags: List[Dict[str, Any]] = []
        all_flags.extend(col_profile.get("analyst_flags", []))
        all_flags.extend(correlation.get("highlights", []))
        all_flags.extend(missingness.get("analyst_flags", []))
        all_flags.extend(drift.get("analyst_flags", []))

        all_flags.sort(
            key=lambda f: _FLAG_PRIORITY.get(f.get("flag", ""), 99)
        )

        # ── Assemble final report ──────────────────────────────────────
        report: Dict[str, Any] = {
            "run_id":       run_id,
            "generated_at": generated_at,
            "dataset_shape": {
                "rows":    col_profile.get("row_count", 0),
                "columns": col_profile.get("column_count", 0),
            },
            "columns":       col_profile.get("columns", {}),
            "correlation":   correlation,
            "missingness":   missingness,
            "drift":         drift,
            "analyst_flags": all_flags,
            "flag_count":    len(all_flags),
        }

        # ── Persist to disk ────────────────────────────────────────────
        self._save(report, run_id)

        logger.info(
            "ProfileReport: complete — %d column(s), %d flag(s) raised.",
            col_profile.get("column_count", 0), len(all_flags),
        )

        self._log_summary(all_flags)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, report: Dict[str, Any], run_id: str) -> None:
        """Persists the report as a pretty-printed JSON file."""
        os.makedirs(self._report_dir, exist_ok=True)
        path = os.path.join(self._report_dir, f"{run_id}_profile.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str, ensure_ascii=False)
            logger.info("ProfileReport: saved to %s", path)
        except OSError as exc:
            logger.error("ProfileReport: failed to save report to %s: %s", path, exc)

    @staticmethod
    def _log_summary(flags: List[Dict[str, Any]]) -> None:
        """Emits a concise analyst-flag summary to the log."""
        if not flags:
            logger.info("ProfileReport: No analyst flags raised — data looks healthy.")
            return

        logger.info("  ┌─ Analyst Flags (Step 3) ──────────────")
        for flag in flags[:10]:  # show top 10
            col = flag.get("column", "")
            tag = flag.get("flag", "")
            detail = flag.get("detail", "")[:80]
            logger.info("  │ [%s] %s: %s …", tag, col, detail)
        if len(flags) > 10:
            logger.info("  │  … and %d more flag(s). See report JSON.", len(flags) - 10)
        logger.info("  └────────────────────────────────────────")
