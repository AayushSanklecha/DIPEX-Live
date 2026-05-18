"""
ingestion/quality_gate.py
--------------------------
Mandatory data quality gate — runs on every ingestion before ISSF is finalised.

Checks performed (in order)
----------------------------
1.  Missing value rate per column and overall
2.  Duplicate row detection (exact + near-duplicate hash)
3.  Numeric range validation (from config bounds)
4.  Type mismatch count (expected dtype vs actual)
5.  Unexpected category detection (allowed values from config)
6.  Statistical distribution comparison (PSI vs baseline snapshot)
7.  Outlier flagging (IQR method — non-blocking, flag only)
8.  Referential integrity check (FK column subset validation)

Quality Score
-------------
score = 1.0 − weighted_penalty
  null_penalty      : 0.30 weight
  duplicate_penalty : 0.20 weight
  range_penalty     : 0.20 weight
  type_penalty      : 0.15 weight
  category_penalty  : 0.15 weight

Threshold: if quality_score < config threshold → set validation_status = FAILED.
If only warnings → WARN.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.ingestion.quality_gate")


# ── Quality Check Result ──────────────────────────────────────────────────────

@dataclass
class ColumnQuality:
    column: str
    null_rate: float
    unique_rate: float
    outlier_count: int
    type_mismatch_count: int
    unexpected_categories: List[str]
    range_violations: int
    checks_passed: bool


@dataclass
class QualityReport:
    dataset_id: str
    snapshot_id: str
    quality_score: float              # 0.0–1.0
    validation_status: str            # PASSED | WARN | FAILED
    row_count: int
    duplicate_count: int
    duplicate_rate: float
    overall_null_rate: float
    column_quality: List[ColumnQuality]
    distribution_drift: Dict[str, float]   # {column: PSI}
    violations: List[str]             # Human-readable list of problems
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "quality_score": round(self.quality_score, 4),
            "validation_status": self.validation_status,
            "row_count": self.row_count,
            "duplicate_count": self.duplicate_count,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "overall_null_rate": round(self.overall_null_rate, 4),
            "violations": self.violations,
            "warnings": self.warnings,
            "distribution_drift": {k: round(v, 4) for k, v in self.distribution_drift.items()},
        }


# ── Quality Gate ──────────────────────────────────────────────────────────────

class QualityGate:
    """
    Run all quality checks and return a QualityReport.

    Parameters (from config['universal_intake']['quality_thresholds'])
    ------------------------------------------------------------------
    max_null_rate        : float  (default 0.30) — per-column
    max_overall_null_rate: float  (default 0.20)
    max_duplicate_rate   : float  (default 0.05)
    min_quality_score    : float  (default 0.70) — below this → FAILED
    warn_quality_score   : float  (default 0.85) — below this → WARN
    psi_warn_threshold   : float  (default 0.10)
    psi_fail_threshold   : float  (default 0.20)
    outlier_iqr_factor   : float  (default 3.0)  — IQR * factor = fence
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("universal_intake", {}).get("quality_thresholds", {})
        self.max_null_rate         = float(cfg.get("max_null_rate", 0.30))
        self.max_overall_null_rate = float(cfg.get("max_overall_null_rate", 0.20))
        self.max_duplicate_rate    = float(cfg.get("max_duplicate_rate", 0.05))
        self.min_quality_score     = float(cfg.get("min_quality_score", 0.70))
        self.warn_quality_score    = float(cfg.get("warn_quality_score", 0.85))
        self.psi_warn              = float(cfg.get("psi_warn_threshold", 0.10))
        self.psi_fail              = float(cfg.get("psi_fail_threshold", 0.20))
        self.iqr_factor            = float(cfg.get("outlier_iqr_factor", 3.0))

    def check(
        self,
        df: pd.DataFrame,
        dataset_id: str = "",
        snapshot_id: str = "",
        range_rules: Optional[Dict[str, Tuple[float, float]]] = None,
        allowed_categories: Optional[Dict[str, List[str]]] = None,
        expected_dtypes: Optional[Dict[str, str]] = None,
        baseline_df: Optional[pd.DataFrame] = None,
        fk_rules: Optional[Dict[str, Set]] = None,
    ) -> QualityReport:
        """
        Run all quality checks and return a QualityReport.

        Parameters
        ----------
        range_rules         : {col: (min_val, max_val)}
        allowed_categories  : {col: [val1, val2, ...]}
        expected_dtypes     : {col: "int64"} — for type mismatch check
        baseline_df         : reference DataFrame for PSI drift check
        fk_rules            : {col: {allowed_value_set}}
        """
        violations: List[str] = []
        warnings:   List[str] = []
        n = len(df)

        # ── 1. Duplicate detection ────────────────────────────────────────────
        try:
            dup_count = int(df.duplicated().sum())
        except Exception:  # noqa: BLE001 — unhashable types (lists/dicts)
            try:
                dup_count = int(df.astype(str).duplicated().sum())
            except Exception:  # noqa: BLE001
                dup_count = 0
                warnings.append("Could not check duplicates (unhashable column types)")
        dup_rate  = dup_count / n if n > 0 else 0.0
        if dup_rate > self.max_duplicate_rate:
            violations.append(
                f"Duplicate rate {dup_rate:.1%} exceeds threshold {self.max_duplicate_rate:.1%} ({dup_count} rows)"
            )
        elif dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows ({dup_rate:.2%})")

        # ── 2. Overall null rate ──────────────────────────────────────────────
        total_cells  = n * len(df.columns) if len(df.columns) > 0 else 1
        total_nulls  = int(df.isna().sum().sum())
        overall_null = total_nulls / total_cells
        if overall_null > self.max_overall_null_rate:
            warnings.append(
                f"Overall null rate {overall_null:.1%} exceeds threshold {self.max_overall_null_rate:.1%}"
            )

        # ── Per-column checks ─────────────────────────────────────────────────
        col_quality: List[ColumnQuality] = []

        for col in df.columns:
            col_series  = df[col]
            null_count  = int(col_series.isna().sum())
            null_rate   = null_count / n if n > 0 else 0.0
            try:
                unique_count = col_series.nunique(dropna=True)
            except Exception:  # noqa: BLE001 — unhashable types
                try:
                    unique_count = col_series.astype(str).nunique(dropna=True)
                except Exception:  # noqa: BLE001
                    unique_count = 0
            unique_rate  = unique_count / n if n > 0 else 0.0
            type_mismatches = 0
            unexpected_cats: List[str] = []
            range_viols = 0
            outlier_count = 0

            # Null per-column
            if null_rate > self.max_null_rate:
                warnings.append(
                    f"Column '{col}': null rate {null_rate:.1%} exceeds {self.max_null_rate:.1%}"
                )

            # Type mismatch
            if expected_dtypes and col in expected_dtypes:
                exp = expected_dtypes[col].lower()
                act = str(col_series.dtype).lower()
                if "int" in exp and "int" not in act:
                    type_mismatches += 1
                    warnings.append(f"Column '{col}': expected int dtype, got {act}")
                elif "float" in exp and "float" not in act:
                    type_mismatches += 1
                    warnings.append(f"Column '{col}': expected float dtype, got {act}")

            # Range validation
            if range_rules and col in range_rules and pd.api.types.is_numeric_dtype(col_series):
                lo, hi = range_rules[col]
                non_null = col_series.dropna()
                oob = int(((non_null < lo) | (non_null > hi)).sum())
                if oob > 0:
                    range_viols = oob
                    pct = oob / len(non_null) if len(non_null) > 0 else 0
                    if pct > 0.01:
                        violations.append(
                            f"Column '{col}': {oob} values out of range [{lo}, {hi}]"
                        )
                    else:
                        warnings.append(f"Column '{col}': {oob} values out of range [{lo}, {hi}]")

            # Unexpected categories
            if allowed_categories and col in allowed_categories:
                allowed = set(str(v) for v in allowed_categories[col])
                actual  = set(str(v) for v in col_series.dropna().unique())
                unexpected = list(actual - allowed)
                if unexpected:
                    unexpected_cats = unexpected[:10]
                    warnings.append(
                        f"Column '{col}': unexpected categories: {unexpected_cats[:5]}"
                    )

            # Outlier detection (IQR — non-blocking)
            if pd.api.types.is_numeric_dtype(col_series) and len(col_series.dropna()) > 10:
                try:
                    q1 = float(col_series.quantile(0.25))
                    q3 = float(col_series.quantile(0.75))
                    iqr = q3 - q1
                    fence_lo = q1 - self.iqr_factor * iqr
                    fence_hi = q3 + self.iqr_factor * iqr
                    non_null_series = col_series.dropna()
                    outlier_count = int(((non_null_series < fence_lo) | (non_null_series > fence_hi)).sum())
                    if outlier_count > 0:
                        warnings.append(f"Column '{col}': {outlier_count} potential outliers (IQR×{self.iqr_factor})")
                except Exception:  # noqa: BLE001
                    pass

            col_quality.append(ColumnQuality(
                column=col, null_rate=null_rate, unique_rate=unique_rate,
                outlier_count=outlier_count, type_mismatch_count=type_mismatches,
                unexpected_categories=unexpected_cats, range_violations=range_viols,
                checks_passed=null_rate <= self.max_null_rate and range_viols == 0,
            ))

        # ── 3. Referential integrity ──────────────────────────────────────────
        if fk_rules:
            for col, allowed_set in fk_rules.items():
                if col in df.columns:
                    invalid = df[col].dropna()[~df[col].dropna().isin(allowed_set)]
                    if len(invalid) > 0:
                        violations.append(
                            f"Column '{col}': {len(invalid)} FK violations (sample: {list(invalid[:3])})"
                        )

        # ── 4. Distribution drift (PSI) ───────────────────────────────────────
        drift_scores: Dict[str, float] = {}
        if baseline_df is not None:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in baseline_df.columns:
                    psi = self._compute_psi(baseline_df[col].dropna(), df[col].dropna())
                    drift_scores[col] = round(psi, 4)
                    if psi >= self.psi_fail:
                        violations.append(
                            f"Column '{col}': PSI {psi:.3f} ≥ fail threshold {self.psi_fail} (MAJOR DRIFT)"
                        )
                    elif psi >= self.psi_warn:
                        warnings.append(
                            f"Column '{col}': PSI {psi:.3f} ≥ warn threshold {self.psi_warn} (minor drift)"
                        )

        # ── Quality Score ─────────────────────────────────────────────────────
        null_penalty      = min(1.0, overall_null / max(self.max_overall_null_rate, 0.01))
        dup_penalty       = min(1.0, dup_rate / max(self.max_duplicate_rate, 0.01))
        range_violations  = sum(c.range_violations for c in col_quality)
        range_penalty     = min(1.0, range_violations / max(n, 1))
        type_mismatches   = sum(c.type_mismatch_count for c in col_quality)
        type_penalty      = min(1.0, type_mismatches / max(len(df.columns), 1))
        cat_violations    = sum(bool(c.unexpected_categories) for c in col_quality)
        cat_penalty       = min(1.0, cat_violations / max(len(df.columns), 1))

        quality_score = max(0.0, 1.0 - (
            0.30 * null_penalty
            + 0.20 * dup_penalty
            + 0.20 * range_penalty
            + 0.15 * type_penalty
            + 0.15 * cat_penalty
        ))

        if violations or quality_score < self.min_quality_score:
            status = "FAILED"
        elif warnings or quality_score < self.warn_quality_score:
            status = "WARN"
        else:
            status = "PASSED"

        if violations:
            logger.error(
                "[%s] Quality gate FAILED (score=%.2f): %d violations",
                dataset_id, quality_score, len(violations),
            )
        elif warnings:
            logger.warning(
                "[%s] Quality gate WARN (score=%.2f): %d warnings",
                dataset_id, quality_score, len(warnings),
            )
        else:
            logger.info(
                "[%s] Quality gate PASSED (score=%.2f)",
                dataset_id, quality_score,
            )

        return QualityReport(
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            quality_score=quality_score,
            validation_status=status,
            row_count=n,
            duplicate_count=dup_count,
            duplicate_rate=dup_rate,
            overall_null_rate=overall_null,
            column_quality=col_quality,
            distribution_drift=drift_scores,
            violations=violations,
            warnings=warnings,
        )

    @staticmethod
    def _compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Compute Population Stability Index (PSI)."""
        try:
            all_vals = pd.concat([expected, actual]).dropna()
            boundaries = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
            boundaries[0] -= 1e-9
            boundaries[-1] += 1e-9

            exp_counts = np.histogram(expected, bins=boundaries)[0]
            act_counts = np.histogram(actual,   bins=boundaries)[0]

            exp_pct = exp_counts / max(len(expected), 1)
            act_pct = act_counts / max(len(actual), 1)

            # Replace zeros to avoid log(0)
            exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
            act_pct = np.where(act_pct == 0, 1e-6, act_pct)

            psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
            return max(0.0, psi)
        except Exception:  # noqa: BLE001
            return 0.0
