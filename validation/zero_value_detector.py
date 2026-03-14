"""
validation/zero_value_detector.py
-----------------------------------
Hard Gate sub-validator 6 — Suspicious Zero-Value Detection.

A '0' is valid for many fields (binary flags, counters that start at zero).
But 0 is SUSPICIOUS when it appears in columns where zero is physically
implausible: revenue, price, age, account balance, quantity, etc.

Detection strategy:
  1. Identify numeric columns (non-binary, i.e. >2 unique values)
  2. Compute the fraction of values that are exactly 0.0
  3. Compare against configurable thresholds:
       zero_warn_threshold  : fraction above which → WARNING  (default 0.10)
       zero_error_threshold : fraction above which → ERROR    (default 0.50)
  4. Optionally enforce a mandatory-nonzero list (always CRITICAL on any zero)

Additionally:
  - Reports per-column zero statistics in the ZeroReport
  - Detects "all-zero" columns separately (→ immediately suspicious)
  - Skips binary columns (0/1 only) — zeros there are semantically valid

Config stanza (all optional)::

    validation:
      zero:
        zero_warn_threshold: 0.10
        zero_error_threshold: 0.50
        mandatory_nonzero: []          # e.g. [revenue, price, age]
        skip_binary_columns: true
        min_unique_for_check: 3        # skip columns with fewer unique values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.validation.zero_value_detector")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZeroViolation:
    column: str
    severity: str          # "WARNING" | "ERROR" | "CRITICAL"
    violation_type: str    # "suspicious_zeros" | "all_zeros" | "mandatory_zero"
    zero_count: int
    zero_pct: float
    total_count: int
    threshold: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "severity": self.severity,
            "type": "ZERO_VIOLATION",
            "violation_type": self.violation_type,
            "zero_count": self.zero_count,
            "zero_pct": round(self.zero_pct, 6),
            "total_count": self.total_count,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class ZeroReport:
    """Per-run summary of zero-value analysis."""
    run_id: str = ""
    columns_checked: int = 0
    columns_skipped: int = 0          # binary / low-unique columns
    violations: List[ZeroViolation] = field(default_factory=list)
    column_stats: Dict[str, Any] = field(default_factory=dict)  # {col: {zero_count, zero_pct}}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "columns_checked": self.columns_checked,
            "columns_skipped": self.columns_skipped,
            "num_violations": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "column_stats": self.column_stats,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class ZeroValueDetector:
    """
    Flags numeric columns with suspicious proportions of exactly-zero values.

    Integrates into Hard Gate 1 as sub-validator 6.
    Can also be used standalone: ``ZeroValueDetector(config).validate(df)``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("validation", {}).get("zero", {})
        self.warn_threshold: float      = float(cfg.get("zero_warn_threshold", 0.10))
        self.error_threshold: float     = float(cfg.get("zero_error_threshold", 0.50))
        self.mandatory_nonzero: List[str] = list(cfg.get("mandatory_nonzero", []))
        self.skip_binary: bool          = bool(cfg.get("skip_binary_columns", True))
        self.min_unique: int            = int(cfg.get("min_unique_for_check", 3))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ZeroValueDetector":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame, run_id: str = "") -> ZeroReport:
        """
        Analyse all numeric columns for suspicious zero concentrations.

        Returns a ZeroReport with per-column stats and ZeroViolation list.
        """
        report = ZeroReport(run_id=run_id)

        if df is None or df.empty:
            return report

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in num_cols:
            series = df[col].dropna()
            n_total = len(series)

            if n_total == 0:
                report.columns_skipped += 1
                continue

            n_unique = series.nunique()

            # Skip binary columns (0/1 indicators) — zeros are semantically OK
            if self.skip_binary and n_unique <= 2:
                report.columns_skipped += 1
                continue

            # Skip very low-cardinality columns (e.g. flag columns with 0/1/2)
            if n_unique < self.min_unique:
                report.columns_skipped += 1
                continue

            report.columns_checked += 1
            zero_count = int((series == 0.0).sum())
            zero_pct = zero_count / n_total

            # Record stats regardless of violation
            report.column_stats[col] = {
                "zero_count": zero_count,
                "zero_pct": round(zero_pct, 6),
                "total_nonzero": n_total - zero_count,
                "n_unique": int(n_unique),
            }

            # ── CRITICAL: column is in mandatory_nonzero list ─────────────────
            if col in self.mandatory_nonzero and zero_count > 0:
                v = ZeroViolation(
                    column=col,
                    severity="CRITICAL",
                    violation_type="mandatory_zero",
                    zero_count=zero_count,
                    zero_pct=zero_pct,
                    total_count=n_total,
                    threshold=0.0,
                    message=(
                        f"[CRITICAL] Column '{col}' is declared mandatory-nonzero "
                        f"but contains {zero_count} zero(s) ({zero_pct:.2%}). "
                        "This likely indicates a data collection bug or upstream error."
                    ),
                )
                report.violations.append(v)
                logger.error(v.message)
                continue

            # ── ERROR: all-zero column ────────────────────────────────────────
            if zero_pct == 1.0 and n_total >= 10:
                v = ZeroViolation(
                    column=col,
                    severity="ERROR",
                    violation_type="all_zeros",
                    zero_count=zero_count,
                    zero_pct=zero_pct,
                    total_count=n_total,
                    threshold=self.error_threshold,
                    message=(
                        f"Column '{col}' is ENTIRELY zero (100% of {n_total} non-null values). "
                        "This strongly indicates a missing upstream default, a reset column, "
                        "or a data pipeline failure."
                    ),
                )
                report.violations.append(v)
                logger.error(v.message)
                continue

            # ── ERROR: exceeds error threshold ────────────────────────────────
            if zero_pct > self.error_threshold:
                v = ZeroViolation(
                    column=col,
                    severity="ERROR",
                    violation_type="suspicious_zeros",
                    zero_count=zero_count,
                    zero_pct=zero_pct,
                    total_count=n_total,
                    threshold=self.error_threshold,
                    message=(
                        f"Column '{col}' has {zero_pct:.2%} zero values "
                        f"(threshold: {self.error_threshold:.2%}). "
                        "This is likely a data quality issue — check for default-zero fills, "
                        "silent failures, or imputation with 0 instead of NaN."
                    ),
                )
                report.violations.append(v)
                logger.error(v.message)

            # ── WARNING: exceeds warn threshold ───────────────────────────────
            elif zero_pct > self.warn_threshold:
                v = ZeroViolation(
                    column=col,
                    severity="WARNING",
                    violation_type="suspicious_zeros",
                    zero_count=zero_count,
                    zero_pct=zero_pct,
                    total_count=n_total,
                    threshold=self.warn_threshold,
                    message=(
                        f"Column '{col}' has {zero_pct:.2%} zero values "
                        f"(warn threshold: {self.warn_threshold:.2%}). "
                        "Review whether zeros are meaningful or represent missing data."
                    ),
                )
                report.violations.append(v)
                logger.warning(v.message)

        # Sort: CRITICAL first, then ERROR, then WARNING
        _sev_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2}
        report.violations.sort(key=lambda v: _sev_order.get(v.severity, 9))

        logger.info(
            "[ZeroValueDetector run_id=%s] checked=%d skipped=%d violations=%d",
            run_id[:8], report.columns_checked, report.columns_skipped, len(report.violations),
        )
        return report
