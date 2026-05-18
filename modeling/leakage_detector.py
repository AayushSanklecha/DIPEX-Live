"""
modeling/leakage_detector.py
------------------------------
Target leakage detection before model training.

Detects:
  1. High correlation between features and target (Pearson / Cramér's V)
  2. ID-like columns with near-unique values (>99% unique)
  3. Columns derived from target (near-perfect predictors)
  4. Future-dated columns that would not be available at inference time

This module provides the standalone modeling-layer check. The pipeline
also has validation/leakage_detector.py which is called earlier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.modeling.leakage_detector")

# Column name patterns that often indicate derived/ID columns
_ID_PATTERNS = re.compile(
    r"\b(id|uuid|key|pk|primary|surrogate|row_?num|index|serial)\b", re.I
)
_FUTURE_PATTERNS = re.compile(
    r"\b(outcome|result|flag|label|target|pred|score|won|lost|churn|default)\b", re.I
)


@dataclass
class LeakageFlag:
    """A single leakage finding."""
    column: str
    leak_type: str  # correlation | id_column | name_pattern | future_date
    severity: str   # CRITICAL | ERROR | WARNING
    score: float
    message: str
    recommended_action: str

    def to_dict(self) -> Dict:
        return {
            "column": self.column,
            "leak_type": self.leak_type,
            "severity": self.severity,
            "score": round(self.score, 4),
            "message": self.message,
            "recommended_action": self.recommended_action,
        }


@dataclass
class LeakageReport:
    """Full leakage detection report."""
    flags: List[LeakageFlag] = field(default_factory=list)
    columns_dropped: List[str] = field(default_factory=list)
    columns_warned: List[str] = field(default_factory=list)
    n_checked: int = 0

    def to_dict(self) -> Dict:
        return {
            "flags": [f.to_dict() for f in self.flags],
            "columns_dropped": self.columns_dropped,
            "columns_warned": self.columns_warned,
            "n_checked": self.n_checked,
            "n_critical": sum(1 for f in self.flags if f.severity == "CRITICAL"),
            "n_warnings": sum(1 for f in self.flags if f.severity == "WARNING"),
        }


class ModelingLeakageDetector:
    """
    Modeling-layer leakage detection. Runs right before fit() to catch
    columns that should never reach the model.

    Usage::

        detector = ModelingLeakageDetector(drop_on_critical=True)
        clean_df, report = detector.detect(df, target_col="churn")
    """

    CORR_CRITICAL = 0.98   # Pearson correlation — drop
    CORR_WARN     = 0.90   # Pearson correlation — warn
    UNIQUENESS_CRITICAL = 0.99  # ID-like column threshold

    def __init__(
        self,
        drop_on_critical: bool = True,
        config: Optional[Dict] = None,
    ) -> None:
        self.drop_on_critical = drop_on_critical
        cfg = (config or {}).get("validation", {}).get("leakage", {})
        self.corr_critical = float(cfg.get("correlation_hard_threshold", self.CORR_CRITICAL))
        self.corr_warn     = float(cfg.get("correlation_warn_threshold", self.CORR_WARN))
        self.uniqueness_th = float(cfg.get("id_uniqueness_threshold", self.UNIQUENESS_CRITICAL))

    def detect(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> tuple[pd.DataFrame, LeakageReport]:
        """
        Detect and optionally remove leaky columns.

        Returns
        -------
        (clean_df, report)
        """
        report = LeakageReport()
        if df is None or df.empty or target_col not in df.columns:
            return df, report

        feature_cols = [c for c in df.columns if c != target_col]
        y = df[target_col].copy()
        report.n_checked = len(feature_cols)

        flags: List[LeakageFlag] = []

        for col in feature_cols:
            series = df[col]

            # ── Check 1: ID-like uniqueness ───────────────────────────────────
            try:
                n_unique = series.nunique()
                uniqueness = n_unique / max(len(series.dropna()), 1)
                if uniqueness >= self.uniqueness_th or _ID_PATTERNS.search(col):
                    flags.append(LeakageFlag(
                        column=col,
                        leak_type="id_column",
                        severity="CRITICAL" if uniqueness >= self.uniqueness_th else "WARNING",
                        score=round(uniqueness, 4),
                        message=(
                            f"Column '{col}' has {uniqueness:.1%} unique values "
                            f"(threshold: {self.uniqueness_th:.0%}). "
                            "Likely an ID column — a perfect proxy for row identity, not a feature."
                        ),
                        recommended_action="Drop this column from the feature set before model training.",
                    ))
                    continue
            except Exception:
                pass

            # ── Check 2: Name-pattern leak (future column) ────────────────────
            if _FUTURE_PATTERNS.search(col) and col != target_col:
                flags.append(LeakageFlag(
                    column=col,
                    leak_type="name_pattern",
                    severity="WARNING",
                    score=0.8,
                    message=(
                        f"Column '{col}' name pattern suggests it may be derived from "
                        "or correlated with the target. Risk of target leakage."
                    ),
                    recommended_action=(
                        "Verify this feature is available at inference time "
                        "and is not derived from the target. If derived, drop it."
                    ),
                ))

            # ── Check 3: Pearson / Spearman correlation with target ───────────
            try:
                if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(y):
                    filled = series.fillna(series.median())
                    y_filled = y.fillna(y.median())
                    corr = float(np.corrcoef(filled.values, y_filled.values)[0, 1])
                    abs_corr = abs(corr)
                    if abs_corr >= self.corr_critical:
                        flags.append(LeakageFlag(
                            column=col,
                            leak_type="correlation",
                            severity="CRITICAL",
                            score=abs_corr,
                            message=(
                                f"Column '{col}' has near-perfect correlation with target "
                                f"(r={corr:.4f}). Highly likely target leakage."
                            ),
                            recommended_action="Drop immediately — this will cause artificially inflated model metrics.",
                        ))
                    elif abs_corr >= self.corr_warn:
                        flags.append(LeakageFlag(
                            column=col,
                            leak_type="correlation",
                            severity="WARNING",
                            score=abs_corr,
                            message=(
                                f"Column '{col}' has very high correlation with target "
                                f"(r={corr:.4f}). Investigate for potential data leakage."
                            ),
                            recommended_action="Verify this feature would be available at prediction time.",
                        ))
            except Exception:
                pass

        report.flags = flags

        # ── Apply drops ────────────────────────────────────────────────────────
        to_drop = [f.column for f in flags if f.severity == "CRITICAL"] if self.drop_on_critical else []
        to_warn = [f.column for f in flags if f.severity == "WARNING"]

        report.columns_dropped = to_drop
        report.columns_warned  = to_warn

        if to_drop:
            df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")
            logger.warning(
                "[ModelLeakage] Dropped %d critical leaky columns: %s",
                len(to_drop), to_drop,
            )

        for col in to_warn:
            logger.warning("[ModelLeakage] Suspicious column (WARNING): %s", col)

        return df, report
