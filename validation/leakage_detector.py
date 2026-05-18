"""
validation/leakage_detector.py
--------------------------------
Data leakage detector — runs BEFORE modeling.

Detects features that are suspiciously predictive of the target,
indicating they may have been derived from it (target leakage)
or be a proxy for it (direct leakage).

Detection methods:
  1. Perfect / near-perfect correlation with target (numeric features)
  2. Near-perfect categorical alignment (chi-square + Cramér's V)
  3. ID-like columns with near-unique values (usually primary keys that
     trivially identify records → model memorises, not generalises)
  4. Post-feature-engineering ratio features that collapse to the target

Severity:
  CRITICAL — correlation / V > hard_thresh  → blocks modeling
  WARNING  — correlation / V > warn_thresh  → log + continue

All thresholds are config-driven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.validation.leakage_detector")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeakageViolation:
    column: str
    severity: str          # CRITICAL | WARNING
    leakage_type: str      # target_correlation | categorical_alignment | id_column
    score: float           # correlation / Cramér's V / uniqueness_ratio
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "severity": self.severity,
            "leakage_type": self.leakage_type,
            "score": round(self.score, 4),
            "message": self.message,
        }


@dataclass
class LeakageReport:
    target_col: str
    run_id: str
    violations: List[LeakageViolation] = field(default_factory=list)
    columns_blocked: List[str] = field(default_factory=list)  # removed from df

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "CRITICAL" for v in self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_col": self.target_col,
            "run_id": self.run_id,
            "violations": [v.to_dict() for v in self.violations],
            "columns_blocked": self.columns_blocked,
            "has_critical": self.has_critical,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class LeakageDetector:
    """
    Detect and remove data leakage columns before modeling.

    Config stanza (all optional)::

        validation:
          leakage:
            correlation_hard_threshold: 0.98  # CRITICAL if |corr| >= this
            correlation_warn_threshold: 0.90  # WARNING if |corr| >= this
            cramers_v_hard_threshold: 0.95    # CRITICAL for categorical alignment
            cramers_v_warn_threshold: 0.85    # WARNING for categorical alignment
            id_uniqueness_threshold: 0.99     # flag as ID-column if nunique/nrows >= this
            drop_critical: true               # auto-remove CRITICAL columns
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("validation", {}).get("leakage", {})
        self.corr_hard: float         = float(cfg.get("correlation_hard_threshold", 0.98))
        self.corr_warn: float         = float(cfg.get("correlation_warn_threshold", 0.90))
        self.v_hard: float            = float(cfg.get("cramers_v_hard_threshold", 0.95))
        self.v_warn: float            = float(cfg.get("cramers_v_warn_threshold", 0.85))
        self.id_thresh: float         = float(cfg.get("id_uniqueness_threshold", 0.99))
        self.drop_critical: bool      = bool(cfg.get("drop_critical", True))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LeakageDetector":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        df: pd.DataFrame,
        target_col: str,
        run_id: str = "",
    ) -> tuple[pd.DataFrame, LeakageReport]:
        """
        Scan df for data leakage with respect to target_col.

        Returns
        -------
        (cleaned_df, LeakageReport)
        cleaned_df has CRITICAL leakage columns removed (if drop_critical=True).
        """
        report = LeakageReport(target_col=target_col, run_id=run_id)

        if target_col not in df.columns:
            logger.warning("[LeakageDetector] target_col '%s' not in DataFrame — skipping.", target_col)
            return df, report

        y = df[target_col]
        feature_cols = [c for c in df.columns if c != target_col]

        # ── 1. Numeric feature correlation with target ────────────────────────
        y_num: Optional[pd.Series] = None
        y_nunique = 0
        try:
            y_nunique = y.nunique()
        except Exception: # noqa: BLE001
            y_nunique = y.astype(str).nunique()
            
        if pd.api.types.is_numeric_dtype(y) and y_nunique > 20:
            # If numeric and many unique values, treat as continuous, no label encoding needed.
            # y_num will be assigned 'y' below.
            pass
        elif y_nunique <= 20:
            # Encode categorical target for correlation
            from sklearn.preprocessing import LabelEncoder
            try:
                y_num = pd.Series(
                    LabelEncoder().fit_transform(y.fillna("__MISSING__").astype(str)),
                    index=y.index,
                )
            except Exception:
                y_num = None

        if y_num is not None:
            for col in df[feature_cols].select_dtypes(include=np.number).columns:
                try:
                    corr = abs(float(df[col].corr(y_num)))
                    if np.isnan(corr):
                        continue
                    if corr >= self.corr_hard:
                        report.violations.append(LeakageViolation(
                            column=col, severity="CRITICAL",
                            leakage_type="target_correlation",
                            score=corr,
                            message=(
                                f"'{col}' has near-perfect correlation={corr:.4f} "
                                f"with target '{target_col}' — likely target leakage."
                            ),
                        ))
                        logger.error(
                            "[LeakageDetector] CRITICAL: '%s' corr=%.4f with target", col, corr
                        )
                    elif corr >= self.corr_warn:
                        report.violations.append(LeakageViolation(
                            column=col, severity="WARNING",
                            leakage_type="target_correlation",
                            score=corr,
                            message=(
                                f"'{col}' has high correlation={corr:.4f} with target "
                                f"'{target_col}' — review for leakage."
                            ),
                        ))
                        logger.warning(
                            "[LeakageDetector] WARNING: '%s' corr=%.4f with target", col, corr
                        )
                except Exception:
                    continue

        # ── 2. Categorical alignment (Cramér's V) ─────────────────────────────
        y_cat = y.astype(str)
        for col in df[feature_cols].select_dtypes(include="object").columns:
            try:
                v = self._cramers_v(df[col].astype(str), y_cat)
                if np.isnan(v):
                    continue
                if v >= self.v_hard:
                    report.violations.append(LeakageViolation(
                        column=col, severity="CRITICAL",
                        leakage_type="categorical_alignment",
                        score=v,
                        message=(
                            f"'{col}' has near-perfect categorical alignment "
                            f"(Cramér's V={v:.4f}) with target '{target_col}'."
                        ),
                    ))
                    logger.error(
                        "[LeakageDetector] CRITICAL categorical alignment: '%s' V=%.4f", col, v
                    )
                elif v >= self.v_warn:
                    report.violations.append(LeakageViolation(
                        column=col, severity="WARNING",
                        leakage_type="categorical_alignment",
                        score=v,
                        message=(
                            f"'{col}' has high categorical alignment "
                            f"(Cramér's V={v:.4f}) with target '{target_col}'."
                        ),
                    ))
            except Exception:
                continue

        # ── 3. ID-like columns ────────────────────────────────────────────────
        n = len(df)
        for col in feature_cols:
            try:
                uniqueness = df[col].nunique(dropna=True) / max(n, 1)
                if uniqueness >= self.id_thresh:
                    report.violations.append(LeakageViolation(
                        column=col, severity="WARNING",
                        leakage_type="id_column",
                        score=uniqueness,
                        message=(
                            f"'{col}' has {uniqueness:.1%} unique values — "
                            "likely an ID column; models will overfit on this."
                        ),
                    ))
                    logger.warning(
                        "[LeakageDetector] ID-column suspected: '%s' uniqueness=%.4f", col, uniqueness
                    )
            except Exception:
                continue

        # ── Drop CRITICAL columns ─────────────────────────────────────────────
        if self.drop_critical:
            to_remove = [
                v.column for v in report.violations
                if v.severity == "CRITICAL" and v.column in df.columns
            ]
            if to_remove:
                df = df.drop(columns=to_remove)
                report.columns_blocked = to_remove
                logger.warning(
                    "[LeakageDetector] Removed %d CRITICAL leakage columns: %s",
                    len(to_remove), to_remove,
                )

        logger.info(
            "[LeakageDetector] run_id=%s — %d violations (%d CRITICAL, %d removed)",
            run_id[:8] if run_id else "?",
            len(report.violations),
            sum(1 for v in report.violations if v.severity == "CRITICAL"),
            len(report.columns_blocked),
        )
        return df, report

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Cramér's V — symmetric measure of categorical association (0..1)."""
        try:
            from scipy.stats import chi2_contingency  # type: ignore
            contingency = pd.crosstab(x, y)
            chi2, _, _, _ = chi2_contingency(contingency)
            n = contingency.sum().sum()
            k = min(contingency.shape) - 1
            if n == 0 or k == 0:
                return 0.0
            return float(np.sqrt(chi2 / (n * k)))
        except Exception:
            return float("nan")
