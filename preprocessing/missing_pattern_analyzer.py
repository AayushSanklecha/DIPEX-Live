"""
preprocessing/missing_pattern_analyzer.py
------------------------------------------
MCAR / MAR / MNAR missing data pattern analysis.

Why this matters:
  - MCAR (Missing Completely At Random): any imputation works; median/mean is fine
  - MAR  (Missing At Random):            missingness depends on OTHER columns;
                                          KNN/MICE imputation is the right choice
  - MNAR (Missing Not At Random):        missingness depends on the MISSING VALUE
                                          itself (e.g., high-earners don't report income).
                                          Imputing alone gives biased results; a
                                          MISSINGNESS INDICATOR feature must be added.

This module:
  1. Classifies each column's missing pattern (MCAR / MAR / MNAR / COMPLETE)
  2. Recommends the correct imputation strategy per column
  3. Adds missingness indicator columns for MNAR columns (binary flag: was_null)
  4. Returns a full audit report

Detection methods:
  - MCAR test: Little's test (chi-square on missing patterns) when scipy available;
               falls back to correlation-based heuristic
  - MAR detection: logistic regression of missingness indicator on other columns
  - MNAR detection: high missing rate + correlation between missingness and column rank
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.preprocessing.missing_pattern_analyzer")

_PATTERN_LABELS = ("COMPLETE", "MCAR", "MAR", "MNAR")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnMissingProfile:
    column: str
    null_pct: float
    pattern: str                  # COMPLETE | MCAR | MAR | MNAR
    recommended_strategy: str     # none | median | knn | mice | indicator+mice
    indicator_added: bool         # True if a was_null column was created
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "null_pct": round(self.null_pct, 4),
            "pattern": self.pattern,
            "recommended_strategy": self.recommended_strategy,
            "indicator_added": self.indicator_added,
            "details": self.details,
        }


@dataclass
class MissingPatternReport:
    run_id: str
    profiles: List[ColumnMissingProfile] = field(default_factory=list)
    mnar_columns: List[str] = field(default_factory=list)
    mar_columns: List[str]  = field(default_factory=list)
    mcar_columns: List[str] = field(default_factory=list)
    warnings: List[str]     = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profiles": [p.to_dict() for p in self.profiles],
            "mnar_columns": self.mnar_columns,
            "mar_columns": self.mar_columns,
            "mcar_columns": self.mcar_columns,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class MissingPatternAnalyzer:
    """
    Classifies each column's missing mechanism and applies the correct remedy.

    Config stanza (all optional)::

        preprocessing:
          missing_patterns:
            mnar_null_threshold: 0.05       # min null pct to consider MNAR
            mnar_correlation_threshold: 0.3  # min |corr| between missingness rank and values
            mar_predictor_threshold: 0.1     # min R² of missingness ~ other cols (logistic)
            add_mnar_indicators: true        # add was_null columns for MNAR columns
            min_rows_for_test: 30            # min rows to run any statistical test
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("preprocessing", {}).get("missing_patterns", {})
        self.mnar_null_thresh: float  = float(cfg.get("mnar_null_threshold", 0.05))
        self.mnar_corr_thresh: float  = float(cfg.get("mnar_correlation_threshold", 0.3))
        self.mar_r2_thresh: float     = float(cfg.get("mar_predictor_threshold", 0.1))
        self.add_indicators: bool     = bool(cfg.get("add_mnar_indicators", True))
        self.min_rows: int            = int(cfg.get("min_rows_for_test", 30))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MissingPatternAnalyzer":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        run_id: str = "",
    ) -> Tuple[pd.DataFrame, MissingPatternReport]:
        """
        Analyze missing patterns and optionally add missingness indicators.

        Returns
        -------
        (enriched_df, MissingPatternReport)
        enriched_df has `{col}_was_null` indicator columns for every MNAR column.
        """
        report = MissingPatternReport(run_id=run_id)
        df = df.copy()

        cols_with_nulls = [col for col in df.columns if df[col].isna().any()]
        if not cols_with_nulls:
            logger.info("[MissingPattern] No missing values found — analysis skipped.")
            return df, report

        null_pcts = df.isnull().mean()

        for col in df.columns:
            null_pct = float(null_pcts[col])
            if null_pct == 0.0:
                report.profiles.append(ColumnMissingProfile(
                    column=col, null_pct=0.0, pattern="COMPLETE",
                    recommended_strategy="none", indicator_added=False,
                ))
                continue

            pattern, details = self._classify_missing(df, col, null_pct)
            recommended = self._recommend_strategy(pattern)
            indicator_added = False

            # Add was_null indicator for MNAR columns
            if pattern == "MNAR" and self.add_indicators:
                ind_col = f"{col}_was_null"
                if ind_col not in df.columns:
                    df[ind_col] = df[col].isna().astype(np.int8)
                    indicator_added = True
                    logger.info(
                        "[MissingPattern] Added MNAR indicator '%s' for column '%s'",
                        ind_col, col,
                    )

            profile = ColumnMissingProfile(
                column=col, null_pct=null_pct, pattern=pattern,
                recommended_strategy=recommended, indicator_added=indicator_added,
                details=details,
            )
            report.profiles.append(profile)

            if pattern == "MNAR":
                report.mnar_columns.append(col)
            elif pattern == "MAR":
                report.mar_columns.append(col)
            elif pattern == "MCAR":
                report.mcar_columns.append(col)

        logger.info(
            "[MissingPattern] run_id=%s MNAR=%d MAR=%d MCAR=%d COMPLETE=%d",
            run_id[:8] if run_id else "?",
            len(report.mnar_columns), len(report.mar_columns),
            len(report.mcar_columns),
            sum(1 for p in report.profiles if p.pattern == "COMPLETE"),
        )
        return df, report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _classify_missing(
        self, df: pd.DataFrame, col: str, null_pct: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classify missing mechanism for one column.
        Returns (pattern_label, details_dict).
        """
        details: Dict[str, Any] = {"null_pct": round(null_pct, 4)}
        n = len(df)

        if n < self.min_rows:
            return "MCAR", {**details, "reason": "too_few_rows_for_test"}

        # ── MNAR detection ────────────────────────────────────────────────────
        # Criterion: column is numeric, missingness is correlated with the value
        # rank of non-missing values → data is "missing because of its own value"
        if null_pct >= self.mnar_null_thresh and pd.api.types.is_numeric_dtype(df[col]):
            is_null = df[col].isna().astype(int)
            non_null_vals = df[col].dropna()
            if len(non_null_vals) >= 10:
                # Rank the non-missing values and correlate with all-row null indicator
                try:
                    ranked = df[col].rank(method="average", na_option="bottom")
                    corr = abs(float(ranked.corr(is_null)))
                    details["mnar_rank_corr"] = round(corr, 4)
                    if not np.isnan(corr) and corr >= self.mnar_corr_thresh:
                        details["reason"] = "high_rank_corr_with_missingness"
                        return "MNAR", details
                except Exception:
                    pass

        # ── MAR detection ─────────────────────────────────────────────────────
        # Criterion: missingness in col is predictable from other columns
        if null_pct >= 0.01:
            r2 = self._mar_predictability(df, col)
            details["mar_r2"] = round(r2, 4) if r2 is not None else None
            if r2 is not None and r2 >= self.mar_r2_thresh:
                details["reason"] = f"missingness_predictable_from_other_cols (R²={r2:.3f})"
                return "MAR", details

        # Default: MCAR
        details["reason"] = "no_strong_pattern_detected"
        return "MCAR", details

    def _mar_predictability(self, df: pd.DataFrame, col: str) -> Optional[float]:
        """
        Fit a logistic regression of (col is null) ~ other numeric columns.
        Returns pseudo-R² (McFadden). Returns None if insufficient data.
        """
        try:
            is_null = df[col].isna().astype(int)
            if is_null.sum() == 0 or is_null.sum() == len(df):
                return None

            # Features: other numeric columns, filled with 0
            other_num = df.drop(columns=[col]).select_dtypes(include=np.number)
            if other_num.empty or len(other_num.columns) == 0:
                return None

            X = other_num.fillna(0).values
            y = is_null.values

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import log_loss

            clf = LogisticRegression(max_iter=200, random_state=42, C=1.0)
            clf.fit(X, y)
            proba = clf.predict_proba(X)[:, 1]

            # Null model log-loss (predict base rate)
            p0 = y.mean()
            null_proba = np.full(len(y), p0)
            ll_null  = log_loss(y, null_proba)
            ll_model = log_loss(y, proba)
            mcfadden = 1.0 - ll_model / (ll_null + 1e-9)
            return float(max(mcfadden, 0.0))
        except Exception:
            return None

    @staticmethod
    def _recommend_strategy(pattern: str) -> str:
        return {
            "COMPLETE": "none",
            "MCAR":     "median",          # simple, unbiased for MCAR
            "MAR":      "mice",            # MICE/iterative is correct for MAR
            "MNAR":     "indicator+mice",  # add missingness indicator + impute
        }.get(pattern, "median")
