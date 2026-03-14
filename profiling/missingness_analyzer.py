"""
profiling/missingness_analyzer.py
-----------------------------------
Step 3 — Data Profiling Engine: Missingness Pattern Analysis.

A senior analyst never treats null values as noise to be discarded.  They ask:
  "Why is this value missing?  Is it random, or is there a pattern?"

This module answers that question using three approaches:

  1. **Row missingness rate**
     Buckets each row by how many of its columns are missing. A spike in
     the 50–100% bucket signals systematic data collection failure for
     certain records.

  2. **Null correlation matrix**
     Converts each column to a boolean null-indicator vector and computes
     pairwise Pearson correlations.  If two columns frequently have null
     values on the SAME rows, they are correlated — which is evidence of
     MAR (Missing At Random, conditional on some other variable) rather
     than MCAR (Missing Completely At Random).

  3. **Column missingness pattern**
     Classifies each column as MCAR, MAR or MNAR_SUSPECTED based on:
       MCAR — no correlation with other null indicators (r < threshold)
       MAR  — correlated with ≥1 other column's null pattern (r ≥ threshold)
       MNAR — not enough information to classify, but null rate is high
              (>30%) and distribution appears non-random (flagged for review)

Usage::

    ma = MissingnessAnalyzer(config)
    result = ma.analyze(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (overridden by config)
# ──────────────────────────────────────────────────────────────────────────────

_MAR_CORR_THRESHOLD:   float = 0.30   # Null-indicator correlation > this → MAR suspected
_HIGH_NULL_THRESHOLD:  float = 0.30   # Null pct > 30% AND no MAR signal → MNAR suspected
_TOP_CORR_PAIRS:       int   = 10     # Maximum correlated null pairs to return


class MissingnessAnalyzer:
    """
    Analyses missing data patterns to guide imputation strategy.

    Args:
        config: Project config dict.  The ``profiling.missingness`` sub-key
                tunes thresholds.  ``None`` uses safe defaults.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("profiling", {}).get("missingness", {})
        self._mar_thresh: float = float(cfg.get("mcar_threshold", _MAR_CORR_THRESHOLD))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs full missingness analysis on the DataFrame.

        Returns:
            {
              "overall_null_pct":       float,
              "complete_rows_pct":      float,
              "row_missingness_buckets": {...},
              "column_patterns":        { col: { null_pct, pattern, mar_correlated_with } },
              "null_corr_pairs":        [ {col_a, col_b, correlation}, ... ],
              "analyst_flags":          [ {...}, ... ],
            }
        """
        if df is None or df.empty:
            return {
                "overall_null_pct":        0.0,
                "complete_rows_pct":       1.0,
                "row_missingness_buckets": {},
                "column_patterns":         {},
                "null_corr_pairs":         [],
                "analyst_flags":           [],
            }

        null_matrix   = df.isnull()
        n_rows, n_cols = df.shape

        overall_null_pct  = float(null_matrix.values.mean())
        complete_rows_pct = float((null_matrix.sum(axis=1) == 0).mean())

        row_buckets   = self._row_missingness_buckets(null_matrix, n_cols)
        null_corr     = self._null_correlation_matrix(null_matrix)
        corr_pairs    = self._top_correlated_null_pairs(null_corr)
        col_patterns  = self._classify_column_patterns(df, null_matrix, null_corr)
        flags         = self._collect_flags(col_patterns, corr_pairs, overall_null_pct)

        logger.info(
            "MissingnessAnalyzer: overall_null_pct=%.2f%%  complete_rows=%.2f%%  "
            "flags=%d",
            overall_null_pct * 100,
            complete_rows_pct * 100,
            len(flags),
        )

        return {
            "overall_null_pct":        round(overall_null_pct, 6),
            "complete_rows_pct":       round(complete_rows_pct, 6),
            "row_missingness_buckets": row_buckets,
            "column_patterns":         col_patterns,
            "null_corr_pairs":         corr_pairs,
            "analyst_flags":           flags,
        }

    # ------------------------------------------------------------------
    # Row missingness bucketing
    # ------------------------------------------------------------------

    @staticmethod
    def _row_missingness_buckets(
        null_matrix: pd.DataFrame, n_cols: int
    ) -> Dict[str, int]:
        """
        Counts rows by fraction of columns missing:
          0%       — complete row (no nulls)
          1–25%    — low missingness
          25–75%   — moderate missingness
          75–99%   — high missingness
          100%     — entirely empty row (all columns null)
        """
        if n_cols == 0:
            return {}

        row_null_pct = null_matrix.mean(axis=1)
        return {
            "complete_0pct":        int((row_null_pct == 0).sum()),
            "low_1_25pct":          int(((row_null_pct > 0) & (row_null_pct <= 0.25)).sum()),
            "moderate_25_75pct":    int(((row_null_pct > 0.25) & (row_null_pct <= 0.75)).sum()),
            "high_75_99pct":        int(((row_null_pct > 0.75) & (row_null_pct < 1.0)).sum()),
            "entirely_null_100pct": int((row_null_pct == 1.0).sum()),
        }

    # ------------------------------------------------------------------
    # Null correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _null_correlation_matrix(null_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Pearson correlation of boolean null-indicator columns.

        Columns with zero variance (never null, or always null) are dropped
        before computing — their correlation is undefined.
        """
        # Keep only columns that have SOME nulls (variance > 0 in the indicator)
        has_some_nulls = null_matrix.columns[null_matrix.any(axis=0)]
        if len(has_some_nulls) < 2:
            return pd.DataFrame()

        null_indicator = null_matrix[has_some_nulls].astype(float)
        
        MAX_NULL_SAMPLES = 10000
        if len(null_indicator) > MAX_NULL_SAMPLES:
            sample_indicator = null_indicator.sample(n=MAX_NULL_SAMPLES, random_state=42)
        else:
            sample_indicator = null_indicator

        try:
            corr = sample_indicator.corr(method="pearson")
        except Exception as exc:
            logger.warning("Null correlation computation failed: %s", exc)
            corr = pd.DataFrame()

        return corr

    # ------------------------------------------------------------------
    # Top correlated null pairs
    # ------------------------------------------------------------------

    def _top_correlated_null_pairs(
        self, null_corr: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Extracts the top N pairs with highest null-indicator correlation,
        excluding self-correlations and duplicate pairs.
        """
        if null_corr.empty:
            return []

        pairs: List[Dict[str, Any]] = []
        cols = null_corr.columns.tolist()

        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = null_corr.loc[a, b]
                if pd.isna(r):
                    continue
                pairs.append({"col_a": a, "col_b": b, "null_correlation": round(float(r), 6)})

        # Sort by absolute correlation descending
        pairs.sort(key=lambda p: abs(p["null_correlation"]), reverse=True)
        return pairs[:_TOP_CORR_PAIRS]

    # ------------------------------------------------------------------
    # Column-level pattern classification
    # ------------------------------------------------------------------

    def _classify_column_patterns(
        self,
        df: pd.DataFrame,
        null_matrix: pd.DataFrame,
        null_corr: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classifies each column's missingness mechanism:
          MCAR — Missing Completely At Random
          MAR  — Missing At Random (correlated with other column(s))
          MNAR_SUSPECTED — Missing Not At Random (high null pct, no MAR signal)
          COMPLETE — no missing values
        """
        patterns: Dict[str, Dict[str, Any]] = {}
        n_rows = len(df)

        for col in df.columns:
            null_count = int(null_matrix[col].sum())
            null_pct   = null_count / n_rows if n_rows > 0 else 0.0

            if null_count == 0:
                patterns[col] = {
                    "null_pct":            0.0,
                    "pattern":             "COMPLETE",
                    "mar_correlated_with": [],
                }
                continue

            # Find columns whose null pattern correlates with this column
            mar_partners: List[str] = []
            if col in null_corr.columns:
                for other in null_corr.columns:
                    if other == col:
                        continue
                    r = null_corr.loc[col, other]
                    if not pd.isna(r) and abs(float(r)) >= self._mar_thresh:
                        mar_partners.append(other)

            if mar_partners:
                mechanism = "MAR"
            elif null_pct >= _HIGH_NULL_THRESHOLD:
                mechanism = "MNAR_SUSPECTED"
            else:
                mechanism = "MCAR"

            patterns[col] = {
                "null_pct":            round(null_pct, 6),
                "pattern":             mechanism,
                "mar_correlated_with": mar_partners,
            }

        return patterns

    # ------------------------------------------------------------------
    # Analyst flags
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_flags(
        col_patterns: Dict[str, Dict[str, Any]],
        corr_pairs: List[Dict[str, Any]],
        overall_null_pct: float,
    ) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []

        if overall_null_pct > 0.20:
            flags.append({
                "column": "DATASET",
                "flag":   "HIGH_OVERALL_MISSINGNESS",
                "detail": f"Dataset-wide null rate is {overall_null_pct:.2%}. "
                          "Imputation strategy required before modelling.",
            })

        for col, info in col_patterns.items():
            pattern = info["pattern"]
            if pattern == "MAR":
                flags.append({
                    "column": col,
                    "flag":   "MAR_PATTERN_DETECTED",
                    "detail": (
                        f"Null values in '{col}' are correlated with "
                        f"{info['mar_correlated_with']}. "
                        "Use conditional (model-based) imputation, not simple mean fill."
                    ),
                })
            elif pattern == "MNAR_SUSPECTED":
                flags.append({
                    "column": col,
                    "flag":   "MNAR_SUSPECTED",
                    "detail": (
                        f"'{col}' has {info['null_pct']:.2%} nulls with no "
                        "detectable MAR signal. Values may be missing non-randomly "
                        "(e.g. refusals, systematic dropout). Investigate at source."
                    ),
                })

        # Flag top correlated null pairs explicitly
        for pair in corr_pairs[:3]:  # Top 3 only
            if abs(pair["null_correlation"]) >= 0.70:
                flags.append({
                    "column": f"{pair['col_a']}::{pair['col_b']}",
                    "flag":   "CORRELATED_NULLS",
                    "detail": (
                        f"Null indicators for '{pair['col_a']}' and "
                        f"'{pair['col_b']}' correlate at r={pair['null_correlation']:.3f}. "
                        "Records are likely missing for the same structural reason."
                    ),
                })

        return flags
