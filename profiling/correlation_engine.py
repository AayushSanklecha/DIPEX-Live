"""
profiling/correlation_engine.py
---------------------------------
Step 3 — Data Profiling Engine: Pairwise Correlation Analysis.

Answers: "Which columns move together, and how strongly?"

Methods:
  Pearson  — linear correlation between numeric columns
             (sensitive to outliers and assumes linearity)
  Spearman — rank-based correlation between numeric columns
             (robust to outliers, captures monotonic relationships)
  Cramér's V — association strength between two categorical columns
               (chi-squared statistic normalised into [0, 1])

Output:
    {
      "pearson":   {(col_a, col_b): r},
      "spearman":  {(col_a, col_b): rho},
      "cramers_v": {(col_a, col_b): V},
      "highlights": [
        {"columns": [col_a, col_b], "method": "pearson",
         "value": 0.97, "flag": "NEAR_DUPLICATE"},
        ...
      ]
    }

Usage::

    ce = CorrelationEngine(config)
    result = ce.compute(df)
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (overridden by config)
# ──────────────────────────────────────────────────────────────────────────────

_STRONG_CORR_THRESHOLD: float   = 0.80
_NEAR_DUP_THRESHOLD:    float   = 0.95
_MIN_SAMPLE_FOR_CORR:   int    = 5     # Minimum non-null rows to compute correlation


class CorrelationEngine:
    """
    Computes Pearson, Spearman and Cramér's V correlation matrices.

    Args:
        config: Project config dict.  The ``profiling.correlation`` sub-key
                controls thresholds.  ``None`` uses safe defaults.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("profiling", {}).get("correlation", {})
        self._strong_thresh:  float = float(cfg.get("strong_threshold",        _STRONG_CORR_THRESHOLD))
        self._near_dup_thresh: float = float(cfg.get("near_duplicate_threshold", _NEAR_DUP_THRESHOLD))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes full pairwise correlation analysis.

        Returns:
            Dict with keys:
              ``pearson``   — {(col_a, col_b): float}
              ``spearman``  — {(col_a, col_b): float}
              ``cramers_v`` — {(col_a, col_b): float}
              ``highlights``— list of flagged high-correlation pairs
        """
        if df is None or df.empty:
            return {"pearson": {}, "spearman": {}, "cramers_v": {}, "highlights": []}

        MAX_CORR_SAMPLES = 5000
        if len(df) > MAX_CORR_SAMPLES:
            logger.info("CorrelationEngine: dataset too large (%d rows), sampling %d rows for faster pairwise correlations.", len(df), MAX_CORR_SAMPLES)
            sample_df = df.sample(n=MAX_CORR_SAMPLES, random_state=42)
        else:
            sample_df = df

        num_cols  = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()

        pearson   = self._compute_pearson(sample_df, num_cols)
        spearman  = self._compute_spearman(sample_df, num_cols)
        cramers_v = self._compute_cramers_v(sample_df, cat_cols)
        highlights = self._collect_highlights(pearson, spearman, cramers_v)

        n_pairs = len(pearson) + len(cramers_v)
        logger.info(
            "CorrelationEngine: %d numeric pair(s), %d categorical pair(s), %d highlight(s).",
            len(pearson), len(cramers_v), len(highlights),
        )

        return {
            "pearson":    self._serialise(pearson),
            "spearman":   self._serialise(spearman),
            "cramers_v":  self._serialise(cramers_v),
            "highlights": highlights,
        }

    # ------------------------------------------------------------------
    # Pearson correlation
    # ------------------------------------------------------------------

    def _compute_pearson(
        self, df: pd.DataFrame, cols: List[str]
    ) -> Dict[Tuple[str, str], float]:
        results: Dict[Tuple[str, str], float] = {}
        for a, b in itertools.combinations(cols, 2):
            sub = df[[a, b]].dropna()
            if len(sub) < _MIN_SAMPLE_FOR_CORR:
                continue
            r, _ = scipy_stats.pearsonr(sub[a], sub[b])
            results[(a, b)] = round(float(r), 6)
        return results

    # ------------------------------------------------------------------
    # Spearman rank correlation
    # ------------------------------------------------------------------

    def _compute_spearman(
        self, df: pd.DataFrame, cols: List[str]
    ) -> Dict[Tuple[str, str], float]:
        results: Dict[Tuple[str, str], float] = {}
        for a, b in itertools.combinations(cols, 2):
            sub = df[[a, b]].dropna()
            if len(sub) < _MIN_SAMPLE_FOR_CORR:
                continue
            rho, _ = scipy_stats.spearmanr(sub[a], sub[b])
            results[(a, b)] = round(float(rho), 6)
        return results

    # ------------------------------------------------------------------
    # Cramér's V (categorical association)
    # ------------------------------------------------------------------

    def _compute_cramers_v(
        self, df: pd.DataFrame, cols: List[str]
    ) -> Dict[Tuple[str, str], float]:
        results: Dict[Tuple[str, str], float] = {}
        for a, b in itertools.combinations(cols, 2):
            sub = df[[a, b]].dropna()
            if len(sub) < _MIN_SAMPLE_FOR_CORR:
                continue
            try:
                v = self._cramers_v_stat(sub[a], sub[b])
                results[(a, b)] = round(float(v), 6)
            except Exception as exc:
                logger.debug(
                    "Cramér's V failed for (%s, %s): %s — skipping.", a, b, exc
                )
        return results

    @staticmethod
    def _cramers_v_stat(x: pd.Series, y: pd.Series) -> float:
        """
        Computes bias-corrected Cramér's V.

        Reference: Bergsma (2013) — consistent non-zero estimator.
        V = 0 → no association
        V = 1 → perfect association
        """
        contingency = pd.crosstab(x, y)
        n = contingency.values.sum()
        if n == 0:
            return 0.0

        chi2 = scipy_stats.chi2_contingency(contingency, correction=False)[0]
        r, k = contingency.shape

        # Bias-corrected form
        phi2      = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
        r_corr    = r - (r - 1) ** 2 / (n - 1)
        k_corr    = k - (k - 1) ** 2 / (n - 1)
        denominator = min(r_corr - 1, k_corr - 1)

        if denominator <= 0:
            return 0.0
        return float(np.sqrt(phi2 / denominator))

    # ------------------------------------------------------------------
    # Highlight collection
    # ------------------------------------------------------------------

    def _collect_highlights(
        self,
        pearson:   Dict[Tuple[str, str], float],
        spearman:  Dict[Tuple[str, str], float],
        cramers_v: Dict[Tuple[str, str], float],
    ) -> List[Dict[str, Any]]:
        highlights: List[Dict[str, Any]] = []

        for method_name, matrix in (("pearson", pearson), ("spearman", spearman)):
            for (a, b), val in matrix.items():
                abs_val = abs(val)
                if abs_val >= self._near_dup_thresh:
                    flag = "NEAR_DUPLICATE_COLUMNS"
                    detail = (
                        f"{method_name.title()} r={val:.4f}. "
                        "Columns are nearly collinear — consider removing one to avoid "
                        "multicollinearity in regression / inflated feature importance."
                    )
                elif abs_val >= self._strong_thresh:
                    flag = "STRONG_CORRELATION"
                    detail = (
                        f"{method_name.title()} r={val:.4f}. "
                        "Review for feature redundancy or causal relationship."
                    )
                else:
                    continue

                highlights.append({
                    "columns": [a, b],
                    "method":  method_name,
                    "value":   val,
                    "flag":    flag,
                    "detail":  detail,
                })

        for (a, b), val in cramers_v.items():
            if val >= self._near_dup_thresh:
                highlights.append({
                    "columns": [a, b],
                    "method":  "cramers_v",
                    "value":   val,
                    "flag":    "NEAR_DUPLICATE_COLUMNS",
                    "detail":  f"Cramér's V={val:.4f}. Near-perfect categorical association.",
                })
            elif val >= self._strong_thresh:
                highlights.append({
                    "columns": [a, b],
                    "method":  "cramers_v",
                    "value":   val,
                    "flag":    "STRONG_CATEGORICAL_ASSOCIATION",
                    "detail":  f"Cramér's V={val:.4f}. Strong association between categories.",
                })

        # Sort by absolute value descending so most correlated pairs appear first
        highlights.sort(key=lambda h: abs(h["value"]), reverse=True)
        return highlights

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(
        matrix: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """Converts tuple-key dict to JSON-safe string-key dict."""
        return {f"{a}::{b}": v for (a, b), v in matrix.items()}
