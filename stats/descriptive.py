"""
stats/descriptive.py
--------------------
Enterprise R-style descriptive statistics engine.

Produces a comprehensive statistical summary including:
  - Central tendency: mean, median, mode
  - Dispersion: std, variance, IQR, range, CV
  - Shape: skewness, kurtosis
  - Percentiles: P5, P25, P50, P75, P95
  - Normality tests: Shapiro-Wilk, D'Agostino-Pearson
  - Zero/negative counts, unique counts, null counts
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.stats.descriptive")


class DescriptiveStats:
    """
    Computes comprehensive descriptive statistics for numeric columns.

    Usage::

        ds = DescriptiveStats()
        report = ds.analyze(df)
        summary_df = ds.to_dataframe(report)
    """

    def __init__(self, normality_alpha: float = 0.05) -> None:
        self.normality_alpha = normality_alpha

    def analyze(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute descriptive stats for all (or specified) numeric columns.

        Returns a dict keyed by column name, each value being a stats dict.
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        result: Dict[str, Any] = {}

        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            result[col] = self._stats_for_series(col, series, df[col])

        return {
            "columns": result,
            "dataset_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(cols),
                "columns_analyzed": cols,
            },
        }

    def _stats_for_series(
        self, col: str, series: pd.Series, full_series: pd.Series
    ) -> Dict[str, Any]:
        n = len(series)
        null_count = int(full_series.isna().sum())

        if n == 0:
            return {"column": col, "n": 0, "null_count": null_count, "error": "All values are null"}

        q = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        iqr = float(q[0.75] - q[0.25])

        mean_val = float(series.mean())
        std_val = float(series.std()) if n > 1 else 0.0
        cv = (std_val / mean_val) if mean_val != 0 else None

        # Mode (first mode if multiple)
        mode_result = series.mode()
        mode_val = float(mode_result.iloc[0]) if not mode_result.empty else None

        # Normality tests (only for n >= 8)
        normality_shapiro = self._shapiro(series, n)
        normality_dagostino = self._dagostino(series, n)

        return {
            "column": col,
            "n": n,
            "null_count": null_count,
            "null_pct": round(null_count / (n + null_count) * 100, 2),
            "mean": round(mean_val, 6),
            "median": round(float(q[0.50]), 6),
            "mode": round(mode_val, 6) if mode_val is not None else None,
            "std": round(std_val, 6),
            "variance": round(float(series.var()), 6),
            "cv": round(cv, 4) if cv is not None else None,
            "min": round(float(series.min()), 6),
            "max": round(float(series.max()), 6),
            "range": round(float(series.max() - series.min()), 6),
            "p5": round(float(q[0.05]), 6),
            "p25": round(float(q[0.25]), 6),
            "p50": round(float(q[0.50]), 6),
            "p75": round(float(q[0.75]), 6),
            "p95": round(float(q[0.95]), 6),
            "iqr": round(iqr, 6),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurt()), 4),
            "zero_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum()),
            "unique_count": int(series.nunique()),
            "normality_shapiro": normality_shapiro,
            "normality_dagostino": normality_dagostino,
        }

    def _shapiro(self, series: pd.Series, n: int) -> Dict[str, Any]:
        if n < 8 or n > 5000:
            return {"test": "shapiro-wilk", "skipped": True, "reason": f"n={n} (need 8≤n≤5000)"}
        try:
            stat, p = scipy_stats.shapiro(series)
            return {
                "test": "shapiro-wilk",
                "statistic": round(float(stat), 6),
                "p_value": round(float(p), 6),
                "is_normal": bool(p > self.normality_alpha),
                "conclusion": "NORMAL" if p > self.normality_alpha else "NON_NORMAL",
            }
        except Exception as exc:  # noqa: BLE001
            return {"test": "shapiro-wilk", "error": str(exc)}

    def _dagostino(self, series: pd.Series, n: int) -> Dict[str, Any]:
        if n < 20:
            return {"test": "dagostino-pearson", "skipped": True, "reason": f"n={n} (need n≥20)"}
        try:
            stat, p = scipy_stats.normaltest(series)
            return {
                "test": "dagostino-pearson",
                "statistic": round(float(stat), 6),
                "p_value": round(float(p), 6),
                "is_normal": bool(p > self.normality_alpha),
                "conclusion": "NORMAL" if p > self.normality_alpha else "NON_NORMAL",
            }
        except Exception as exc:  # noqa: BLE001
            return {"test": "dagostino-pearson", "error": str(exc)}

    def to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        """Convert analysis report to a wide-format summary DataFrame."""
        rows = []
        for col, stats in report.get("columns", {}).items():
            row = {"column": col}
            for k, v in stats.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        row[f"{k}.{subk}"] = subv
                else:
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)
