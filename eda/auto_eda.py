"""
eda/auto_eda.py
---------------
AI & ANALYTICS SERVICE LAYER — Automated EDA

AutoEDA.run(df) produces a structured EDAReport covering:
  - Distributions  (numeric descriptive stats + categorical value counts)
  - Correlation    (top-N correlated pairs via Pearson / Spearman)
  - Outlier detection (IQR + Z-score flags)
  - Missing value patterns (per-column + block patterns)
  - Data type summary

Reuses existing DIPEX modules (stats/, profiling/) where available.
No new external dependencies beyond pandas / numpy (already in requirements).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.eda.auto_eda")


# ── EDA Report ────────────────────────────────────────────────────────────────

@dataclass
class EDAReport:
    """JSON-serialisable report produced by AutoEDA."""
    dataset_shape: Tuple[int, int] = (0, 0)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    distributions: Dict[str, Any] = field(default_factory=dict)
    correlations: Dict[str, Any] = field(default_factory=dict)
    outliers: Dict[str, Any] = field(default_factory=dict)
    missing_patterns: Dict[str, Any] = field(default_factory=dict)
    dtype_summary: Dict[str, str] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    html_report_path: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "dataset_shape": list(self.dataset_shape),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "distributions": self.distributions,
            "correlations": self.correlations,
            "outliers": self.outliers,
            "missing_patterns": self.missing_patterns,
            "dtype_summary": self.dtype_summary,
            "insights": self.insights,
            "html_report_path": self.html_report_path,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ── AutoEDA ───────────────────────────────────────────────────────────────────

class AutoEDA:
    """
    Automated Exploratory Data Analysis engine.

    Usage::

        eda = AutoEDA(config=config)
        report = eda.run(df)
        print(report.insights)       # list of human-readable findings
        data = report.to_dict()      # JSON-serialisable
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        top_n_correlations: int = 10,
        outlier_z_thresh: float = 3.0,
        max_cat_unique: int = 50,
    ):
        self.config = config or {}
        self.top_n_correlations = top_n_correlations
        self.outlier_z_thresh = outlier_z_thresh
        self.max_cat_unique = max_cat_unique

    def run(self, df: pd.DataFrame, run_id: Optional[str] = None) -> EDAReport:
        """Run full automated EDA on df and return an EDAReport."""
        t0 = time.perf_counter()
        if df is None or df.empty:
            return EDAReport()

        report = EDAReport(dataset_shape=df.shape)

        # ── Column categorisation ──────────────────────────────────────────
        report.numeric_columns = list(df.select_dtypes(include="number").columns)
        report.categorical_columns = list(df.select_dtypes(exclude="number").columns)
        report.dtype_summary = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # ── Distributions ──────────────────────────────────────────────────
        report.distributions = self._distributions(df, report.numeric_columns, report.categorical_columns)

        # ── Correlations ───────────────────────────────────────────────────
        report.correlations = self._correlations(df, report.numeric_columns)

        # ── Outliers ───────────────────────────────────────────────────────
        report.outliers = self._outliers(df, report.numeric_columns)

        # ── Missing value patterns ─────────────────────────────────────────
        report.missing_patterns = self._missing_patterns(df)

        # ── Auto-insights ──────────────────────────────────────────────────
        report.insights = self._generate_insights(df, report)

        # ── HTML Visual Report (Priority 4 Gap Fix) ────────────────────────
        report.html_report_path = self._generate_html_report(df, run_id)

        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[AutoEDA][%s] rows=%d cols=%d numeric=%d cat=%d insights=%d elapsed=%.0fms",
            (run_id or "")[:8], df.shape[0], df.shape[1],
            len(report.numeric_columns), len(report.categorical_columns),
            len(report.insights), report.elapsed_ms,
        )
        return report

    # ── Private analysis methods ──────────────────────────────────────────────

    def _generate_html_report(self, df: pd.DataFrame, run_id: Optional[str]) -> Optional[str]:
        """
        Generates a visually rich HTML Data Profiling report if ydata-profiling is available.
        """
        try:
            from ydata_profiling import ProfileReport
            import os
            
            # Ensure output directory exists (using a temp-like directory for now, 
            # in a real system this would go to a specific run_id folder like /data/reports/)
            output_dir = "reports_output"
            os.makedirs(output_dir, exist_ok=True)
            
            run_str = run_id or f"run_{int(time.time())}"
            filepath = os.path.join(output_dir, f"eda_profile_{run_str}.html")
            
            # Subsample if dataset is absolutely massive to save memory
            df_prof = df.sample(min(len(df), 10000), random_state=42) if len(df) > 10000 else df
            
            logger.info("Generating ydata-profiling HTML report...")
            profile = ProfileReport(df_prof, title=f"DIPEX Profiling Report ({run_str})", explorative=True, minimal=len(df)>5000)
            profile.to_file(filepath)
            
            logger.info("Saved EDA HTML report to %s", filepath)
            return filepath
            
        except ImportError:
            logger.info("ydata-profiling not installed. Skipping interactive HTML EDA report generation.")
            return None
        except Exception as e:
            logger.warning("Failed to generate EDA HTML profile: %s", e)
            return None

    def _distributions(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: List[str],
    ) -> Dict:
        out: Dict = {"numeric": {}, "categorical": {}}

        # Numeric — try existing stats module first, fall back to pandas
        for col in numeric_cols:
            s = df[col].dropna()
            try:
                from stats.descriptive import DescriptiveStats
                stat_out = DescriptiveStats().analyze(df[[col]])
                out["numeric"][col] = stat_out.get(col, {}) or {}
            except Exception:
                out["numeric"][col] = {
                    "count": int(s.count()),
                    "mean": float(s.mean()) if len(s) else None,
                    "std": float(s.std()) if len(s) else None,
                    "min": float(s.min()) if len(s) else None,
                    "25%": float(s.quantile(0.25)) if len(s) else None,
                    "50%": float(s.quantile(0.50)) if len(s) else None,
                    "75%": float(s.quantile(0.75)) if len(s) else None,
                    "max": float(s.max()) if len(s) else None,
                    "skewness": float(s.skew()) if len(s) > 2 else None,
                    "kurtosis": float(s.kurtosis()) if len(s) > 2 else None,
                }

        # Categorical — top value counts
        for col in cat_cols:
            s = df[col].dropna()
            n_unique = s.nunique()
            if n_unique <= self.max_cat_unique:
                vc = s.value_counts().head(20).to_dict()
                out["categorical"][col] = {
                    "n_unique": int(n_unique),
                    "top_values": {str(k): int(v) for k, v in vc.items()},
                }
            else:
                out["categorical"][col] = {
                    "n_unique": int(n_unique),
                    "note": f"High cardinality ({n_unique} unique values)",
                }

        return out

    def _correlations(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        if len(numeric_cols) < 2:
            return {"pairs": [], "note": "fewer than 2 numeric columns"}

        corr_df = df[numeric_cols].corr(method="pearson")
        pairs = []
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1 :]:
                val = corr_df.loc[col_a, col_b]
                if not np.isnan(val):
                    pairs.append({"col_a": col_a, "col_b": col_b, "pearson_r": round(float(val), 4)})

        # Sort by absolute correlation, return top-N
        pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        top = pairs[: self.top_n_correlations]

        # Try Spearman for top pairs
        for pair in top:
            try:
                sp = df[[pair["col_a"], pair["col_b"]]].dropna()
                if len(sp) > 2:
                    spearman_r = sp[pair["col_a"]].corr(sp[pair["col_b"]], method="spearman")
                    pair["spearman_r"] = round(float(spearman_r), 4)
            except Exception:
                pass

        return {
            "top_pairs": top,
            "strong_correlations": [p for p in top if abs(p["pearson_r"]) >= 0.7],
        }

    def _outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        out: Dict = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            col_stats: Dict = {}

            # IQR method
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            iqr_outliers = int(((s < iqr_lower) | (s > iqr_upper)).sum())
            col_stats["iqr"] = {
                "lower_fence": round(iqr_lower, 4),
                "upper_fence": round(iqr_upper, 4),
                "outlier_count": iqr_outliers,
                "outlier_pct": round(iqr_outliers / len(s) * 100, 2),
            }

            # Z-score method
            mean, std = float(s.mean()), float(s.std())
            if std > 0:
                z_outliers = int((np.abs((s - mean) / std) > self.outlier_z_thresh).sum())
                col_stats["zscore"] = {
                    "threshold": self.outlier_z_thresh,
                    "outlier_count": z_outliers,
                    "outlier_pct": round(z_outliers / len(s) * 100, 2),
                }

            out[col] = col_stats

        total_outlier_cols = sum(
            1 for v in out.values()
            if v.get("iqr", {}).get("outlier_count", 0) > 0
        )
        return {"by_column": out, "columns_with_outliers": total_outlier_cols}

    def _missing_patterns(self, df: pd.DataFrame) -> Dict:
        """Per-column missing stats + block pattern detection."""
        total = len(df)
        per_col = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            per_col[col] = {
                "null_count": null_count,
                "null_pct": round(null_count / total * 100, 2) if total else 0.0,
            }

        # Try existing profiling module
        try:
            from profiling.missingness_analyzer import MissingnessAnalyzer
            analyzer = MissingnessAnalyzer()
            patterns = analyzer.analyze(df)
            return {"per_column": per_col, "pattern_analysis": patterns}
        except Exception:
            pass

        high_missing = [c for c, v in per_col.items() if v["null_pct"] > 20]
        return {
            "per_column": per_col,
            "high_missing_columns": high_missing,
            "total_missing_cells": int(df.isnull().sum().sum()),
        }

    def _generate_insights(self, df: pd.DataFrame, report: EDAReport) -> List[str]:
        """Generate human-readable insight strings from the EDA report."""
        insights = []

        # Dataset size
        rows, cols = report.dataset_shape
        insights.append(f"Dataset has {rows:,} rows and {cols} columns "
                        f"({len(report.numeric_columns)} numeric, {len(report.categorical_columns)} categorical).")

        # Missing data
        total_missing = report.missing_patterns.get("total_missing_cells", 0)
        if total_missing:
            pct = round(total_missing / (rows * cols) * 100, 1) if rows * cols else 0
            insights.append(f"{total_missing:,} missing values detected ({pct}% of all cells).")
        high_miss = report.missing_patterns.get("high_missing_columns", [])
        if high_miss:
            insights.append(f"Columns with >20% missing: {', '.join(high_miss[:5])}.")

        # Outliers
        outlier_cols = report.outliers.get("columns_with_outliers", 0)
        if outlier_cols:
            insights.append(f"{outlier_cols} numeric column(s) contain IQR outliers.")

        # Strong correlations
        strong = report.correlations.get("strong_correlations", [])
        if strong:
            top = strong[0]
            insights.append(
                f"Strong correlation (r={top['pearson_r']}) found between "
                f"'{top['col_a']}' and '{top['col_b']}'."
            )

        # Skewness warnings
        for col, stats in report.distributions.get("numeric", {}).items():
            skew = stats.get("skewness")
            if skew is not None and abs(float(skew)) > 2:
                insights.append(f"Column '{col}' is highly skewed (skewness={round(skew, 2)}).")

        return insights
