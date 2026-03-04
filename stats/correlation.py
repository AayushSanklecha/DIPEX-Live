"""
stats/correlation.py
---------------------
Enterprise correlation analysis engine.

Produces:
  - Full Pearson / Spearman / Kendall correlation matrices with p-values
  - Pairwise significance matrix (which pairs pass alpha threshold)
  - Variance Inflation Factor (VIF) for multicollinearity diagnostics
  - Point-biserial correlation (numeric vs. binary target)
  - Phi coefficient (binary vs. binary)
  - Target correlation ranking (most correlated features with target)

All results are serialisable dicts for API / dashboard consumption.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.stats.correlation")


class CorrelationAnalyzer:
    """
    Full correlation analysis suite.

    Usage::

        ca = CorrelationAnalyzer()
        report = ca.analyze(df, target="churn")
        print(report["target_correlation"]["ranked"])
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ── Full matrix ───────────────────────────────────────────────────────────

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        method: str = "pearson",
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute a full correlation matrix with p-values.

        Parameters
        ----------
        method : 'pearson' | 'spearman' | 'kendall'
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        sub = df[cols].dropna()

        if len(sub) < 3:
            return {"error": "Insufficient data for correlation matrix (need ≥ 3 rows)"}

        corr = sub.corr(method=method)

        # Compute p-values pairwise
        p_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i == j:
                    continue
                try:
                    if method == "pearson":
                        _, p = scipy_stats.pearsonr(sub[c1], sub[c2])
                    elif method == "spearman":
                        _, p = scipy_stats.spearmanr(sub[c1], sub[c2])
                    else:  # kendall
                        _, p = scipy_stats.kendalltau(sub[c1], sub[c2])
                    p_matrix.loc[c1, c2] = p
                except Exception:  # noqa: BLE001
                    pass

        # Strong correlations (|r| > 0.7, significant)
        strong_pairs = []
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i >= j:
                    continue
                r = float(corr.loc[c1, c2])
                p = float(p_matrix.loc[c1, c2])
                if abs(r) > 0.7 and p < self.alpha:
                    strong_pairs.append({
                        "col_a": c1, "col_b": c2,
                        "r": round(r, 6), "p_value": round(p, 8),
                        "significant": True,
                        "strength": "very_strong" if abs(r) > 0.9 else "strong",
                    })

        return {
            "method": method,
            "n_observations": len(sub),
            "columns": cols,
            "correlation_matrix": {
                c: {c2: round(float(corr.loc[c, c2]), 6) for c2 in cols}
                for c in cols
            },
            "p_value_matrix": {
                c: {c2: round(float(p_matrix.loc[c, c2]), 8) for c2 in cols}
                for c in cols
            },
            "strong_correlations": sorted(strong_pairs, key=lambda x: abs(x["r"]), reverse=True),
            "significance_threshold": self.alpha,
        }

    # ── VIF ──────────────────────────────────────────────────────────────────

    def vif_analysis(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Variance Inflation Factor — detect multicollinearity."""
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        sub = df[cols].dropna()

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X = sub.values
            vif_data = []
            for i, col in enumerate(cols):
                try:
                    vif = float(variance_inflation_factor(X, i))
                    vif_data.append({
                        "feature": col,
                        "vif": round(vif, 4),
                        "multicollinearity": (
                            "SEVERE" if vif > 10 else
                            "MODERATE" if vif > 5 else
                            "LOW"
                        ),
                    })
                except Exception:  # noqa: BLE001
                    vif_data.append({"feature": col, "vif": None, "multicollinearity": "UNKNOWN"})
            return {
                "vif_scores": sorted(vif_data, key=lambda x: (x["vif"] or 0), reverse=True),
                "high_vif_features": [d["feature"] for d in vif_data if (d["vif"] or 0) > 5],
                "interpretation": "VIF > 10 indicates severe multicollinearity; > 5 moderate.",
            }
        except ImportError:
            return {"error": "statsmodels required for VIF: pip install statsmodels"}

    # ── Target correlation ────────────────────────────────────────────────────

    def target_correlation(
        self,
        df: pd.DataFrame,
        target: str,
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """Rank all numeric features by their correlation with a target column."""
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
        if target not in df.columns:
            return {"error": f"Target '{target}' not found in DataFrame"}

        ranked = []
        for col in num_cols:
            sub = df[[col, target]].dropna()
            if len(sub) < 5:
                continue
            try:
                if method == "pearson":
                    r, p = scipy_stats.pearsonr(sub[col], sub[target])
                elif method == "spearman":
                    r, p = scipy_stats.spearmanr(sub[col], sub[target])
                else:
                    r, p = scipy_stats.kendalltau(sub[col], sub[target])
                ranked.append({
                    "feature": col,
                    "r": round(float(r), 6),
                    "abs_r": round(abs(float(r)), 6),
                    "p_value": round(float(p), 8),
                    "significant": bool(p < self.alpha),
                })
            except Exception:  # noqa: BLE001
                pass

        ranked.sort(key=lambda x: x["abs_r"], reverse=True)
        return {
            "target": target,
            "method": method,
            "ranked": ranked,
            "top_features": [r["feature"] for r in ranked[:10] if r["significant"]],
        }

    # ── Full analysis ─────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        method: str = "spearman",
    ) -> Dict[str, Any]:
        """Run the full correlation analysis suite."""
        report: Dict[str, Any] = {}
        report["correlation_matrix"] = self.correlation_matrix(df, method=method)
        report["vif"] = self.vif_analysis(df)
        if target and target in df.columns:
            report["target_correlation"] = self.target_correlation(df, target, method=method)
        return report
