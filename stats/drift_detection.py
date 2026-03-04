"""
stats/drift_detection.py
-------------------------
Enterprise data distribution drift detection.

Metrics:
  - Population Stability Index (PSI)      — industry standard for model monitoring
  - KL Divergence (Kullback-Leibler)      — information-theoretic drift measure
  - Jensen-Shannon Divergence             — symmetric, bounded version of KL
  - Wasserstein Distance (Earth Mover's)  — geometric distribution difference
  - Chi-square test (categorical drift)   — for categorical columns

Severity classification:
  - PSI < 0.10  → STABLE
  - PSI < 0.20  → MINOR_DRIFT
  - PSI ≥ 0.20  → MAJOR_DRIFT

Usage::

    dd = DriftDetector()
    report = dd.detect(baseline_df, current_df)
    print(report["summary"]["columns_with_major_drift"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.stats.drift_detection")

# PSI thresholds (actuarial / financial industry standard)
PSI_STABLE = 0.10
PSI_MINOR  = 0.20               # >= MINOR, < MAJOR → minor drift
# >= PSI_MINOR → MAJOR_DRIFT


def _safe_log(x: float) -> float:
    return np.log(x) if x > 1e-10 else np.log(1e-10)


class DriftDetector:
    """
    Column-level distribution drift detector.

    Usage::

        dd = DriftDetector(n_bins=10)
        result = dd.detect(reference_df, current_df)
    """

    def __init__(self, n_bins: int = 10, alpha: float = 0.05) -> None:
        self.n_bins = n_bins
        self.alpha = alpha

    # ── Main entry point ──────────────────────────────────────────────────────

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run full drift detection across all (or specified) columns."""
        shared_cols = columns or [
            c for c in reference.columns
            if c in current.columns
        ]

        column_results: Dict[str, Any] = {}
        for col in shared_cols:
            ref_series = reference[col].dropna()
            cur_series = current[col].dropna()

            if len(ref_series) == 0 or len(cur_series) == 0:
                column_results[col] = {"error": "insufficient data"}
                continue

            is_numeric = pd.api.types.is_numeric_dtype(ref_series)
            if is_numeric:
                column_results[col] = self._numeric_drift(col, ref_series, cur_series)
            else:
                column_results[col] = self._categorical_drift(col, ref_series, cur_series)

        # Summary
        major = [c for c, v in column_results.items()
                 if v.get("severity") == "MAJOR_DRIFT"]
        minor = [c for c, v in column_results.items()
                 if v.get("severity") == "MINOR_DRIFT"]
        stable = [c for c, v in column_results.items()
                  if v.get("severity") == "STABLE"]

        overall_severity = (
            "MAJOR_DRIFT" if major else
            "MINOR_DRIFT" if minor else
            "STABLE"
        )

        return {
            "reference_rows": len(reference),
            "current_rows": len(current),
            "columns_checked": len(shared_cols),
            "overall_severity": overall_severity,
            "summary": {
                "columns_with_major_drift": major,
                "columns_with_minor_drift": minor,
                "stable_columns": stable,
            },
            "column_results": column_results,
        }

    # ── Numeric drift ─────────────────────────────────────────────────────────

    def _numeric_drift(
        self, col: str, ref: pd.Series, cur: pd.Series
    ) -> Dict[str, Any]:
        psi = self._psi(ref, cur)
        kl  = self._kl_divergence(ref, cur)
        js  = self._js_divergence(ref, cur)
        wd  = self._wasserstein(ref, cur)
        ks_stat, ks_p = self._ks_test(ref, cur)

        severity = self._psi_severity(psi)

        return {
            "column": col,
            "dtype": "numeric",
            "severity": severity,
            "psi": round(psi, 6),
            "kl_divergence": round(kl, 6),
            "js_divergence": round(js, 6),
            "wasserstein_distance": round(wd, 6),
            "ks_statistic": round(ks_stat, 6),
            "ks_p_value": round(ks_p, 8),
            "ks_significant": bool(ks_p < self.alpha),
            "reference_stats": self._quick_stats(ref),
            "current_stats": self._quick_stats(cur),
            "interpretation": self._interpret_psi(psi, col),
        }

    # ── Categorical drift ─────────────────────────────────────────────────────

    def _categorical_drift(
        self, col: str, ref: pd.Series, cur: pd.Series
    ) -> Dict[str, Any]:
        psi = self._psi_categorical(ref, cur)
        chi2_result = self._chi2_drift(ref, cur)
        severity = self._psi_severity(psi)

        return {
            "column": col,
            "dtype": "categorical",
            "severity": severity,
            "psi": round(psi, 6),
            "chi2_statistic": chi2_result.get("statistic"),
            "chi2_p_value": chi2_result.get("p_value"),
            "chi2_significant": chi2_result.get("significant", False),
            "reference_distribution": ref.value_counts(normalize=True).round(4).to_dict(),
            "current_distribution": cur.value_counts(normalize=True).round(4).to_dict(),
            "new_categories": list(set(cur.unique()) - set(ref.unique())),
            "missing_categories": list(set(ref.unique()) - set(cur.unique())),
            "interpretation": self._interpret_psi(psi, col),
        }

    # ── PSI (numeric) ─────────────────────────────────────────────────────────

    def _psi(self, ref: pd.Series, cur: pd.Series) -> float:
        """Population Stability Index for numeric columns."""
        all_vals = np.concatenate([ref.values, cur.values])
        bins = np.percentile(all_vals, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        ref_pct = (ref_counts + 0.5) / (len(ref) + 0.5 * self.n_bins)
        cur_pct = (cur_counts + 0.5) / (len(cur) + 0.5 * self.n_bins)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)

    # ── PSI (categorical) ─────────────────────────────────────────────────────

    def _psi_categorical(self, ref: pd.Series, cur: pd.Series) -> float:
        all_cats = set(ref.unique()) | set(cur.unique())
        ref_dist = ref.value_counts(normalize=True)
        cur_dist = cur.value_counts(normalize=True)

        psi = 0.0
        for cat in all_cats:
            ref_p = float(ref_dist.get(cat, 0.0)) + 1e-6
            cur_p = float(cur_dist.get(cat, 0.0)) + 1e-6
            psi += (cur_p - ref_p) * np.log(cur_p / ref_p)
        return max(float(psi), 0.0)

    # ── KL divergence ─────────────────────────────────────────────────────────

    def _kl_divergence(self, ref: pd.Series, cur: pd.Series) -> float:
        all_vals = np.concatenate([ref.values, cur.values])
        bins = np.percentile(all_vals, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        ref_p, _ = np.histogram(ref, bins=bins, density=True)
        cur_p, _ = np.histogram(cur, bins=bins, density=True)
        ref_p = ref_p + 1e-10
        cur_p = cur_p + 1e-10
        ref_p /= ref_p.sum()
        cur_p /= cur_p.sum()
        return float(scipy_stats.entropy(cur_p, ref_p))

    # ── JS divergence ─────────────────────────────────────────────────────────

    def _js_divergence(self, ref: pd.Series, cur: pd.Series) -> float:
        all_vals = np.concatenate([ref.values, cur.values])
        bins = np.percentile(all_vals, np.linspace(0, 100, self.n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        ref_p, _ = np.histogram(ref, bins=bins, density=True)
        cur_p, _ = np.histogram(cur, bins=bins, density=True)
        ref_p = ref_p + 1e-10; ref_p /= ref_p.sum()
        cur_p = cur_p + 1e-10; cur_p /= cur_p.sum()
        m = 0.5 * (ref_p + cur_p)
        js = 0.5 * scipy_stats.entropy(ref_p, m) + 0.5 * scipy_stats.entropy(cur_p, m)
        return float(np.clip(js, 0.0, 1.0))

    # ── Wasserstein ───────────────────────────────────────────────────────────

    def _wasserstein(self, ref: pd.Series, cur: pd.Series) -> float:
        return float(scipy_stats.wasserstein_distance(ref.values, cur.values))

    # ── KS test ───────────────────────────────────────────────────────────────

    def _ks_test(self, ref: pd.Series, cur: pd.Series):
        try:
            result = scipy_stats.ks_2samp(ref.values, cur.values)
            return float(result.statistic), float(result.pvalue)
        except Exception:  # noqa: BLE001
            return 0.0, 1.0

    # ── Chi-square for categorical ────────────────────────────────────────────

    def _chi2_drift(self, ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
        all_cats = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = [ref.value_counts().get(c, 0) for c in all_cats]
        cur_counts = [cur.value_counts().get(c, 0) for c in all_cats]
        if sum(ref_counts) == 0 or sum(cur_counts) == 0:
            return {}
        try:
            contingency = np.array([ref_counts, cur_counts])
            chi2, p, dof, _ = scipy_stats.chi2_contingency(contingency)
            return {"statistic": round(float(chi2), 6), "p_value": round(float(p), 8),
                    "significant": bool(p < self.alpha), "dof": dof}
        except Exception:  # noqa: BLE001
            return {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _psi_severity(psi: float) -> str:
        if psi < PSI_STABLE:
            return "STABLE"
        elif psi < PSI_MINOR:
            return "MINOR_DRIFT"
        else:
            return "MAJOR_DRIFT"

    @staticmethod
    def _interpret_psi(psi: float, col: str) -> str:
        if psi < PSI_STABLE:
            return f"'{col}' distribution is stable (PSI={psi:.4f})"
        elif psi < PSI_MINOR:
            return f"'{col}' shows minor drift (PSI={psi:.4f}) — monitor closely"
        else:
            return f"'{col}' shows MAJOR drift (PSI={psi:.4f}) — retrain model or investigate data pipeline"

    @staticmethod
    def _quick_stats(s: pd.Series) -> Dict[str, float]:
        return {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "p25": round(float(s.quantile(0.25)), 6),
            "p50": round(float(s.quantile(0.50)), 6),
            "p75": round(float(s.quantile(0.75)), 6),
        }
