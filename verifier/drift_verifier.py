"""
verifier/drift_verifier.py
---------------------------
Production-grade drift verification using PSI, KL-Divergence,
Jensen-Shannon Divergence, and Wasserstein Distance.

Implements Hard Gate 2 drift check — any column exceeding configured
thresholds results in a REJECT decision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.verifier.drift")

# PSI interpretation: < 0.10 = negligible | 0.10–0.25 = moderate | > 0.25 = critical
_PSI_WARN: float = 0.10
_PSI_CRITICAL: float = 0.25
_KL_CRITICAL: float = 0.50
_JS_CRITICAL: float = 0.30
_BINS: int = 10
_EPS: float = 1e-10


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = _BINS) -> float:
    """
    Population Stability Index (PSI) between two numeric distributions.
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    all_vals = np.concatenate([expected, actual])
    bin_edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = np.maximum(exp_counts / (len(expected) + _EPS), _EPS)
    act_pct = np.maximum(act_counts / (len(actual) + _EPS), _EPS)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(abs(psi), 6)


def _compute_kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = _BINS) -> float:
    """KL-Divergence D_KL(P || Q). Unbounded — higher = more drift."""
    all_vals = np.concatenate([p, q])
    bin_edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist = np.maximum(p_hist, _EPS)
    q_hist = np.maximum(q_hist, _EPS)
    kl = float(np.sum(p_hist * np.log(p_hist / q_hist)))
    return round(abs(kl), 6)


def _compute_js_divergence(p: np.ndarray, q: np.ndarray, bins: int = _BINS) -> float:
    """Jensen-Shannon Divergence. Bounded [0, 1] — symmetric KL."""
    all_vals = np.concatenate([p, q])
    bin_edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist = np.maximum(p_hist, _EPS)
    q_hist = np.maximum(q_hist, _EPS)
    m = 0.5 * (p_hist + q_hist)
    js = 0.5 * np.sum(p_hist * np.log(p_hist / m)) + 0.5 * np.sum(q_hist * np.log(q_hist / m))
    return round(float(js), 6)


def _compute_wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    """Earth mover's distance (Wasserstein-1). Normalized by range."""
    try:
        from scipy.stats import wasserstein_distance
        dist = float(wasserstein_distance(p, q))
        rng = max(np.ptp(np.concatenate([p, q])), _EPS)
        return round(dist / rng, 6)
    except Exception:  # noqa: BLE001
        return 0.0


def _classify_drift_psi(psi: float) -> str:
    if psi < _PSI_WARN:
        return "STABLE"
    if psi < _PSI_CRITICAL:
        return "WARN"
    return "DRIFT"


class DriftVerifier:
    """
    Hard gate for data drift during verification phase.

    Computes PSI, KL-Divergence, Jensen-Shannon Divergence, and
    Wasserstein Distance between reference (baseline) and current
    distributions for all numeric columns.

    Pass/fail is based on PSI — the industry standard for data drift
    monitoring in financial and enterprise analytics.
    """

    def __init__(
        self,
        psi_threshold: float = _PSI_CRITICAL,
        kl_threshold: float = _KL_CRITICAL,
        js_threshold: float = _JS_CRITICAL,
        bins: int = _BINS,
        warn_only: bool = False,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.js_threshold = js_threshold
        self.bins = bins
        # warn_only=True: gate always passes but records drift details
        self.warn_only = warn_only

    def verify(
        self,
        current_df: Optional[pd.DataFrame] = None,
        reference_df: Optional[pd.DataFrame] = None,
        drift_scores: Optional[Dict[str, float]] = None,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Verifies drift between current and reference datasets.

        Accepts either:
        - Two DataFrames (current_df, reference_df): full metric computation
        - A pre-computed dict (drift_scores): PSI per column from profiler

        Returns:
            dict with keys: metric, passed, value, severity, column_details, detail
        """
        # ── Path A: pre-computed PSI scores from profiler ──────────────────
        if drift_scores is not None:
            return self._verify_from_scores(drift_scores)

        # ── Path B: compute from DataFrames ─────────────────────────────────
        if current_df is None or reference_df is None:
            return {
                "metric": "drift_psi",
                "value": 0.0,
                "passed": True,
                "severity": "STABLE",
                "column_details": {},
                "detail": "No reference data provided — drift check skipped.",
            }

        cols = columns or [
            c for c in current_df.select_dtypes("number").columns
            if c in reference_df.columns
        ]

        column_details: Dict[str, Dict[str, Any]] = {}
        critical_cols: List[str] = []
        warn_cols: List[str] = []
        max_psi: float = 0.0

        for col in cols:
            cur = current_df[col].dropna().values
            ref = reference_df[col].dropna().values
            if len(cur) < 5 or len(ref) < 5:
                continue

            psi = _compute_psi(ref, cur, bins=self.bins)
            kl = _compute_kl_divergence(ref, cur, bins=self.bins)
            js = _compute_js_divergence(ref, cur, bins=self.bins)
            wass = _compute_wasserstein(ref, cur)
            severity = _classify_drift_psi(psi)

            column_details[col] = {
                "psi": psi,
                "kl_divergence": kl,
                "js_divergence": js,
                "wasserstein": wass,
                "severity": severity,
            }

            if psi > self.psi_threshold or kl > self.kl_threshold or js > self.js_threshold:
                critical_cols.append(col)
            elif psi > _PSI_WARN:
                warn_cols.append(col)

            max_psi = max(max_psi, psi)

        all_passed = (len(critical_cols) == 0) or self.warn_only

        if critical_cols:
            detail = (
                f"Critical drift detected in {len(critical_cols)} column(s): "
                f"{', '.join(critical_cols[:5])}. Max PSI={max_psi:.4f}."
            )
            overall_severity = "DRIFT"
        elif warn_cols:
            detail = (
                f"Moderate drift in {len(warn_cols)} column(s): "
                f"{', '.join(warn_cols[:5])}. Max PSI={max_psi:.4f}."
            )
            overall_severity = "WARN"
        else:
            detail = f"No significant drift detected. Max PSI={max_psi:.4f}."
            overall_severity = "STABLE"

        logger.info(
            "DriftVerifier: severity=%s critical_cols=%d warn_cols=%d max_psi=%.4f",
            overall_severity, len(critical_cols), len(warn_cols), max_psi,
        )

        return {
            "metric": "drift_psi",
            "value": round(max_psi, 6),
            "passed": bool(all_passed),
            "severity": overall_severity,
            "critical_columns": critical_cols,
            "warn_columns": warn_cols,
            "column_details": column_details,
            "detail": detail,
        }

    def _verify_from_scores(self, drift_scores: Dict[str, float]) -> Dict[str, Any]:
        """Legacy path: accepts pre-computed PSI per column from profiler."""
        critical = {col: psi for col, psi in drift_scores.items() if psi > self.psi_threshold}
        warn = {col: psi for col, psi in drift_scores.items()
                if _PSI_WARN < psi <= self.psi_threshold}
        max_psi = max(drift_scores.values(), default=0.0)
        severity = _classify_drift_psi(max_psi)

        return {
            "metric": "drift_psi",
            "value": round(max_psi, 6),
            "passed": bool(len(critical) == 0) or self.warn_only,
            "severity": severity,
            "critical_columns": list(critical.keys()),
            "warn_columns": list(warn.keys()),
            "column_details": {col: {"psi": psi, "severity": _classify_drift_psi(psi)}
                               for col, psi in drift_scores.items()},
            "detail": (
                f"Critical drift in: {', '.join(list(critical.keys())[:5])}"
                if critical else f"No critical drift detected. Max PSI={max_psi:.4f}."
            ),
        }
