"""
stats/confidence_intervals.py
-------------------------------
Enterprise confidence interval estimation suite.

Methods:
  - Exact (t-distribution for means, Wilson for proportions)
  - Bootstrap (percentile, BCa — bias-corrected and accelerated)
  - Bayesian credible intervals (Beta conjugate for Bernoulli, optional)
  - Difference in means CI (two-sample)
  - Difference in proportions CI

Used by: executive_report, stats API, verification layer.

Usage::

    ci = ConfidenceIntervalEstimator()
    result = ci.mean_ci(series, confidence=0.95)
    result = ci.proportion_ci(successes=45, n=100, method="wilson")
    result = ci.bootstrap_ci(series, stat_fn=np.median, n_boot=5000)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.stats.confidence_intervals")


class ConfidenceIntervalEstimator:
    """
    Multi-method confidence interval estimator.

    Usage::

        ci = ConfidenceIntervalEstimator(confidence=0.95)
        print(ci.mean_ci(df["revenue"]))
        print(ci.bootstrap_ci(df["revenue"], stat_fn=np.median))
    """

    def __init__(self, confidence: float = 0.95, n_boot: int = 5000, random_state: int = 42) -> None:
        self.confidence = confidence
        self.alpha      = 1 - confidence
        self.n_boot     = n_boot
        self.rng        = np.random.default_rng(random_state)

    # ── Mean CI (exact t-distribution) ───────────────────────────────────────

    def mean_ci(self, series: pd.Series, confidence: Optional[float] = None) -> Dict[str, Any]:
        """Exact confidence interval for the mean using t-distribution."""
        conf = confidence or self.confidence
        data = series.dropna().values
        n = len(data)
        if n < 2:
            return {"error": "Need ≥ 2 observations for mean CI"}

        mean = float(np.mean(data))
        se   = float(scipy_stats.sem(data))
        t_crit = float(scipy_stats.t.ppf(1 - (1 - conf) / 2, df=n - 1))
        margin = t_crit * se

        return {
            "method": "t_distribution",
            "confidence": conf,
            "n": n,
            "mean": round(mean, 6),
            "std_error": round(se, 6),
            "lower": round(mean - margin, 6),
            "upper": round(mean + margin, 6),
            "margin_of_error": round(margin, 6),
            "t_critical": round(t_crit, 4),
        }

    # ── Proportion CI (Wilson) ────────────────────────────────────────────────

    def proportion_ci(
        self,
        successes: int,
        n: int,
        method: str = "wilson",
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Confidence interval for a proportion."""
        conf = confidence or self.confidence
        alpha = 1 - conf
        p_hat = successes / n if n > 0 else 0.0
        z = float(scipy_stats.norm.ppf(1 - alpha / 2))

        if method == "wilson":
            # Wilson score interval (preferred — valid for small proportions)
            denom = 1 + z ** 2 / n
            center = (p_hat + z ** 2 / (2 * n)) / denom
            half   = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
            lower  = max(0.0, float(center - half))
            upper  = min(1.0, float(center + half))
        else:
            # Wald (simple, can be out-of-bounds for extreme proportions)
            se = np.sqrt(p_hat * (1 - p_hat) / n) if n > 0 else 0.0
            lower = max(0.0, p_hat - z * se)
            upper = min(1.0, p_hat + z * se)

        return {
            "method": method,
            "confidence": conf,
            "successes": successes,
            "n": n,
            "proportion": round(p_hat, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "z_critical": round(z, 4),
        }

    # ── Bootstrap CI ─────────────────────────────────────────────────────────

    def bootstrap_ci(
        self,
        series: pd.Series,
        stat_fn: Callable = np.mean,
        method: str = "bca",
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence interval.

        method: 'percentile' | 'bca' (bias-corrected and accelerated)
        """
        conf  = confidence or self.confidence
        alpha = 1 - conf
        data  = series.dropna().values
        n     = len(data)
        if n < 10:
            return {"error": "Need ≥ 10 observations for bootstrap CI"}

        observed = float(stat_fn(data))

        # Draw bootstrap samples
        boot_stats = np.array([
            float(stat_fn(self.rng.choice(data, size=n, replace=True)))
            for _ in range(self.n_boot)
        ])

        if method == "bca":
            lower, upper = self._bca(data, boot_stats, stat_fn, observed, alpha)
        else:  # percentile
            lower = float(np.percentile(boot_stats, alpha / 2 * 100))
            upper = float(np.percentile(boot_stats, (1 - alpha / 2) * 100))

        return {
            "method": f"bootstrap_{method}",
            "confidence": conf,
            "n": n,
            "n_bootstrap": self.n_boot,
            "observed_statistic": round(observed, 6),
            "lower": round(lower, 6),
            "upper": round(upper, 6),
            "boot_std": round(float(boot_stats.std()), 6),
        }

    def _bca(self, data, boot_stats, stat_fn, observed, alpha):
        """Bias-corrected and accelerated (BCa) bootstrap."""
        # Bias correction
        z0 = float(scipy_stats.norm.ppf(np.mean(boot_stats < observed) + 1e-10))
        # Acceleration (jackknife)
        n = len(data)
        jack_stats = np.array([float(stat_fn(np.delete(data, i))) for i in range(n)])
        jack_mean  = jack_stats.mean()
        numer = np.sum((jack_mean - jack_stats) ** 3)
        denom = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
        a = float(numer / denom) if denom != 0 else 0.0

        z_lo = scipy_stats.norm.ppf(alpha / 2)
        z_hi = scipy_stats.norm.ppf(1 - alpha / 2)
        alpha_lo = float(scipy_stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))))
        alpha_hi = float(scipy_stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))))
        alpha_lo = max(0.001, min(alpha_lo, 0.999))
        alpha_hi = max(0.001, min(alpha_hi, 0.999))
        lower = float(np.percentile(boot_stats, alpha_lo * 100))
        upper = float(np.percentile(boot_stats, alpha_hi * 100))
        return lower, upper

    # ── Two-sample difference CI ──────────────────────────────────────────────

    def difference_in_means_ci(
        self,
        group_a: pd.Series,
        group_b: pd.Series,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Confidence interval for the difference in means (Welch's approach)."""
        conf = confidence or self.confidence
        a, b = group_a.dropna().values, group_b.dropna().values
        na, nb = len(a), len(b)

        diff = float(np.mean(a) - np.mean(b))
        se_sq = a.var(ddof=1) / na + b.var(ddof=1) / nb
        se    = float(np.sqrt(se_sq))

        # Welch-Satterthwaite degrees of freedom
        dof_num = se_sq ** 2
        dof_den = (a.var(ddof=1) / na) ** 2 / (na - 1) + (b.var(ddof=1) / nb) ** 2 / (nb - 1)
        dof = float(dof_num / dof_den) if dof_den > 0 else na + nb - 2
        t_crit = float(scipy_stats.t.ppf(1 - (1 - conf) / 2, df=dof))
        margin = t_crit * se

        return {
            "method": "welch_difference_in_means",
            "confidence": conf,
            "n_a": na, "n_b": nb,
            "mean_a": round(float(np.mean(a)), 6),
            "mean_b": round(float(np.mean(b)), 6),
            "difference": round(diff, 6),
            "std_error": round(se, 6),
            "lower": round(diff - margin, 6),
            "upper": round(diff + margin, 6),
            "dof": round(dof, 2),
            "t_critical": round(t_crit, 4),
            "contains_zero": bool((diff - margin) < 0 < (diff + margin)),
        }
