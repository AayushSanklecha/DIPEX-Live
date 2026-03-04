"""
stats/hypothesis_tests.py
--------------------------
Enterprise R-style hypothesis testing engine.

Tests supported:
  - One-sample t-test
  - Two-sample t-test (Welch's, equal/unequal variance)
  - Paired t-test
  - One-way ANOVA
  - Chi-square test of independence
  - Mann-Whitney U test
  - Wilcoxon signed-rank test
  - Kruskal-Wallis H test
  - Levene's test for equality of variances
  - Pearson / Spearman correlation tests

Each test returns a structured dict with: statistic, p_value, effect_size,
confidence_interval, conclusion, interpretation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.stats.hypothesis_tests")


def _result(
    test_name: str,
    statistic: float,
    p_value: float,
    effect_size: Optional[float],
    conclusion: str,
    interpretation: str,
    alpha: float,
    extra: Optional[Dict] = None,
) -> Dict[str, Any]:
    return {
        "test": test_name,
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 8),
        "alpha": alpha,
        "significant": bool(p_value < alpha),
        "effect_size": round(float(effect_size), 4) if effect_size is not None else None,
        "conclusion": conclusion,
        "interpretation": interpretation,
        **(extra or {}),
    }


class HypothesisTester:
    """
    Hypothesis testing engine.

    Usage::

        ht = HypothesisTester(alpha=0.05)
        result = ht.two_sample_t(group_a, group_b)
        print(result["conclusion"])
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ── t-tests ───────────────────────────────────────────────────────────────

    def one_sample_t(
        self, sample: pd.Series, popmean: float = 0.0, alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        """One-sample t-test against a known population mean."""
        arr = sample.dropna().values
        stat, p = scipy_stats.ttest_1samp(arr, popmean=popmean, alternative=alternative)
        d = (arr.mean() - popmean) / arr.std() if arr.std() > 0 else 0.0
        sig = p < self.alpha
        return _result(
            "one-sample-t", stat, p, d,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=(
                f"Mean ({arr.mean():.4f}) is significantly different from {popmean}"
                if sig else
                f"No significant difference from {popmean}"
            ),
            alpha=self.alpha,
            extra={"n": len(arr), "sample_mean": float(arr.mean()), "pop_mean": popmean, "cohen_d": round(d, 4)},
        )

    def two_sample_t(
        self,
        group_a: pd.Series,
        group_b: pd.Series,
        equal_var: bool = False,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """Welch's two-sample t-test (equal_var=False) or Student's (equal_var=True)."""
        a = group_a.dropna().values
        b = group_b.dropna().values
        stat, p = scipy_stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
        # Cohen's d (pooled std)
        pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
        effect_interp = "small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large")
        sig = p < self.alpha
        return _result(
            "two-sample-t (Welch's)" if not equal_var else "two-sample-t (Student's)",
            stat, p, d,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=(
                f"Groups are significantly different (d={d:.3f}, {effect_interp} effect)"
                if sig else
                "Groups are not significantly different"
            ),
            alpha=self.alpha,
            extra={"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "cohen_d": round(d, 4), "effect_size_label": effect_interp},
        )

    def paired_t(self, before: pd.Series, after: pd.Series) -> Dict[str, Any]:
        """Paired t-test."""
        df_pair = pd.DataFrame({"before": before, "after": after}).dropna()
        stat, p = scipy_stats.ttest_rel(df_pair["before"], df_pair["after"])
        diff = (df_pair["after"] - df_pair["before"]).values
        d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
        sig = p < self.alpha
        return _result(
            "paired-t", stat, p, d,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation="Significant before/after difference" if sig else "No significant before/after difference",
            alpha=self.alpha,
            extra={"n_pairs": len(df_pair), "mean_diff": float(diff.mean()), "cohen_d": round(d, 4)},
        )

    # ── ANOVA ────────────────────────────────────────────────────────────────

    def one_way_anova(self, groups: Dict[str, pd.Series]) -> Dict[str, Any]:
        """One-way ANOVA across multiple groups."""
        arrays = [g.dropna().values for g in groups.values()]
        stat, p = scipy_stats.f_oneway(*arrays)
        # Eta-squared effect size
        grand_mean = np.concatenate(arrays).mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
        ss_total = sum(((v - grand_mean) ** 2).sum() for v in arrays)
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        sig = p < self.alpha
        return _result(
            "one-way-ANOVA", stat, p, eta2,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=("At least one group mean is significantly different" if sig else "Group means are not significantly different"),
            alpha=self.alpha,
            extra={"groups": list(groups.keys()), "group_sizes": {k: len(g.dropna()) for k, g in groups.items()}, "eta_squared": round(eta2, 4)},
        )

    # ── Chi-square ───────────────────────────────────────────────────────────

    def chi_square(self, observed: pd.DataFrame) -> Dict[str, Any]:
        """Chi-square test of independence on a contingency table."""
        chi2, p, dof, expected = scipy_stats.chi2_contingency(observed)
        n = observed.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1))) if n > 0 else 0.0
        sig = p < self.alpha
        return _result(
            "chi-square", chi2, p, cramers_v,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=("Variables are significantly associated (dependent)" if sig else "Variables are independent"),
            alpha=self.alpha,
            extra={"dof": int(dof), "cramers_v": round(cramers_v, 4), "n": int(n)},
        )

    # ── Non-parametric ───────────────────────────────────────────────────────

    def mann_whitney_u(
        self, group_a: pd.Series, group_b: pd.Series, alternative: str = "two-sided"
    ) -> Dict[str, Any]:
        """Mann-Whitney U non-parametric two-sample test."""
        a, b = group_a.dropna().values, group_b.dropna().values
        stat, p = scipy_stats.mannwhitneyu(a, b, alternative=alternative)
        # Rank-biserial correlation as effect size
        r = 1 - (2 * stat) / (len(a) * len(b)) if len(a) * len(b) > 0 else 0.0
        sig = p < self.alpha
        return _result(
            "mann-whitney-u", stat, p, abs(r),
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=("Distributions differ significantly" if sig else "No significant difference"),
            alpha=self.alpha,
            extra={"n_a": len(a), "n_b": len(b), "rank_biserial_r": round(r, 4)},
        )

    def kruskal_wallis(self, groups: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Kruskal-Wallis H test (non-parametric one-way ANOVA)."""
        arrays = [g.dropna().values for g in groups.values()]
        stat, p = scipy_stats.kruskal(*arrays)
        sig = p < self.alpha
        return _result(
            "kruskal-wallis", stat, p, None,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=("At least one group distribution differs" if sig else "No significant distributional differences"),
            alpha=self.alpha,
            extra={"groups": list(groups.keys()), "n_groups": len(groups)},
        )

    def wilcoxon_signed_rank(self, before: pd.Series, after: pd.Series) -> Dict[str, Any]:
        """Wilcoxon signed-rank test (paired non-parametric)."""
        df_pair = pd.DataFrame({"before": before, "after": after}).dropna()
        stat, p = scipy_stats.wilcoxon(df_pair["before"], df_pair["after"])
        sig = p < self.alpha
        return _result(
            "wilcoxon-signed-rank", stat, p, None,
            conclusion="REJECT_H0" if sig else "FAIL_TO_REJECT_H0",
            interpretation=("Significant paired difference" if sig else "No significant paired difference"),
            alpha=self.alpha,
            extra={"n_pairs": len(df_pair)},
        )

    # ── Correlation ──────────────────────────────────────────────────────────

    def pearson_correlation(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Pearson correlation test."""
        df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
        r, p = scipy_stats.pearsonr(df_pair["x"], df_pair["y"])
        sig = p < self.alpha
        return _result(
            "pearson-correlation", r, p, abs(r),
            conclusion="SIGNIFICANT" if sig else "NOT_SIGNIFICANT",
            interpretation=f"r={r:.4f} ({'significant' if sig else 'not significant'})",
            alpha=self.alpha,
            extra={"n": len(df_pair), "pearson_r": round(r, 4)},
        )

    def spearman_correlation(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Spearman rank correlation test."""
        df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
        rho, p = scipy_stats.spearmanr(df_pair["x"], df_pair["y"])
        sig = p < self.alpha
        return _result(
            "spearman-correlation", rho, p, abs(rho),
            conclusion="SIGNIFICANT" if sig else "NOT_SIGNIFICANT",
            interpretation=f"ρ={rho:.4f} ({'significant' if sig else 'not significant'})",
            alpha=self.alpha,
            extra={"n": len(df_pair), "spearman_rho": round(rho, 4)},
        )
