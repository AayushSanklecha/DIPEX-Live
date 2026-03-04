"""
stats/permutation_tests.py
---------------------------
Permutation tests + multiple testing correction.

Permutation tests:
  - Two-sample mean difference
  - Correlation permutation test
  - ANOVA F-statistic permutation test
  - Custom statistic permutation test

Multiple testing correction:
  - Bonferroni
  - Holm-Bonferroni (step-down)
  - Benjamini-Hochberg (FDR)
  - Benjamini-Yekutieli (FDR under dependence)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.stats.permutation_tests")


class PermutationTester:
    """
    Non-parametric permutation testing engine.

    Usage::

        pt = PermutationTester(n_permutations=5000)
        result = pt.two_sample_mean(group_a, group_b)
        corrected = pt.correct_multiple([0.04, 0.03, 0.20, 0.001], method="fdr_bh")
    """

    def __init__(self, n_permutations: int = 5000, random_state: int = 42) -> None:
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_state)

    # ── Two-sample mean permutation ───────────────────────────────────────────

    def two_sample_mean(
        self,
        group_a: pd.Series,
        group_b: pd.Series,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Permutation test for difference in means."""
        a = group_a.dropna().values
        b = group_b.dropna().values
        observed_diff = float(np.mean(a) - np.mean(b))
        combined = np.concatenate([a, b])
        na = len(a)

        null_diffs = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm = self.rng.permutation(combined)
            null_diffs[i] = np.mean(perm[:na]) - np.mean(perm[na:])

        p_value = float(np.mean(np.abs(null_diffs) >= abs(observed_diff)))
        effect_size = observed_diff / combined.std() if combined.std() > 0 else 0.0

        return {
            "test": "permutation_two_sample_mean",
            "n_permutations": self.n_permutations,
            "observed_statistic": round(observed_diff, 6),
            "p_value": round(p_value, 6),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
            "effect_size": round(effect_size, 4),
            "conclusion": "REJECT_H0" if p_value < alpha else "FAIL_TO_REJECT_H0",
            "interpretation": (
                f"Mean difference ({observed_diff:.4f}) is statistically significant"
                if p_value < alpha else
                "No significant difference in means"
            ),
            "null_distribution_mean": round(float(null_diffs.mean()), 6),
            "null_distribution_std": round(float(null_diffs.std()), 6),
        }

    # ── Correlation permutation ───────────────────────────────────────────────

    def correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        method: str = "pearson",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Permutation test for correlation significance."""
        sub = pd.DataFrame({"x": x, "y": y}).dropna()
        x_arr, y_arr = sub["x"].values, sub["y"].values

        if method == "spearman":
            from scipy.stats import spearmanr
            observed_r, _ = spearmanr(x_arr, y_arr)
            stat_fn = lambda a, b: spearmanr(a, b)[0]  # noqa: E731
        else:
            from scipy.stats import pearsonr
            observed_r, _ = pearsonr(x_arr, y_arr)
            stat_fn = lambda a, b: pearsonr(a, b)[0]  # noqa: E731

        null_rs = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm_y = self.rng.permutation(y_arr)
            null_rs[i] = stat_fn(x_arr, perm_y)

        p_value = float(np.mean(np.abs(null_rs) >= abs(observed_r)))

        return {
            "test": f"permutation_correlation_{method}",
            "n_permutations": self.n_permutations,
            "observed_r": round(float(observed_r), 6),
            "p_value": round(p_value, 6),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
            "conclusion": "SIGNIFICANT" if p_value < alpha else "NOT_SIGNIFICANT",
        }

    # ── ANOVA permutation ─────────────────────────────────────────────────────

    def anova(
        self,
        groups: Dict[str, pd.Series],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Permutation test for one-way ANOVA F-statistic."""
        from scipy.stats import f_oneway

        arrays = [g.dropna().values for g in groups.values()]
        labels = np.concatenate([
            np.full(len(a), i) for i, a in enumerate(arrays)
        ])
        combined = np.concatenate(arrays)
        observed_f, _ = f_oneway(*arrays)

        null_fs = np.empty(self.n_permutations)
        group_sizes = [len(a) for a in arrays]
        for i in range(self.n_permutations):
            perm = self.rng.permutation(combined)
            permed_groups = []
            idx = 0
            for size in group_sizes:
                permed_groups.append(perm[idx:idx + size])
                idx += size
            try:
                null_fs[i] = f_oneway(*permed_groups)[0]
            except Exception:  # noqa: BLE001
                null_fs[i] = 0.0

        p_value = float(np.mean(null_fs >= observed_f))

        return {
            "test": "permutation_anova",
            "n_permutations": self.n_permutations,
            "groups": list(groups.keys()),
            "observed_F": round(float(observed_f), 6),
            "p_value": round(p_value, 6),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
            "conclusion": "REJECT_H0" if p_value < alpha else "FAIL_TO_REJECT_H0",
        }

    # ── Custom statistic ──────────────────────────────────────────────────────

    def custom(
        self,
        data: np.ndarray,
        stat_fn: Callable[[np.ndarray], float],
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """Permutation test for any custom scalar statistic."""
        observed = float(stat_fn(data))
        null_dist = np.array([stat_fn(self.rng.permutation(data)) for _ in range(self.n_permutations)])

        if alternative == "two-sided":
            p_value = float(np.mean(np.abs(null_dist) >= abs(observed)))
        elif alternative == "greater":
            p_value = float(np.mean(null_dist >= observed))
        else:
            p_value = float(np.mean(null_dist <= observed))

        return {
            "test": "permutation_custom",
            "n_permutations": self.n_permutations,
            "observed_statistic": round(observed, 6),
            "p_value": round(p_value, 6),
            "alpha": alpha,
            "significant": bool(p_value < alpha),
        }


# ── Multiple Testing Correction ───────────────────────────────────────────────

class MultipleTestingCorrector:
    """
    Multiple testing correction methods.

    Usage::

        mtc = MultipleTestingCorrector()
        result = mtc.correct([0.04, 0.001, 0.20, 0.03], method="fdr_bh", alpha=0.05)
    """

    def correct(
        self,
        p_values: List[float],
        method: str = "fdr_bh",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction.

        Methods
        -------
        bonferroni      — Conservative, controls FWER
        holm            — Holm-Bonferroni step-down, controls FWER
        fdr_bh          — Benjamini-Hochberg, controls FDR (recommended)
        fdr_by          — Benjamini-Yekutieli, FDR under dependence
        """
        p = np.array(p_values, dtype=float)
        m = len(p)

        methods: Dict[str, Any] = {}

        # Bonferroni
        bonf = np.minimum(p * m, 1.0)
        methods["bonferroni"] = {
            "adjusted_p": bonf.tolist(),
            "rejected": (bonf < alpha).tolist(),
            "n_rejected": int((bonf < alpha).sum()),
        }

        # Holm-Bonferroni
        order = np.argsort(p)
        holm_adj = np.zeros(m)
        running_max = 0.0
        for rank, idx in enumerate(order):
            val = p[idx] * (m - rank)
            running_max = max(running_max, val)
            holm_adj[idx] = min(running_max, 1.0)
        methods["holm"] = {
            "adjusted_p": holm_adj.tolist(),
            "rejected": (holm_adj < alpha).tolist(),
            "n_rejected": int((holm_adj < alpha).sum()),
        }

        # Benjamini-Hochberg (FDR)
        bh_adj = self._bh(p, alpha)
        methods["fdr_bh"] = {
            "adjusted_p": bh_adj.tolist(),
            "rejected": (bh_adj < alpha).tolist(),
            "n_rejected": int((bh_adj < alpha).sum()),
        }

        # Benjamini-Yekutieli
        cm = sum(1.0 / k for k in range(1, m + 1))
        by_adj = self._bh(p * cm, alpha)
        methods["fdr_by"] = {
            "adjusted_p": by_adj.tolist(),
            "rejected": (by_adj < alpha).tolist(),
            "n_rejected": int((by_adj < alpha).sum()),
        }

        selected = methods.get(method, methods["fdr_bh"])
        return {
            "n_tests": m,
            "alpha": alpha,
            "method": method,
            "raw_p_values": p.tolist(),
            "adjusted_p_values": selected["adjusted_p"],
            "rejected": selected["rejected"],
            "n_rejected": selected["n_rejected"],
            "all_methods_summary": {k: v["n_rejected"] for k, v in methods.items()},
        }

    @staticmethod
    def _bh(p: np.ndarray, alpha: float) -> np.ndarray:
        m = len(p)
        order = np.argsort(p)
        adj = np.zeros(m)
        running_min = 1.0
        for i in range(m - 1, -1, -1):
            idx = order[i]
            val = p[idx] * m / (i + 1)
            running_min = min(running_min, val)
            adj[idx] = running_min
        return np.minimum(adj, 1.0)
