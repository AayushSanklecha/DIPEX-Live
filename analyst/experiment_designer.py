"""
analyst/experiment_designer.py
--------------------------------
Senior analyst — Experiment Design module.

Provides:
  - design_ab_test(): sample size, split ratio, duration estimate
  - calculate_power(): achieved power for given n, effect size, alpha
  - validate_experiment(): SUTVA, balance, novelty effect checklist
  - detect_leakage(): treatment contamination scan
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("dipex.experiment_designer")


from dataclasses import dataclass as _dc_ed, field as _fld_ed


@_dc_ed
class ExperimentDesign:
    """Result returned by ExperimentDesigner.design()."""
    name: str
    primary_metric: str
    baseline_rate: float
    mde: float
    alpha: float
    power: float
    n_per_group: int
    total_n: int
    runtime_days: Optional[int]
    daily_traffic: Optional[int]
    cohen_h: float
    warnings: list = _fld_ed(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "primary_metric": self.primary_metric,
            "n_per_group": self.n_per_group,
            "total_n": self.total_n,
            "runtime_days": self.runtime_days,
            "warnings": self.warnings,
        }


class ExperimentDesigner:
    """
    Statistically rigorous A/B test designer and validator.
    All calculations use standard frequentist power analysis.
    """

    def __init__(self, store_dir: Optional[str] = None) -> None:
        self.store_dir = Path(store_dir) if store_dir else None
        if self.store_dir:
            self.store_dir.mkdir(parents=True, exist_ok=True)

    def design(
        self,
        name: str,
        primary_metric: str,
        baseline_rate: float,
        mde: float = 0.05,
        alpha: float = 0.05,
        power: float = 0.80,
        daily_traffic: Optional[int] = None,
    ) -> ExperimentDesign:
        """
        Design an A/B test — convenience wrapper around design_ab_test().

        Args:
            name: Experiment name
            primary_metric: Metric column name
            baseline_rate: Current rate / mean (in [0,1] for proportions; larger values handled)
            mde: Minimum detectable effect (relative, e.g. 0.05 = 5% lift)
            alpha: Significance level
            power: Desired statistical power
            daily_traffic: Optional daily traffic for runtime estimate

        Returns:
            ExperimentDesign object
        """
        # Clamp baseline_rate to valid range for proportion test
        rate = max(0.001, min(0.999, float(baseline_rate)))
        result = self.design_ab_test(
            baseline_rate=rate,
            min_detectable_effect=mde,
            alpha=alpha,
            power=power,
            traffic_per_day=daily_traffic,
        )

        warnings = []
        n_per_group = result["sample_size_per_group"]
        if daily_traffic and daily_traffic > 0:
            runtime_days = result["duration_days"]
            if runtime_days and runtime_days > 90:
                warnings.append(f"Underpowered or slow: runtime estimate {runtime_days} days for daily traffic {daily_traffic}.")
        else:
            runtime_days = None

        achieved = self.calculate_power(
            n=n_per_group, baseline_rate=rate, min_detectable_effect=mde, alpha=alpha
        )
        if not achieved["is_adequately_powered"]:
            warnings.append(
                f"Underpowered: power={achieved['achieved_power']:.1%} with n={n_per_group}/group."
            )

        design = ExperimentDesign(
            name=name,
            primary_metric=primary_metric,
            baseline_rate=rate,
            mde=mde,
            alpha=alpha,
            power=power,
            n_per_group=n_per_group,
            total_n=n_per_group * 2,
            runtime_days=runtime_days,
            daily_traffic=daily_traffic,
            cohen_h=result["cohen_h"],
            warnings=warnings,
        )

        # Persist design as JSON
        if self.store_dir:
            import hashlib, uuid
            slug = hashlib.md5(name.encode()).hexdigest()[:8]
            out_path = self.store_dir / f"design_{slug}.json"
            import json as _json
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(design.to_dict(), f, indent=2)

        logger.info("ExperimentDesigner.design(): '%s' → n=%d/group", name, n_per_group)
        return design

    def validate_result(
        self,
        design: ExperimentDesign,
        control_data,
        treatment_data,
    ) -> Dict[str, Any]:
        """
        Validate an A/B test result using Welch t-test.

        Args:
            design: ExperimentDesign from .design()
            control_data: Series or array of control group metric values
            treatment_data: Series or array of treatment group metric values

        Returns:
            {p_value, significant, recommendation, ...}
        """
        import numpy as _np
        ctrl = _np.asarray(control_data, dtype=float)
        trt = _np.asarray(treatment_data, dtype=float)
        _, p_val = stats.ttest_ind(ctrl, trt, equal_var=False)
        significant = p_val < design.alpha
        lift = float((trt.mean() - ctrl.mean()) / ctrl.mean()) if ctrl.mean() != 0 else 0.0
        return {
            "p_value": round(float(p_val), 6),
            "significant": significant,
            "lift_pct": round(lift * 100, 2),
            "ctrl_mean": round(float(ctrl.mean()), 4),
            "treatment_mean": round(float(trt.mean()), 4),
            "recommendation": (
                f"Result is statistically significant at alpha={design.alpha} "
                f"(p={p_val:.4f}). Lift: {lift:.1%}. SHIP."
                if significant else
                f"Result is NOT significant (p={p_val:.4f}). Do NOT ship yet."
            ),
        }

    def design_ab_test(
        self,
        baseline_rate: float,
        min_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        traffic_per_day: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Design an A/B test with explicit statistical guarantees.

        Args:
            baseline_rate: Current conversion/metric rate (e.g., 0.05 for 5%)
            min_detectable_effect: Minimum relative effect to detect (e.g., 0.10 for 10% lift)
            alpha: Type I error rate (default 5%)
            power: Statistical power (default 80%)
            traffic_per_day: Daily traffic units (for duration estimate)

        Returns:
            {sample_size_per_group, total_sample, split_ratio, duration_days,
             z_alpha, z_beta, effect_size_absolute}
        """
        if not 0 < baseline_rate < 1:
            raise ValueError("baseline_rate must be in (0, 1)")
        if not 0 < min_detectable_effect:
            raise ValueError("min_detectable_effect must be > 0")

        variant_rate = baseline_rate * (1 + min_detectable_effect)
        variant_rate = min(variant_rate, 0.9999)

        # Cohen's h for proportions
        h = 2 * (math.asin(math.sqrt(variant_rate)) - math.asin(math.sqrt(baseline_rate)))
        effect_size_h = abs(h)

        # Power analysis z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-tailed
        z_beta = stats.norm.ppf(power)

        # Sample size per group (standard formula for proportions)
        pooled_p = (baseline_rate + variant_rate) / 2
        n_per_group = math.ceil(
            (z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p)) +
             z_beta * math.sqrt(baseline_rate * (1 - baseline_rate) +
                                variant_rate * (1 - variant_rate))) ** 2
            / (min_detectable_effect * baseline_rate) ** 2
        )

        total_n = n_per_group * 2
        duration_days = None
        if traffic_per_day and traffic_per_day > 0:
            duration_days = math.ceil(total_n / traffic_per_day)

        return {
            "baseline_rate": baseline_rate,
            "variant_rate": round(variant_rate, 4),
            "min_detectable_effect_relative": min_detectable_effect,
            "effect_size_absolute": round(variant_rate - baseline_rate, 4),
            "alpha": alpha,
            "power": power,
            "z_alpha": round(z_alpha, 4),
            "z_beta": round(z_beta, 4),
            "sample_size_per_group": n_per_group,
            "total_sample_required": total_n,
            "split_ratio": "50/50 (control/treatment)",
            "duration_days": duration_days,
            "cohen_h": round(effect_size_h, 4),
            "notes": (
                f"To detect a {min_detectable_effect:.1%} relative lift with "
                f"{alpha:.0%} significance and {power:.0%} power, "
                f"you need {n_per_group:,} users per group."
            ),
        }

    def calculate_power(
        self,
        n: int,
        baseline_rate: float,
        min_detectable_effect: float,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Calculate achieved statistical power for given sample size.

        Returns:
            {achieved_power, is_adequately_powered, n, effect_size}
        """
        variant_rate = baseline_rate * (1 + min_detectable_effect)
        variant_rate = min(variant_rate, 0.9999)
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        se = math.sqrt(
            (baseline_rate * (1 - baseline_rate) + variant_rate * (1 - variant_rate)) / n
        )
        z_test = abs(variant_rate - baseline_rate) / se if se > 0 else 0.0
        z_power = z_test - z_alpha
        achieved_power = float(stats.norm.cdf(z_power))

        return {
            "n_per_group": n,
            "baseline_rate": baseline_rate,
            "variant_rate": round(variant_rate, 4),
            "effect_size_absolute": round(variant_rate - baseline_rate, 4),
            "alpha": alpha,
            "achieved_power": round(achieved_power, 4),
            "is_adequately_powered": achieved_power >= 0.80,
            "recommendation": (
                "Adequate power ✓" if achieved_power >= 0.80
                else f"Underpowered — need larger sample. Current power: {achieved_power:.1%}"
            ),
        }

    def validate_experiment(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        date_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate an experiment for statistical and design integrity.
        Checks: SUTVA, group balance, novelty effect, metric distribution.

        Args:
            df: Experiment DataFrame (one row per unit)
            group_col: Column identifying control/treatment (0/1 or "A"/"B")
            metric_col: Primary outcome metric column
            date_col: Optional date column for novelty effect check

        Returns:
            Validation checklist with pass/fail for each check
        """
        checks: List[Dict[str, Any]] = []

        # 1. SUTVA: unique units (no row per user-session if should be user-level)
        total = len(df)
        unique_per_group = df.groupby(group_col).size().to_dict()
        checks.append({
            "check": "SUTVA — independent units",
            "passed": df.duplicated([group_col, metric_col]).sum() == 0,
            "detail": f"Total rows: {total}. Groups: {unique_per_group}. "
                      "Manual verification required for SUTVA (no interference).",
        })

        # 2. Group balance
        groups = df[group_col].unique()
        n_groups = len(groups)
        if n_groups >= 2:
            counts = [len(df[df[group_col] == g]) for g in groups]
            ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
            balanced = ratio < 1.2
            checks.append({
                "check": "Group balance",
                "passed": balanced,
                "detail": f"Group ratio (max/min): {ratio:.2f} — {'balanced ✓' if balanced else 'imbalanced ✗'}",
            })

        # 3. Metric distribution (normality check for large samples)
        if metric_col in df.columns:
            series = df[metric_col].dropna()
            if len(series) > 30:
                _, p_val = stats.shapiro(series[:5000]) if len(series) <= 5000 else (None, 0.01)
                normal_ish = p_val is None or p_val > 0.01
                checks.append({
                    "check": "Metric distribution",
                    "passed": True,  # We note but don't fail non-normal distributions
                    "detail": (
                        f"Metric '{metric_col}': {'approximately normal' if normal_ish else 'non-normal'} "
                        f"(Shapiro p={p_val:.4f if p_val else 'N/A'}). "
                        "Use non-parametric tests for non-normal distributions."
                    ),
                })

        # 4. Novelty effect (early-period inflated engagement)
        if date_col and date_col in df.columns:
            df_dated = df.copy()
            df_dated[date_col] = pd.to_datetime(df_dated[date_col], errors="coerce", utc=True)
            date_range = df_dated[date_col].max() - df_dated[date_col].min()
            novelty_risk = date_range.days < 7 if hasattr(date_range, "days") else False
            checks.append({
                "check": "Novelty effect risk",
                "passed": not novelty_risk,
                "detail": (
                    f"Experiment ran for ~{date_range.days} days. "
                    f"{'Short duration — novelty effect risk. ✗' if novelty_risk else 'Adequate duration ✓'}"
                ),
            })

        all_passed = all(c["passed"] for c in checks)
        return {
            "overall_valid": all_passed,
            "checks": checks,
            "recommendation": (
                "Experiment design is valid." if all_passed
                else "One or more design checks failed. Review before launching."
            ),
        }

    def detect_leakage(
        self,
        df: pd.DataFrame,
        group_col: str,
        pre_treatment_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Scan for treatment leakage: pre-treatment columns should not differ
        significantly between groups.

        Performs Welch t-test for numeric columns;
        Chi-square for categorical columns.

        Returns:
            {detected: bool, leaky_columns: [...], test_results: [...]}
        """
        test_results = []
        leaky_cols = []

        groups = df[group_col].unique()
        if len(groups) < 2:
            return {"detected": False, "leaky_columns": [], "test_results": [],
                    "note": "Need at least 2 groups"}

        g1, g2 = groups[0], groups[1]
        g1_df = df[df[group_col] == g1]
        g2_df = df[df[group_col] == g2]

        for col in pre_treatment_cols:
            if col not in df.columns:
                continue
            series1 = g1_df[col].dropna()
            series2 = g2_df[col].dropna()
            if len(series1) < 5 or len(series2) < 5:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                _, p_val = stats.ttest_ind(series1, series2, equal_var=False)
                leaky = p_val < 0.05
                test_results.append({
                    "column": col,
                    "test": "Welch t-test",
                    "p_value": round(float(p_val), 6),
                    "leakage_detected": leaky,
                })
            else:
                ct = pd.crosstab(df[group_col], df[col])
                if ct.shape[1] > 1:
                    _, p_val, _, _ = stats.chi2_contingency(ct)
                    leaky = p_val < 0.05
                    test_results.append({
                        "column": col,
                        "test": "Chi-square",
                        "p_value": round(float(p_val), 6),
                        "leakage_detected": leaky,
                    })
                else:
                    leaky = False

            if leaky:
                leaky_cols.append(col)

        return {
            "detected": len(leaky_cols) > 0,
            "leaky_columns": leaky_cols,
            "test_results": test_results,
            "recommendation": (
                f"⚠️ Leakage detected in {len(leaky_cols)} columns: {', '.join(leaky_cols)}. "
                "Treatment groups differ significantly on pre-treatment features."
                if leaky_cols else "✓ No leakage detected — groups are balanced on pre-treatment features."
            ),
        }
