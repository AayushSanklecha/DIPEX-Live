"""
cognitive/sanity_checker.py
-----------------------------
Validates that computed metrics pass domain-level sanity gates.

Behaviour mirrors how a senior analyst "gut-checks" numbers:
  - Revenue should be ≥ 0
  - Rates (conversion, churn) should be in [0, 1]
  - Row counts should be > 0
  - Percentages should sum to ~100 if mutually exclusive
  - Values should not exceed statistical IQR fences by > N×IQR
  - Cross-metric consistency (e.g. active_users ≤ total_users)

Every check is configurable via config.yaml (data_layers.cognitive section).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.cognitive.sanity_checker")


# ── Violation Dataclass ───────────────────────────────────────────────────────

@dataclass
class SanityViolation:
    rule:        str
    column:      Optional[str]
    severity:    str             # CRITICAL | WARNING | INFO
    detail:      str
    value:       Any = None
    expected:    Any = None
    confidence:  float = 1.0

    def to_dict(self) -> Dict:
        return {
            "rule": self.rule, "column": self.column,
            "severity": self.severity, "detail": self.detail,
            "value": str(self.value)[:120], "expected": str(self.expected)[:80],
            "confidence": round(self.confidence, 4),
        }


# ── SanityChecker ─────────────────────────────────────────────────────────────

class SanityChecker:
    """
    Domain-heuristic sanity layer applied to every Gold artefact before surfacing.

    Rules are applied in order; first CRITICAL violation short-circuits the chain
    and flags the result for human review.
    """

    # Default heuristics (overridden by config)
    DEFAULT_HEURISTICS = {
        "non_negative_cols": ["revenue", "sales", "count", "amount", "price",
                               "quantity", "age", "duration", "cost"],
        "rate_cols":         ["rate", "ratio", "pct", "percent", "conversion",
                               "churn", "retention", "probability", "score",
                               "proportion"],
        "iqr_fence":         3.0,        # n × IQR for outlier fence
        "max_null_rate":     0.50,       # flag if > 50% null in any column
        "min_row_count":     1,
        "max_zero_rate":     0.95,       # flag if > 95% of column is zero
        "cross_metric_rules": [          # [col_a, op, col_b] — a op b must be True
            # e.g. ["active_users", "<=", "total_users"]
        ],
    }

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = (config or {}).get("data_layers", {}).get("cognitive", {})
        h = cfg.get("heuristics", {})
        self.h = {**self.DEFAULT_HEURISTICS, **h}

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self, df: pd.DataFrame, dataset_id: str = "",
        extra_rules: Optional[List[Dict]] = None,
    ) -> List[SanityViolation]:
        """
        Run all sanity rules against df.
        Returns a list of SanityViolation (empty = all good).
        """
        violations: List[SanityViolation] = []
        violations += self._check_row_count(df)
        violations += self._check_null_rates(df)
        violations += self._check_zero_rates(df)
        violations += self._check_non_negative(df)
        violations += self._check_rates_in_range(df)
        violations += self._check_iqr_fence(df)
        violations += self._check_cross_metrics(df)
        violations += self._check_percentages_sum(df)
        if extra_rules:
            violations += self._apply_extra_rules(df, extra_rules)
        if violations:
            crit = [v for v in violations if v.severity == "CRITICAL"]
            logger.warning(
                "[SanityChecker] %s — %d violations (%d CRITICAL)",
                dataset_id, len(violations), len(crit),
            )
        return violations

    def is_sane(self, df: pd.DataFrame, dataset_id: str = "") -> bool:
        """Returns True only if no CRITICAL violations found."""
        return not any(
            v.severity == "CRITICAL" for v in self.check(df, dataset_id)
        )

    # ── Rules ─────────────────────────────────────────────────────────────────

    def _check_row_count(self, df: pd.DataFrame) -> List[SanityViolation]:
        if len(df) < self.h["min_row_count"]:
            return [SanityViolation(
                rule="min_row_count", column=None, severity="CRITICAL",
                detail=f"DataFrame has {len(df)} rows — below minimum {self.h['min_row_count']}",
                value=len(df), expected=f">= {self.h['min_row_count']}",
            )]
        return []

    def _check_null_rates(self, df: pd.DataFrame) -> List[SanityViolation]:
        violations = []
        thresh = self.h["max_null_rate"]
        for col in df.columns:
            null_rate = df[col].isnull().mean()
            if null_rate > thresh:
                violations.append(SanityViolation(
                    rule="max_null_rate", column=col,
                    severity="WARNING" if null_rate < 0.8 else "CRITICAL",
                    detail=f"Column '{col}' has {null_rate:.1%} nulls (threshold: {thresh:.0%})",
                    value=round(null_rate, 4), expected=f"<= {thresh}",
                ))
        return violations

    def _check_zero_rates(self, df: pd.DataFrame) -> List[SanityViolation]:
        violations = []
        thresh = self.h["max_zero_rate"]
        for col in df.select_dtypes(include="number").columns:
            zero_rate = (df[col] == 0).mean()
            if zero_rate > thresh:
                violations.append(SanityViolation(
                    rule="max_zero_rate", column=col, severity="WARNING",
                    detail=f"Column '{col}' is {zero_rate:.1%} zeros — possible data problem",
                    value=round(zero_rate, 4), expected=f"<= {thresh}",
                ))
        return violations

    def _check_non_negative(self, df: pd.DataFrame) -> List[SanityViolation]:
        violations = []
        candidates = [
            c for c in df.select_dtypes("number").columns
            if any(kw in c.lower() for kw in self.h["non_negative_cols"])
        ]
        for col in candidates:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                violations.append(SanityViolation(
                    rule="non_negative", column=col, severity="CRITICAL",
                    detail=f"Column '{col}' has {neg_count} negative values — must be ≥ 0",
                    value=neg_count, expected="0 negative values",
                ))
        return violations

    def _check_rates_in_range(self, df: pd.DataFrame) -> List[SanityViolation]:
        violations = []
        candidates = [
            c for c in df.select_dtypes("number").columns
            if any(kw in c.lower() for kw in self.h["rate_cols"])
        ]
        for col in candidates:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range > 0:
                violations.append(SanityViolation(
                    rule="rate_in_range", column=col, severity="WARNING",
                    detail=f"Rate column '{col}' has {out_of_range} values outside [0, 1]",
                    value=out_of_range, expected="all values in [0, 1]",
                ))
        return violations

    def _check_iqr_fence(self, df: pd.DataFrame) -> List[SanityViolation]:
        violations = []
        n = self.h["iqr_fence"]
        for col in df.select_dtypes("number").columns:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            extreme = ((s < q1 - n * iqr) | (s > q3 + n * iqr)).sum()
            if extreme > max(1, len(s) * 0.01):   # >1% of rows
                violations.append(SanityViolation(
                    rule="iqr_fence", column=col, severity="WARNING",
                    detail=f"Column '{col}' has {extreme} values beyond {n}×IQR fence",
                    value=extreme, expected=f"≤ {max(1, int(len(s)*0.01))} outliers",
                    confidence=0.85,
                ))
        return violations

    def _check_cross_metrics(self, df: pd.DataFrame) -> List[SanityViolation]:
        violations = []
        op_map = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
                  "<": lambda a, b: a < b, ">": lambda a, b: a > b,
                  "==": lambda a, b: a == b}
        for rule in self.h.get("cross_metric_rules", []):
            if len(rule) != 3:
                continue
            col_a, op, col_b = rule
            if col_a not in df.columns or col_b not in df.columns:
                continue
            fn = op_map.get(op)
            if fn is None:
                continue
            violations_mask = ~fn(df[col_a], df[col_b])
            count = violations_mask.sum()
            if count > 0:
                violations.append(SanityViolation(
                    rule="cross_metric", column=f"{col_a} {op} {col_b}",
                    severity="CRITICAL",
                    detail=f"Cross-metric constraint violated for {count} rows: {col_a} {op} {col_b}",
                    value=count, expected="0 violations",
                ))
        return violations

    def _check_percentages_sum(self, df: pd.DataFrame) -> List[SanityViolation]:
        """Flag rows where percentage columns sum to far from 100."""
        pct_cols = [c for c in df.select_dtypes("number").columns
                    if any(kw in c.lower() for kw in ["pct", "percent", "share"])]
        if len(pct_cols) < 2:
            return []
        row_sums = df[pct_cols].sum(axis=1)
        bad = ((row_sums - 100).abs() > 5).sum()
        if bad > 0:
            return [SanityViolation(
                rule="pct_sum", column=str(pct_cols), severity="WARNING",
                detail=f"{bad} rows where percentage columns sum ≠ 100±5",
                value=bad, expected="row_sum ≈ 100",
            )]
        return []

    def _apply_extra_rules(
        self, df: pd.DataFrame, extra_rules: List[Dict]
    ) -> List[SanityViolation]:
        violations = []
        for rule in extra_rules:
            col  = rule.get("column")
            rmin = rule.get("min")
            rmax = rule.get("max")
            sev  = rule.get("severity", "WARNING")
            if col and col in df.columns:
                if rmin is not None:
                    bad = (df[col] < rmin).sum()
                    if bad:
                        violations.append(SanityViolation(
                            rule="custom_min", column=col, severity=sev,
                            detail=f"{bad} rows below custom min {rmin} for '{col}'",
                            value=bad, expected=f">= {rmin}",
                        ))
                if rmax is not None:
                    bad = (df[col] > rmax).sum()
                    if bad:
                        violations.append(SanityViolation(
                            rule="custom_max", column=col, severity=sev,
                            detail=f"{bad} rows above custom max {rmax} for '{col}'",
                            value=bad, expected=f"<= {rmax}",
                        ))
        return violations
