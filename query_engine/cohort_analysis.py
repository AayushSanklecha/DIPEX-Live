"""
query_engine/cohort_analysis.py
---------------------------------
Enterprise cohort analysis engine backed by DuckDB.

Analyses:
  - Cohort Retention Matrix  (classic monthly/weekly cohort table)
  - Nth-period retention rate
  - LTV (Lifetime Value) cohort curves
  - Cohort size distribution

Terminology:
  - cohort_col   : column identifying when the user/entity joined (e.g. signup_month)
  - entity_col   : column identifying the unique entity (e.g. user_id)
  - activity_col : column identifying the activity date (e.g. purchase_date)
  - value_col    : optional metric to aggregate (e.g. revenue)

Usage::

    ca = CohortAnalyzer()
    result = ca.retention_matrix(df,
        cohort_col="signup_month",
        entity_col="user_id",
        activity_col="activity_month",
    )
    print(result["matrix"])           # {cohort: {period_0: n, period_1: n, ...}}
    print(result["retention_rates"])  # same but as % of cohort_size
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("dipex.query_engine.cohort")


class CohortAnalyzer:
    """
    SQL-backed cohort analysis using DuckDB.
    Falls back to pandas-native if DuckDB unavailable.
    """

    def retention_matrix(
        self,
        df: pd.DataFrame,
        cohort_col: str,
        entity_col: str,
        activity_col: str,
        max_periods: int = 12,
    ) -> Dict[str, Any]:
        """
        Compute cohort retention matrix.

        Returns a dict with:
          - matrix        : {cohort → {period_0: count, period_1: count, ...}}
          - retention_rates: same but as % of initial cohort size
          - cohort_sizes  : {cohort → n_entities}
          - period_avg_retention: average retention rate per period across cohorts
        """
        if cohort_col not in df.columns or entity_col not in df.columns or activity_col not in df.columns:
            return {"error": f"Required columns missing. Need: {cohort_col}, {entity_col}, {activity_col}"}

        df = df[[cohort_col, entity_col, activity_col]].dropna().copy()

        # Convert to period strings for grouping
        for col in [cohort_col, activity_col]:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.to_period("M").astype(str)
            else:
                df[col] = df[col].astype(str)

        # Period index = ordinal distance from cohort start
        cohort_min = df.groupby(entity_col)[cohort_col].min().rename("entity_cohort")
        df = df.merge(cohort_min, on=entity_col, how="left")

        # Build sorted period list
        all_periods = sorted(df[cohort_col].unique())
        period_idx = {p: i for i, p in enumerate(all_periods)}

        df["cohort_period"] = df[activity_col].map(period_idx) - df["entity_cohort"].map(period_idx)
        df = df[df["cohort_period"] >= 0]
        df = df[df["cohort_period"] <= max_periods]

        # Cohort sizes
        cohort_sizes_ser = (
            df[df["cohort_period"] == 0]
            .groupby("entity_cohort")[entity_col].nunique()
        )
        cohort_sizes = cohort_sizes_ser.to_dict()

        # Retention counts
        retention = (
            df.groupby(["entity_cohort", "cohort_period"])[entity_col]
            .nunique()
            .reset_index()
            .rename(columns={entity_col: "n_active", "entity_cohort": "cohort"})
        )

        # Build matrix
        matrix: Dict[str, Dict] = {}
        rates_matrix: Dict[str, Dict] = {}

        for cohort in sorted(cohort_sizes.keys()):
            base = cohort_sizes.get(cohort, 1) or 1
            sub = retention[retention["cohort"] == cohort].set_index("cohort_period")["n_active"]
            matrix[cohort] = {}
            rates_matrix[cohort] = {}
            for p in range(max_periods + 1):
                n = int(sub.get(p, 0))
                matrix[cohort][f"period_{p}"] = n
                rates_matrix[cohort][f"period_{p}"] = round(n / base * 100, 1)

        # Average retention per period
        period_avgs: Dict[str, float] = {}
        for p in range(max_periods + 1):
            key = f"period_{p}"
            vals = [v[key] for v in rates_matrix.values() if v.get(key, 0) > 0]
            period_avgs[key] = round(float(np.mean(vals)), 1) if vals else 0.0

        return {
            "cohort_col": cohort_col,
            "entity_col": entity_col,
            "activity_col": activity_col,
            "n_cohorts": len(cohort_sizes),
            "max_periods": max_periods,
            "cohort_sizes": cohort_sizes,
            "matrix": matrix,
            "retention_rates": rates_matrix,
            "period_avg_retention": period_avgs,
        }

    def ltv_cohorts(
        self,
        df: pd.DataFrame,
        cohort_col: str,
        entity_col: str,
        activity_col: str,
        value_col: str,
        max_periods: int = 12,
    ) -> Dict[str, Any]:
        """
        Compute cumulative LTV (Lifetime Value) cohort curves.

        Returns cumulative sum of `value_col` per cohort per period.
        """
        required = [cohort_col, entity_col, activity_col, value_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"error": f"Missing columns: {missing}"}

        df = df[required].dropna().copy()

        for col in [cohort_col, activity_col]:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.to_period("M").astype(str)
            else:
                df[col] = df[col].astype(str)

        cohort_min = df.groupby(entity_col)[cohort_col].min().rename("entity_cohort")
        df = df.merge(cohort_min, on=entity_col, how="left")

        all_periods = sorted(df[cohort_col].unique())
        period_idx = {p: i for i, p in enumerate(all_periods)}

        df["cohort_period"] = (
            df[activity_col].map(period_idx) - df["entity_cohort"].map(period_idx)
        )
        df = df[(df["cohort_period"] >= 0) & (df["cohort_period"] <= max_periods)]

        ltv_agg = (
            df.groupby(["entity_cohort", "cohort_period"])[value_col]
            .sum()
            .reset_index()
            .rename(columns={"entity_cohort": "cohort", value_col: "revenue"})
        )

        ltv_matrix: Dict[str, Dict] = {}
        for cohort in ltv_agg["cohort"].unique():
            sub = ltv_agg[ltv_agg["cohort"] == cohort].set_index("cohort_period")["revenue"]
            cumulative = 0.0
            ltv_matrix[cohort] = {}
            for p in range(max_periods + 1):
                cumulative += float(sub.get(p, 0))
                ltv_matrix[cohort][f"period_{p}"] = round(cumulative, 2)

        return {
            "cohort_col": cohort_col,
            "entity_col": entity_col,
            "value_col": value_col,
            "n_cohorts": len(ltv_matrix),
            "ltv_matrix": ltv_matrix,
        }

    def summary_stats(self, retention_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary stats from a retention_matrix result."""
        rates = retention_result.get("retention_rates", {})
        period_avg = retention_result.get("period_avg_retention", {})

        day_1_ret = period_avg.get("period_1", 0.0)
        day_7_ret = period_avg.get("period_7", 0.0)

        # Best/worst cohort by 3-month retention
        period_3 = {k: v.get("period_3", 0) for k, v in rates.items()}
        best_cohort = max(period_3, key=period_3.get) if period_3 else None
        worst_cohort = min(period_3, key=period_3.get) if period_3 else None

        return {
            "avg_period_1_retention": day_1_ret,
            "avg_period_7_retention": day_7_ret,
            "best_cohort_period_3": best_cohort,
            "worst_cohort_period_3": worst_cohort,
            "n_cohorts": retention_result.get("n_cohorts", 0),
        }
