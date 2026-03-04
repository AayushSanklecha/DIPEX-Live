"""
analyst/excel_engine.py
-------------------------
Excel Simulation Engine — replicates common spreadsheet operations as
deterministic, reproducible computation steps on Gold-layer DataFrames.

Simulated operations:
  1. pivot_table       — GROUP BY + aggregation (SUMIF-style)
  2. vlookup           — Label-based merge (LEFT JOIN on key)
  3. conditional_count — COUNTIF / COUNTIFS equivalent
  4. conditional_sum   — SUMIF / SUMIFS equivalent
  5. kpi_summary       — KPI card values (SUM, AVG, MIN, MAX, COUNT)
  6. percent_of_total  — Column/row percentage shares
  7. rank_percentile   — RANK / PERCENTRANK equivalent
  8. running_total     — Cumulative SUM
  9. yoy_change        — Year-over-year % change on time series
  10. basic_formulas   — IF / AND / OR / ISNULL / COALESCE equivalents
  11. highlight_rules  — Conditional formatting rules as boolean mask columns

All operations return a structured DataFrame — no actual Excel files are written.
Every output uses copy semantics (no mutation of source).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.analyst.excel_engine")


class ExcelEngine:
    """
    Pure-Python replica of spreadsheet analysis patterns.
    Each method is a self-contained, reproducible transformation.
    """

    # ── 1. Pivot Table ────────────────────────────────────────────────────────

    @staticmethod
    def pivot_table(
        df: pd.DataFrame,
        index: Union[str, List[str]],
        values: Union[str, List[str]],
        aggfunc: Union[str, Dict] = "sum",
        columns: Optional[str] = None,
        fill_value: float = 0.0,
        margins: bool = False,
    ) -> pd.DataFrame:
        """
        Equivalent to Excel PivotTable / pandas.pivot_table.
        Returns a flat DataFrame (reset_index applied automatically).
        """
        pt = pd.pivot_table(
            df, index=index, values=values, columns=columns,
            aggfunc=aggfunc, fill_value=fill_value, margins=margins,
        )
        return pt.reset_index()

    # ── 2. VLOOKUP / XLOOKUP ─────────────────────────────────────────────────

    @staticmethod
    def vlookup(
        df: pd.DataFrame,
        lookup_df: pd.DataFrame,
        lookup_col: str,
        return_cols: Optional[List[str]] = None,
        how: str = "left",
    ) -> pd.DataFrame:
        """
        Equivalent to Excel VLOOKUP / XLOOKUP.
        Merges df with lookup_df on lookup_col and returns specified columns.
        If return_cols is None, returns all columns from lookup_df.
        """
        return_cols = return_cols or [c for c in lookup_df.columns if c != lookup_col]
        merged = df.merge(lookup_df[[lookup_col] + return_cols], on=lookup_col, how=how)
        return merged.copy()

    # ── 3. COUNTIF / COUNTIFS ─────────────────────────────────────────────────

    @staticmethod
    def countif(
        df: pd.DataFrame,
        conditions: Dict[str, Any],
        result_col: str = "_countif",
    ) -> int:
        """
        Count rows matching all conditions (AND logic ~ COUNTIFS).
        Returns scalar count; also adds mask column to a copy for inspection.
        """
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in conditions.items():
            if col not in df.columns:
                continue
            if callable(val):
                mask &= val(df[col])
            elif isinstance(val, (list, tuple)):
                mask &= df[col].isin(val)
            else:
                mask &= df[col] == val
        return int(mask.sum())

    # ── 4. SUMIF / SUMIFS ─────────────────────────────────────────────────────

    @staticmethod
    def sumif(
        df: pd.DataFrame,
        conditions: Dict[str, Any],
        sum_col: str,
    ) -> float:
        """Equivalent to Excel SUMIFS — sum sum_col where all conditions hold."""
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in conditions.items():
            if col not in df.columns:
                continue
            mask &= df[col] == val if not callable(val) else val(df[col])
        return float(df.loc[mask, sum_col].sum()) if sum_col in df.columns else 0.0

    # ── 5. KPI Summary ────────────────────────────────────────────────────────

    @staticmethod
    def kpi_summary(
        df: pd.DataFrame,
        kpi_definitions: Optional[Dict[str, Dict]] = None,
    ) -> pd.DataFrame:
        """
        Compute KPI summary cards.
        kpi_definitions: {kpi_name: {"col": "revenue", "agg": "sum", "label": "Total Revenue"}}
        If not provided, generates summaries for all numeric columns.
        """
        rows = []
        if not kpi_definitions:
            for col in df.select_dtypes("number").columns:
                s = df[col].dropna()
                rows.append({
                    "kpi": col, "value": float(s.sum()),
                    "avg": round(float(s.mean()), 4),
                    "min": float(s.min()), "max": float(s.max()),
                    "count": len(s), "null_count": int(df[col].isnull().sum()),
                })
        else:
            for name, spec in kpi_definitions.items():
                col = spec.get("col", name)
                agg = spec.get("agg", "sum")
                if col not in df.columns:
                    rows.append({"kpi": name, "value": None, "error": f"Column '{col}' not found"})
                    continue
                s = df[col].dropna()
                fn_map = {"sum": s.sum, "mean": s.mean, "count": len,
                          "max": s.max, "min": s.min, "median": s.median}
                value = fn_map.get(agg, s.sum)()
                rows.append({
                    "kpi": name, "label": spec.get("label", name),
                    "column": col, "aggregation": agg,
                    "value": round(float(value), 4),
                })
        return pd.DataFrame(rows)

    # ── 6. Percent of Total ───────────────────────────────────────────────────

    @staticmethod
    def percent_of_total(
        df: pd.DataFrame,
        value_col: str,
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Equivalent to Excel % of total pivot field.
        If group_col provided, computes % within each group.
        """
        result = df.copy()
        if group_col and group_col in df.columns:
            result[f"{value_col}_pct_of_group"] = (
                result[value_col] / result.groupby(group_col)[value_col].transform("sum")
            ).round(4)
        total = df[value_col].sum()
        result[f"{value_col}_pct_of_total"] = (result[value_col] / (total + 1e-9)).round(4)
        return result

    # ── 7. RANK / PERCENTRANK ─────────────────────────────────────────────────

    @staticmethod
    def rank_percentile(
        df: pd.DataFrame,
        value_col: str,
        ascending: bool = False,
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Adds rank and percentile columns for value_col."""
        result = df.copy()
        method = "min"
        if group_col and group_col in df.columns:
            result[f"{value_col}_rank"] = (
                result.groupby(group_col)[value_col].rank(ascending=ascending, method=method)
            )
        else:
            result[f"{value_col}_rank"] = result[value_col].rank(
                ascending=ascending, method=method
            )
        result[f"{value_col}_percentile"] = (
            result[value_col].rank(pct=True, ascending=ascending).round(4)
        )
        return result

    # ── 8. Running Total (Cumulative SUM) ─────────────────────────────────────

    @staticmethod
    def running_total(
        df: pd.DataFrame,
        value_col: str,
        sort_col: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Adds cumulative sum column, optionally sorted and/or grouped."""
        result = df.copy()
        if sort_col and sort_col in df.columns:
            result = result.sort_values(sort_col)
        if group_col and group_col in df.columns:
            result[f"{value_col}_cumsum"] = (
                result.groupby(group_col)[value_col].cumsum()
            )
        else:
            result[f"{value_col}_cumsum"] = result[value_col].cumsum()
        return result

    # ── 9. Year-over-Year Change ──────────────────────────────────────────────

    @staticmethod
    def yoy_change(
        df: pd.DataFrame,
        date_col: str,
        value_col: str,
        period: str = "Y",
    ) -> pd.DataFrame:
        """Compute year-over-year (or period-over-period) % change."""
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
        result["_period"] = result[date_col].dt.to_period(period).astype(str)
        agg = result.groupby("_period")[value_col].sum().reset_index()
        agg[f"{value_col}_yoy_change_pct"] = agg[value_col].pct_change() * 100
        return agg

    # ── 10. Basic Formula Engine ──────────────────────────────────────────────

    @staticmethod
    def formula(
        df: pd.DataFrame,
        expressions: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Evaluate simple column expressions (Excel-like formulas).
        expressions: {"new_col": "revenue - cost", "margin": "revenue / sales"}
        Safe eval using pandas .eval().
        """
        result = df.copy()
        for col_name, expr in expressions.items():
            try:
                result[col_name] = result.eval(expr)
            except Exception as e:  # noqa: BLE001
                logger.warning("[ExcelEngine] Formula '%s'='%s' failed: %s", col_name, expr, e)
                result[col_name] = np.nan
        return result

    # ── 11. Conditional Formatting Rules ──────────────────────────────────────

    @staticmethod
    def highlight_rules(
        df: pd.DataFrame,
        rules: List[Dict],
    ) -> pd.DataFrame:
        """
        Apply conditional formatting rules as boolean indicator columns.

        rules: [{"col": "revenue", "op": ">", "value": 1000, "label": "high_revenue"}]
        Supported ops: >, >=, <, <=, ==, !=, between, isin, isnull, notnull
        """
        result = df.copy()
        op_map: Dict[str, Callable] = {
            ">":  lambda s, v: s > v,
            ">=": lambda s, v: s >= v,
            "<":  lambda s, v: s < v,
            "<=": lambda s, v: s <= v,
            "==": lambda s, v: s == v,
            "!=": lambda s, v: s != v,
            "between": lambda s, v: s.between(v[0], v[1]),
            "isin":    lambda s, v: s.isin(v),
            "isnull":  lambda s, v: s.isnull(),
            "notnull": lambda s, v: s.notnull(),
        }
        for rule in rules:
            col   = rule.get("col")
            op    = rule.get("op", ">")
            val   = rule.get("value")
            label = rule.get("label", f"{col}_{op}_{val}")
            if col and col in df.columns:
                fn = op_map.get(op, lambda s, v: pd.Series([False] * len(s), index=s.index))
                try:
                    result[f"_flag_{label}"] = fn(df[col], val).astype(bool)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[ExcelEngine] Rule '%s' error: %s", label, e)
        return result
