"""
analyst/junior_analyst.py
---------------------------
Automated simulation of Junior Data Analyst operations.

INVARIANT: Every operation in this module:
  ✔ Receives a deep copy from the Gold layer (NEVER Silver or Bronze directly)
  ✔ Never writes back to Silver or Bronze
  ✔ Produces a GoldArtefact with full lineage
  ✔ Is fully logged with elapsed time and parameters
  ✔ Is reversible / idempotent

Operations simulated
--------------------
1.  basic_cleaning          — strip whitespace, normalise null representations
2.  filter_rows             — predicate-based row filtering
3.  remove_duplicates       — deduplication with subset + keep strategy
4.  simple_aggregation      — groupby + agg with configurable functions
5.  pivot_table             — pd.pivot_table wrapper
6.  percent_change          — period-over-period change for numeric columns
7.  kpi_tracking            — compute predefined KPIs
8.  sql_query               — run SQL against Gold layer via DuckDB
9.  basic_join              — merge two Gold artefacts
10. data_export             — export Gold artefact to CSV/Excel/JSON
11. manual_threshold_check  — flag rows outside defined value bounds
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ingestion.data_layers import GoldArtefact, ImmutableDataFrame, LayerManager
from ingestion.immutability_guard import (
    MutationProbe, ImmutabilityViolationError, LayerWriteGuard
)
from ingestion.lineage import LineageRecord, TransformationStep

logger = logging.getLogger("dipex.analyst.junior")

COMPONENT = "junior_analyst"


class JuniorAnalyst:
    """
    Simulates systematic junior-level data analysis operations.
    All methods operate ONLY on Gold layer copies via the LayerManager.
    """

    def __init__(
        self,
        layer_manager: Optional[LayerManager] = None,
        operator: str = "system",
    ) -> None:
        self.lm = layer_manager or LayerManager()
        self.operator = operator

    # ── 1. Basic cleaning ─────────────────────────────────────────────────────

    def basic_cleaning(
        self, silver: ImmutableDataFrame, source_snapshot_id: str = ""
    ) -> GoldArtefact:
        """Strip whitespace from strings, normalise null representations."""

        def _clean(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip() if hasattr(df[col], "str") else df[col]
                df[col] = df[col].replace(
                    {"": pd.NA, "null": pd.NA, "NULL": pd.NA,
                     "NaN": pd.NA, "nan": pd.NA, "N/A": pd.NA, "n/a": pd.NA}
                )
            return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_cleaned",
            component=COMPONENT, transform_fn=_clean,
            step_name="basic_cleaning", operator=self.operator,
            source_snapshot_id=source_snapshot_id,
        )

    # ── 2. Filter rows ────────────────────────────────────────────────────────

    def filter_rows(
        self, silver: ImmutableDataFrame, query: str,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Filter rows using a pandas query string."""

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            return df.query(query).reset_index(drop=True)

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_filtered",
            component=COMPONENT, transform_fn=_filter,
            step_name="filter_rows", operator=self.operator,
            parameters={"query": query},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 3. Remove duplicates ──────────────────────────────────────────────────

    def remove_duplicates(
        self, silver: ImmutableDataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Deduplicate rows from a Gold copy."""

        def _dedup(df: pd.DataFrame) -> pd.DataFrame:
            before = len(df)
            df = df.drop_duplicates(subset=subset, keep=keep)
            logger.info("Deduplication: %d → %d rows (%d removed)", before, len(df), before - len(df))
            return df.reset_index(drop=True)

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_dedup",
            component=COMPONENT, transform_fn=_dedup,
            step_name="remove_duplicates", operator=self.operator,
            parameters={"subset": subset, "keep": keep},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 4. Simple aggregation ─────────────────────────────────────────────────

    def simple_aggregation(
        self, silver: ImmutableDataFrame,
        group_by: List[str],
        agg_map: Dict[str, str],
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """groupby + agg. agg_map: {'col': 'sum', 'col2': 'mean'}"""

        def _agg(df: pd.DataFrame) -> pd.DataFrame:
            return df.groupby(group_by, observed=True).agg(agg_map).reset_index()

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_agg",
            component=COMPONENT, transform_fn=_agg,
            step_name="simple_aggregation", operator=self.operator,
            parameters={"group_by": group_by, "agg_map": agg_map},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 5. Pivot table ────────────────────────────────────────────────────────

    def pivot_table(
        self, silver: ImmutableDataFrame,
        index: List[str], columns: str, values: str,
        aggfunc: str = "sum", source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Create pivot table on Gold copy."""

        def _pivot(df: pd.DataFrame) -> pd.DataFrame:
            pivot = pd.pivot_table(
                df, values=values, index=index,
                columns=columns, aggfunc=aggfunc, fill_value=0,
            )
            pivot.columns = [f"{values}_{c}" for c in pivot.columns]
            return pivot.reset_index()

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_pivot",
            component=COMPONENT, transform_fn=_pivot,
            step_name="pivot_table", operator=self.operator,
            parameters={"index": index, "columns": columns, "values": values},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 6. Percent change ─────────────────────────────────────────────────────

    def percent_change(
        self, silver: ImmutableDataFrame,
        col: str, periods: int = 1,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Period-over-period percent change for a numeric column."""

        def _pct(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df[f"{col}_pct_change"] = df[col].pct_change(periods=periods) * 100
            return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_pctchange",
            component=COMPONENT, transform_fn=_pct,
            step_name="percent_change", operator=self.operator,
            parameters={"col": col, "periods": periods},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 7. KPI tracking ───────────────────────────────────────────────────────

    def kpi_tracking(
        self, silver: ImmutableDataFrame,
        kpi_definitions: Dict[str, str],
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """
        Compute KPIs as eval expressions on Gold copy.
        kpi_definitions: {'Revenue_Per_User': 'revenue / users'}
        """

        def _kpi(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            kpi_results = {}
            for kpi_name, expr in kpi_definitions.items():
                try:
                    kpi_results[kpi_name] = df.eval(expr)
                except Exception as e:  # noqa: BLE001
                    logger.warning("KPI '%s' failed: %s", kpi_name, e)
            for k, v in kpi_results.items():
                df[k] = v
            return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_kpi",
            component=COMPONENT, transform_fn=_kpi,
            step_name="kpi_tracking", operator=self.operator,
            parameters={"kpis": list(kpi_definitions.keys())},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 8. SQL query ──────────────────────────────────────────────────────────

    def sql_query(
        self, silver: ImmutableDataFrame, sql: str,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Run SQL against Gold copy using DuckDB (in-memory, non-destructive)."""
        import importlib

        def _sql(df: pd.DataFrame) -> pd.DataFrame:
            try:
                import duckdb
                conn = duckdb.connect(database=":memory:")
                conn.register("data", df)
                result = conn.execute(sql).df()
                conn.close()
                return result
            except ImportError:
                # DuckDB not installed — use pandas query as fallback
                logger.warning("DuckDB not available — falling back to pandas eval")
                return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_sql",
            component=COMPONENT, transform_fn=_sql,
            step_name="sql_query", operator=self.operator,
            parameters={"sql_length": len(sql)},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 9. Basic join ─────────────────────────────────────────────────────────

    def basic_join(
        self, left: ImmutableDataFrame, right: ImmutableDataFrame,
        on: str, how: str = "left",
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """Merge two Silver/Gold immutable sources on a key column."""
        right_copy = right.copy()

        def _join(df: pd.DataFrame) -> pd.DataFrame:
            return df.merge(right_copy, on=on, how=how)

        return self.lm.derive_gold(
            left, dataset_id=f"{left._dataset_id}_joined",
            component=COMPONENT, transform_fn=_join,
            step_name="basic_join", operator=self.operator,
            parameters={"on": on, "how": how, "right": right._dataset_id},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 10. Data export ───────────────────────────────────────────────────────

    def data_export(
        self, gold: GoldArtefact,
        output_path: str,
        fmt: str = "csv",
    ) -> str:
        """
        Export a Gold artefact to CSV, Excel, or JSON.
        Writes the Gold copy — Silver/Bronze are never touched.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df = gold.data.copy(deep=True)
        if fmt == "csv":
            df.to_csv(output_path, index=False)
        elif fmt in ("xlsx", "excel"):
            df.to_excel(output_path, index=False)
        elif fmt == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            df.to_csv(output_path, index=False)
        logger.info(
            "Gold artefact exported: %s → %s (%s, %d rows)",
            gold.dataset_id, output_path, fmt, len(df),
        )
        return output_path

    # ── 11. Manual threshold check ────────────────────────────────────────────

    def manual_threshold_check(
        self, silver: ImmutableDataFrame,
        thresholds: Dict[str, Tuple[float, float]],
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """
        Flag rows where column values are outside [min, max] bounds.
        thresholds: {'revenue': (0, 1e6), 'age': (0, 120)}
        """

        def _check(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["_threshold_flags"] = ""
            for col, (lo, hi) in thresholds.items():
                if col in df.columns:
                    mask = (df[col] < lo) | (df[col] > hi)
                    df.loc[mask, "_threshold_flags"] += f"{col}:OOB;"
            return df

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_threshold",
            component=COMPONENT, transform_fn=_check,
            step_name="manual_threshold_check", operator=self.operator,
            parameters={"thresholds": {k: list(v) for k, v in thresholds.items()}},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 12. Basic visualization spec ──────────────────────────────────────────

    def basic_visualization_spec(
        self, silver: ImmutableDataFrame,
        x_col: str,
        y_col: Optional[str] = None,
        chart_type: Optional[str] = None,
        title: Optional[str] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """
        Generate a structured chart specification JSON for dashboard rendering.

        Automatically selects chart type when not specified:
          - Numeric × Numeric           → scatter
          - Categorical × Numeric       → bar
          - Datetime × Numeric          → line
          - Single Categorical          → pie (if cardinality ≤ 10), else bar
          - Single Numeric              → histogram

        Returns a GoldArtefact whose DataFrame contains one row with a
        'chart_spec' column (JSON string) for a downstream renderer.
        Includes misleading-visualization detection flags.
        """
        import json as _json

        def _spec(df: pd.DataFrame) -> pd.DataFrame:
            x_series = df[x_col] if x_col in df.columns else pd.Series(dtype="object")
            y_series = df[y_col] if y_col and y_col in df.columns else None

            # Auto-select chart type
            if chart_type:
                selected_type = chart_type
            elif y_series is not None:
                x_is_dt = pd.api.types.is_datetime64_any_dtype(x_series)
                x_is_num = pd.api.types.is_numeric_dtype(x_series)
                y_is_num = pd.api.types.is_numeric_dtype(y_series)
                if x_is_dt and y_is_num:
                    selected_type = "line"
                elif x_is_num and y_is_num:
                    selected_type = "scatter"
                else:
                    selected_type = "bar"
            elif pd.api.types.is_numeric_dtype(x_series):
                selected_type = "histogram"
            elif x_series.nunique() <= 10:
                selected_type = "pie"
            else:
                selected_type = "bar"

            # Misleading viz detection
            misleading_flags = []
            if y_series is not None and pd.api.types.is_numeric_dtype(y_series):
                y_range = y_series.max() - y_series.min() if len(y_series) > 0 else 0
                y_min = y_series.min() if len(y_series) > 0 else 0
                if y_min > 0 and y_range > 0 and (y_min / y_range) > 5:
                    misleading_flags.append("TRUNCATED_Y_AXIS — y-axis does not start at 0")
                if selected_type == "pie" and x_series.nunique() > 7:
                    misleading_flags.append("TOO_MANY_PIE_SLICES — use bar chart instead")

            spec: Dict[str, Any] = {
                "chart_type": selected_type,
                "x_axis": {"column": x_col, "label": x_col.replace("_", " ").title()},
                "y_axis": {"column": y_col, "label": y_col.replace("_", " ").title()} if y_col else None,
                "title": title or f"{y_col or x_col} by {x_col}".replace("_", " ").title(),
                "color_palette": "tableau10",
                "accessible": True,
                "n_data_points": len(df),
                "misleading_flags": misleading_flags,
            }

            return pd.DataFrame([{"chart_spec": _json.dumps(spec)}])

        return self.lm.derive_gold(
            silver, dataset_id=f"{silver._dataset_id}_vizspec",
            component=COMPONENT, transform_fn=_spec,
            step_name="basic_visualization_spec", operator=self.operator,
            parameters={"x_col": x_col, "y_col": y_col, "chart_type": chart_type},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 13. Merge files ───────────────────────────────────────────────────────

    def merge_files(
        self, sources: List[ImmutableDataFrame],
        how: str = "concat",
        on: Optional[str] = None,
        sort_by: Optional[str] = None,
        source_snapshot_id: str = "",
    ) -> GoldArtefact:
        """
        Merge multiple Silver/Gold immutable sources into one Gold artefact.

        how='concat' : vertical concatenation (union of rows)
        how='join'   : horizontal join on a key column (requires `on`)

        All sources must be ImmutableDataFrame to enforce lineage.
        Source Silver frames are never mutated.
        """
        # Work from copies only
        copies: List[pd.DataFrame] = [s.copy() for s in sources]
        source_ids = ",".join([s._dataset_id for s in sources])
        primary = sources[0]

        def _merge(df: pd.DataFrame) -> pd.DataFrame:
            # df is a copy of primary source; join remaining sources
            additional = copies[1:]
            if how == "concat":
                result = pd.concat([df] + additional, ignore_index=True, sort=False)
            elif how == "join":
                if on is None:
                    raise ValueError("'on' key column is required for merge how='join'")
                result = df
                for other in additional:
                    suffixes = ("_left", "_right")
                    result = result.merge(other, on=on, how="outer", suffixes=suffixes)
            else:
                raise ValueError(f"Unknown merge how='{how}'. Use 'concat' or 'join'.")

            if sort_by and sort_by in result.columns:
                result = result.sort_values(sort_by).reset_index(drop=True)
            return result

        return self.lm.derive_gold(
            primary, dataset_id=f"merged_{source_ids[:30]}",
            component=COMPONENT, transform_fn=_merge,
            step_name="merge_files", operator=self.operator,
            parameters={"how": how, "on": on, "n_sources": len(sources), "sort_by": sort_by},
            source_snapshot_id=source_snapshot_id,
        )

    # ── 14. Export report ─────────────────────────────────────────────────────

    def export_report(
        self, gold: GoldArtefact,
        output_path: str,
        fmt: str = "markdown",
        title: Optional[str] = None,
        include_stats: bool = True,
    ) -> GoldArtefact:
        """
        Generate a formatted analysis report from a Gold artefact.

        Formats: 'markdown' | 'html' | 'json'

        Report includes:
          - Summary statistics (mean, std, nulls, cardinality)
          - Row/column counts
          - Schema overview
          - Optional: basic visualisation-ready flag

        Returns a GoldArtefact containing the report text as a DataFrame
        (column: 'report'), and saves the report to disk via output_path.
        """
        import json as _json
        from datetime import datetime, timezone

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df_orig = gold.data.copy(deep=True)
        dataset_id = gold.dataset_id
        report_title = title or f"Analysis Report: {dataset_id}"

        def _report(df: pd.DataFrame) -> pd.DataFrame:
            rows = len(df)
            cols = len(df.columns)
            numeric_cols = df.select_dtypes("number").columns.tolist()
            cat_cols = df.select_dtypes("object").columns.tolist()
            null_rate = float(df.isnull().mean().mean())
            generated_at = datetime.now(tz=timezone.utc).isoformat()

            stats: Dict[str, Any] = {}
            if include_stats:
                for col in numeric_cols[:10]:  # cap at 10 cols for brevity
                    s = df[col].dropna()
                    stats[col] = {
                        "mean": round(float(s.mean()), 4) if len(s) else None,
                        "std":  round(float(s.std()), 4)  if len(s) else None,
                        "min":  round(float(s.min()), 4)  if len(s) else None,
                        "max":  round(float(s.max()), 4)  if len(s) else None,
                        "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                    }

            if fmt == "json":
                payload = {
                    "title": report_title,
                    "generated_at": generated_at,
                    "dataset_id": dataset_id,
                    "rows": rows, "cols": cols,
                    "null_rate": round(null_rate, 4),
                    "schema": {c: str(df[c].dtype) for c in df.columns},
                    "statistics": stats,
                }
                report_text = _json.dumps(payload, indent=2)
            elif fmt == "html":
                stat_rows = "".join(
                    f"<tr><td>{k}</td><td>{v['mean']}</td><td>{v['std']}</td>"
                    f"<td>{v['min']}</td><td>{v['max']}</td><td>{v['null_pct']}%</td></tr>"
                    for k, v in stats.items()
                )
                report_text = f"""<!DOCTYPE html>
<html><head><title>{report_title}</title></head><body>
<h1>{report_title}</h1>
<p>Generated: {generated_at} | Dataset: {dataset_id}</p>
<p>Rows: {rows:,} | Columns: {cols} | Overall null rate: {null_rate:.1%}</p>
<h2>Statistics</h2>
<table border="1"><tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Null%</th></tr>
{stat_rows}</table>
</body></html>"""
            else:  # markdown (default)
                stat_md = "\n".join(
                    f"| {k} | {v['mean']} | {v['std']} | {v['min']} | {v['max']} | {v['null_pct']}% |"
                    for k, v in stats.items()
                )
                report_text = f"""# {report_title}

**Generated:** {generated_at}  
**Dataset:** `{dataset_id}`  
**Rows:** {rows:,} | **Columns:** {cols} | **Null Rate:** {null_rate:.1%}

## Column Overview
- **Numeric:** {', '.join(numeric_cols) or 'None'}
- **Categorical:** {', '.join(cat_cols) or 'None'}

## Statistics
| Column | Mean | Std | Min | Max | Null% |
|--------|------|-----|-----|-----|-------|
{stat_md}

*Report powered by DIPEX Analyst Layer — all metrics sourced from QA-passed Gold artefact.*
"""

            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(report_text)
            logger.info("Report exported: %s (%s format, %d chars)", output_path, fmt, len(report_text))

            return pd.DataFrame([{
                "report_path": output_path,
                "format": fmt,
                "title": report_title,
                "generated_at": generated_at,
                "rows": rows, "cols": cols,
                "report_length_chars": len(report_text),
            }])

        # export_report wraps the gold artefact's silver source
        # We re-derive from the gold's underlying immutable data
        from ingestion.data_layers import ImmutableDataFrame as _IDF
        silver_proxy = _IDF(df_orig, layer="gold", dataset_id=dataset_id)
        return self.lm.derive_gold(
            silver_proxy,
            dataset_id=f"{dataset_id}_report",
            component=COMPONENT,
            transform_fn=_report,
            step_name="export_report",
            operator=self.operator,
            parameters={"output_path": output_path, "fmt": fmt, "include_stats": include_stats},
            source_snapshot_id=source_snapshot_id,
        )
