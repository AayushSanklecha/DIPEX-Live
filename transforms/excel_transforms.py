"""
transforms/excel_transforms.py
--------------------------------
Excel-style data manipulation transforms for DIPEX pipelines.

Provides common spreadsheet-style operations that analysts often apply
to tabular data: normalization, binning, pivoting, type coercion,
conditional flagging, and Excel-compatible formula emulation.

All transforms accept a DataFrame and return a (transformed) DataFrame.
They are designed to be registered in the TransformRegistry.

Usage
-----
    from transforms.excel_transforms import ExcelTransforms
    from transforms.transform_registry import TransformRegistry

    registry = TransformRegistry()
    registry.register("normalize_zscore", ExcelTransforms.normalize_zscore)
    registry.register("bin_numeric",      ExcelTransforms.bin_numeric)
    df_clean = registry.apply("normalize_zscore", df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.transforms.excel")


class ExcelTransforms:
    """
    Collection of stateless Excel-style DataFrame transform methods.

    All methods are @staticmethod — they can be used directly or
    registered into a TransformRegistry.
    """

    # ── Normalization ─────────────────────────────────────────────────────────

    @staticmethod
    def normalize_zscore(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        suffix: str = "_zscore",
    ) -> pd.DataFrame:
        """
        Z-score normalise numeric columns.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str, optional
            Columns to normalise. Defaults to all numeric columns.
        suffix : str
            Suffix appended to original column name for output column.
            If empty string, overwrites the original column.

        Returns
        -------
        pd.DataFrame with normalised columns added (or in-place if suffix='').
        """
        df = df.copy()
        cols = columns or list(df.select_dtypes(include=[np.number]).columns)
        for col in cols:
            s = df[col]
            mu = s.mean()
            sigma = s.std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                normalised = pd.Series(np.zeros(len(s)), index=s.index)
            else:
                normalised = (s - mu) / sigma
            out_col = col if not suffix else f"{col}{suffix}"
            df[out_col] = normalised
        return df

    @staticmethod
    def normalize_minmax(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0.0, 1.0),
        suffix: str = "_mm",
    ) -> pd.DataFrame:
        """
        Min-max scale numeric columns into [feature_range[0], feature_range[1]].
        Handles zero-range columns (constant → 0.0).
        """
        df = df.copy()
        lo, hi = feature_range
        cols = columns or list(df.select_dtypes(include=[np.number]).columns)
        for col in cols:
            s = df[col].replace([np.inf, -np.inf], np.nan)
            col_min = s.min()
            col_max = s.max()
            rng = col_max - col_min
            if rng == 0 or np.isnan(rng):
                scaled = pd.Series(np.full(len(s), lo), index=s.index)
            else:
                scaled = lo + (s - col_min) / rng * (hi - lo)
            out_col = col if not suffix else f"{col}{suffix}"
            df[out_col] = scaled
        return df

    # ── Binning ───────────────────────────────────────────────────────────────

    @staticmethod
    def bin_numeric(
        df: pd.DataFrame,
        column: str,
        bins: Union[int, List[float]] = 5,
        labels: Optional[List[str]] = None,
        out_col: Optional[str] = None,
        include_lowest: bool = True,
    ) -> pd.DataFrame:
        """
        Bin a numeric column into categorical intervals (Excel IFS / VLOOKUP style).

        Parameters
        ----------
        df : pd.DataFrame
        column : str
            Column to bin.
        bins : int or list of float
            Number of equal-width bins, or explicit bin edges.
        labels : list of str, optional
            Labels for each bin. Length must be len(bins)-1 if bins is a list,
            or `bins` if it is an integer.
        out_col : str, optional
            Output column name. Defaults to f"{column}_bin".
        """
        df = df.copy()
        out = out_col or f"{column}_bin"
        df[out] = pd.cut(
            df[column],
            bins=bins,
            labels=labels,
            include_lowest=include_lowest,
        )
        return df

    # ── Type coercion ─────────────────────────────────────────────────────────

    @staticmethod
    def coerce_types(
        df: pd.DataFrame,
        type_map: Dict[str, str],
        errors: str = "coerce",
    ) -> pd.DataFrame:
        """
        Coerce columns to specified dtypes.

        Parameters
        ----------
        type_map : dict
            { column_name: dtype_string }, e.g. {"age": "int32", "date": "datetime64[ns]"}
        errors : str
            'coerce' silently converts failures to NaN; 'raise' raises on error.
        """
        df = df.copy()
        for col, dtype in type_map.items():
            if col not in df.columns:
                logger.warning("coerce_types: column '%s' not found — skipped", col)
                continue
            try:
                if "datetime" in dtype:
                    df[col] = pd.to_datetime(df[col], errors=errors)
                elif dtype in ("int", "int32", "int64"):
                    df[col] = pd.to_numeric(df[col], errors=errors).astype("Int64")
                elif dtype in ("float", "float32", "float64"):
                    df[col] = pd.to_numeric(df[col], errors=errors)
                elif dtype in ("bool", "boolean"):
                    df[col] = df[col].astype(bool)
                elif dtype in ("str", "string", "object"):
                    df[col] = df[col].astype(str)
                elif dtype == "category":
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as exc:  # noqa: BLE001
                logger.warning("coerce_types: failed to coerce '%s' to '%s': %s", col, dtype, exc)
        return df

    # ── Conditional flagging ──────────────────────────────────────────────────

    @staticmethod
    def flag_outliers(
        df: pd.DataFrame,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
        flag_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add a boolean flag column marking outliers in a numeric column.

        Parameters
        ----------
        method : str
            'iqr'  — flag if value < Q1 - threshold*IQR or > Q3 + threshold*IQR
            'zscore' — flag if |z-score| > threshold
        """
        df = df.copy()
        flag = flag_col or f"{column}_outlier"
        s = pd.to_numeric(df[column], errors="coerce")

        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            df[flag] = (s < q1 - threshold * iqr) | (s > q3 + threshold * iqr)
        elif method == "zscore":
            mu, sigma = s.mean(), s.std(ddof=1)
            z = (s - mu) / sigma if sigma != 0 else pd.Series(np.zeros(len(s)), index=s.index)
            df[flag] = z.abs() > threshold
        else:
            raise ValueError(f"Unknown outlier method: {method!r}. Use 'iqr' or 'zscore'.")

        return df

    @staticmethod
    def add_conditional_column(
        df: pd.DataFrame,
        conditions: List[tuple],
        out_col: str,
        default: Any = None,
    ) -> pd.DataFrame:
        """
        Excel-IF-style conditional column.

        Parameters
        ----------
        conditions : list of (condition_series_or_callable, value)
            Applied in order — first match wins (like Excel IFS).
        out_col : str
            Name of the new column.
        default : any
            Value when no condition matches.

        Example
        -------
            ExcelTransforms.add_conditional_column(df, [
                (df["score"] >= 90, "A"),
                (df["score"] >= 75, "B"),
                (df["score"] >= 60, "C"),
            ], out_col="grade", default="F")
        """
        df = df.copy()
        result = pd.Series([default] * len(df), index=df.index)
        # Apply in reverse so first condition wins
        for cond, val in reversed(conditions):
            mask = cond(df) if callable(cond) else cond
            result[mask] = val
        df[out_col] = result
        return df

    # ── Missing value handling ────────────────────────────────────────────────

    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        strategy: str = "median",
        columns: Optional[List[str]] = None,
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """
        Fill missing values using a specified strategy.

        Parameters
        ----------
        strategy : str
            'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'
        fill_value : any
            Only used when strategy='constant'.
        """
        df = df.copy()
        cols = columns or list(df.columns)

        for col in cols:
            if col not in df.columns:
                continue
            s = df[col]
            if strategy == "mean":
                df[col] = s.fillna(pd.to_numeric(s, errors="coerce").mean())
            elif strategy == "median":
                df[col] = s.fillna(pd.to_numeric(s, errors="coerce").median())
            elif strategy == "mode":
                mode = s.mode(dropna=True)
                df[col] = s.fillna(mode.iloc[0] if len(mode) > 0 else None)
            elif strategy == "ffill":
                df[col] = s.ffill()
            elif strategy == "bfill":
                df[col] = s.bfill()
            elif strategy == "constant":
                df[col] = s.fillna(fill_value)
            else:
                raise ValueError(f"Unknown fill strategy: {strategy!r}.")

        return df

    # ── Aggregation / pivot ───────────────────────────────────────────────────

    @staticmethod
    def pivot_summary(
        df: pd.DataFrame,
        index: str,
        columns: str,
        values: str,
        aggfunc: str = "mean",
    ) -> pd.DataFrame:
        """
        Excel-PivotTable-style summary.

        Returns
        -------
        pd.DataFrame — pivot table (index × columns → aggregated values)
        """
        return pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0,
        ).reset_index()

    @staticmethod
    def add_row_stats(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        stats: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add row-wise statistics (Excel SUMIF / AVERAGEIF style) as new columns.

        Parameters
        ----------
        columns : list of str
            Numeric columns to compute row stats over.
        stats : list of str
            Any of: 'sum', 'mean', 'min', 'max', 'std', 'count_null'.
        """
        df = df.copy()
        num_cols = columns or list(df.select_dtypes(include=[np.number]).columns)
        requested = stats or ["sum", "mean"]
        subset = df[num_cols].apply(pd.to_numeric, errors="coerce")

        for stat in requested:
            if stat == "sum":
                df[f"_row_sum"]   = subset.sum(axis=1)
            elif stat == "mean":
                df["_row_mean"]   = subset.mean(axis=1)
            elif stat == "min":
                df["_row_min"]    = subset.min(axis=1)
            elif stat == "max":
                df["_row_max"]    = subset.max(axis=1)
            elif stat == "std":
                df["_row_std"]    = subset.std(axis=1, ddof=1)
            elif stat == "count_null":
                df["_row_nulls"]  = df[num_cols].isna().sum(axis=1)
            else:
                logger.warning("add_row_stats: unknown stat '%s' — skipped", stat)
        return df
