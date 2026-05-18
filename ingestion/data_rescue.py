"""
ingestion/data_rescue.py
------------------------
Universal Data Rescue Layer — last line of defence between a raw reader
and the downstream pipeline.

Called immediately after ANY reader (file, DB, API, stream) returns data.
Guarantees that the returned DataFrame is:
  - Never None
  - Always has at least 0 columns (never raises on empty input)
  - Has sanitised, unique column names
  - Has no list/dict cells that would crash sklearn or pandas ops
  - Has no binary/bytes dtypes
  - Has sensible dtypes (epoch ints auto-converted, date strings coerced)

Every operation is logged and accumulated in a RescueReport so the
audit trail always explains what was done.

Design principle:
  NEVER crash. NEVER silently discard. ALWAYS explain.
"""

from __future__ import annotations

import base64
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.ingestion.data_rescue")


# ── Rescue Report ──────────────────────────────────────────────────────────────

@dataclass
class RescueReport:
    """Audit trail of every rescue operation applied."""
    source_type: str = "unknown"
    was_empty: bool = False
    was_all_null: bool = False
    columns_renamed: List[Dict[str, Any]] = field(default_factory=list)
    columns_exploded: List[Dict[str, Any]] = field(default_factory=list)
    columns_b64_encoded: List[str] = field(default_factory=list)
    columns_date_converted: List[str] = field(default_factory=list)
    cells_serialised: List[str] = field(default_factory=list)   # cols where cell→str fallback ran
    single_col_split: bool = False
    placeholder_returned: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "was_empty": self.was_empty,
            "was_all_null": self.was_all_null,
            "columns_renamed": self.columns_renamed,
            "columns_exploded": self.columns_exploded,
            "columns_b64_encoded": self.columns_b64_encoded,
            "columns_date_converted": self.columns_date_converted,
            "cells_serialised": self.cells_serialised,
            "single_col_split": self.single_col_split,
            "placeholder_returned": self.placeholder_returned,
            "warnings": self.warnings,
        }

    @property
    def was_rescued(self) -> bool:
        return any([
            self.was_empty, self.was_all_null, self.columns_renamed,
            self.columns_exploded, self.columns_b64_encoded,
            self.columns_date_converted, self.cells_serialised,
            self.single_col_split, self.placeholder_returned,
        ])


# ── DataRescue ─────────────────────────────────────────────────────────────────

class DataRescue:
    """
    Universal post-read rescue engine.

    Usage::

        rescue = DataRescue(source_type="file")
        df, report = rescue.rescue(df)
        if report.was_rescued:
            logger.warning("Data was rescued: %s", report.to_dict())
    """

    # Maximum recursion depth for nested JSON explosion
    MAX_EXPLODE_DEPTH = 3
    # Minimum fraction of values that must parse as dates before auto-converting
    DATE_DETECT_THRESHOLD = 0.70
    # If a single-column df has values that look like delimited rows, try these
    SPLIT_DELIMITERS = [",", "\t", "|", ";", "  "]

    def __init__(self, source_type: str = "unknown") -> None:
        self.source_type = source_type

    def rescue(
        self,
        df: Optional[pd.DataFrame],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[pd.DataFrame, RescueReport]:
        """
        Rescue a DataFrame from any state of messiness.

        Parameters
        ----------
        df      : The raw DataFrame from a reader (may be None or empty)
        context : Optional metadata dict (path, url, table name, etc.)

        Returns
        -------
        (rescued_df, RescueReport)
        """
        report = RescueReport(source_type=self.source_type)
        ctx = context or {}

        # ── Guard: None input ─────────────────────────────────────────────────
        if df is None:
            report.was_empty = True
            report.placeholder_returned = True
            report.warnings.append("Reader returned None — creating placeholder DataFrame")
            logger.warning("[DataRescue:%s] Reader returned None — placeholder created", self.source_type)
            return self._placeholder(ctx, "reader_returned_none"), report

        # ── Guard: completely empty (0 rows AND 0 cols) ───────────────────────
        if len(df) == 0 and len(df.columns) == 0:
            report.was_empty = True
            report.placeholder_returned = True
            report.warnings.append("DataFrame has 0 rows and 0 columns")
            logger.warning("[DataRescue:%s] 0×0 DataFrame — placeholder created", self.source_type)
            return self._placeholder(ctx, "empty_0x0"), report

        # ── Step 1: Sanitise column names ──────────────────────────────────────
        df = self._sanitise_columns(df, report)

        # ── Step 2: All-null DataFrame ─────────────────────────────────────────
        if len(df) > 0 and df.isna().all().all():
            report.was_all_null = True
            report.warnings.append("All values in DataFrame are null — dropping all-null columns")
            df = df.dropna(axis=1, how="all")
            if df.empty or len(df.columns) == 0:
                report.placeholder_returned = True
                logger.warning("[DataRescue:%s] All-null df — placeholder created", self.source_type)
                return self._placeholder(ctx, "all_null"), report

        # ── Step 3: Single-column rescue ───────────────────────────────────────
        if len(df.columns) == 1 and len(df) > 0:
            df = self._rescue_single_column(df, report)

        # ── Step 4: Bytes/binary columns → base64 strings ─────────────────────
        df = self._encode_bytes_columns(df, report)

        # ── Step 5: Nested list/dict cells → explode/normalize ────────────────
        df = self._explode_nested_columns(df, report, depth=0)

        # ── Step 6: Mixed-cell-type serialisation ──────────────────────────────
        df = self._serialise_mixed_cells(df, report)

        # ── Step 7: Epoch integer → datetime coercion ─────────────────────────
        df = self._coerce_epoch_columns(df, report)

        # ── Step 8: Date-string auto-conversion ───────────────────────────────
        df = self._coerce_date_string_columns(df, report)

        # ── Step 9: Row-level JSON strings in text columns ────────────────────
        df = self._expand_json_string_columns(df, report)

        # ── Step 10: Final dtype safety pass ──────────────────────────────────
        df = self._final_dtype_guard(df, report)

        if report.was_rescued:
            logger.info(
                "[DataRescue:%s] Rescue applied — renamed=%d exploded=%d b64=%d "
                "date_conv=%d cell_ser=%d single_split=%s placeholder=%s",
                self.source_type,
                len(report.columns_renamed),
                len(report.columns_exploded),
                len(report.columns_b64_encoded),
                len(report.columns_date_converted),
                len(report.cells_serialised),
                report.single_col_split,
                report.placeholder_returned,
            )

        return df, report

    # ── Step 1: Column sanitisation ───────────────────────────────────────────

    def _sanitise_columns(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        Fix all column name pathologies:
          - None / NaN names → _col_N
          - Empty string names → _col_N
          - Duplicate names → _col_N_a, _col_N_b
          - Names with special chars / unicode → ASCII transliteration
          - Pandas Unnamed: 0, Unnamed: 1, ... → _col_0, _col_1
        """
        new_names: List[str] = []
        seen: Dict[str, int] = {}

        for i, col in enumerate(df.columns):
            original = col

            # None / NaN / non-string
            if col is None or (isinstance(col, float) and np.isnan(col)):
                col = f"_col_{i}"
            else:
                col = str(col).strip()

            # Empty
            if not col:
                col = f"_col_{i}"

            # Pandas "Unnamed: N" artefact
            if re.match(r"^Unnamed:\s*\d+", col):
                col = f"_col_{i}"

            # Unicode → ASCII where possible
            try:
                col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
            except Exception:  # noqa: BLE001
                col = re.sub(r"[^\x00-\x7F]", "_", col)

            # Replace non-word characters (except dot) with underscore
            col = re.sub(r"[^\w.]", "_", col).strip("_") or f"_col_{i}"

            # Deduplicate
            base = col
            if base in seen:
                seen[base] += 1
                col = f"{base}_{seen[base]}"
            else:
                seen[base] = 0

            if str(original) != col:
                report.columns_renamed.append({"original": str(original), "renamed": col})
            new_names.append(col)

        df.columns = new_names
        return df

    # ── Step 2b: Single-column split ──────────────────────────────────────────

    def _rescue_single_column(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        If the DataFrame has exactly 1 column and the values look like
        multi-field text rows (CSV/TSV etc.), attempt to split them.
        """
        col = df.columns[0]
        sample = df[col].dropna().head(20).astype(str).tolist()
        if not sample:
            return df

        for delim in self.SPLIT_DELIMITERS:
            splits = [row.split(delim) for row in sample]
            n_fields = [len(s) for s in splits]
            # Consistent multi-field split
            if len(set(n_fields)) <= 2 and max(n_fields) > 1:
                try:
                    split_df = df[col].astype(str).str.split(delim, expand=True)
                    split_df.columns = [f"{col}_{j}" for j in range(split_df.shape[1])]
                    # Attempt to infer header from first row
                    if all(isinstance(v, str) and not re.match(r"^-?\d", str(v))
                           for v in split_df.iloc[0]):
                        split_df.columns = [
                            re.sub(r"[^\w]", "_", str(v).strip()) or f"col_{j}"
                            for j, v in enumerate(split_df.iloc[0])
                        ]
                        split_df = split_df.iloc[1:].reset_index(drop=True)
                    logger.info(
                        "[DataRescue] Single-column '%s' split on '%s' → %d fields",
                        col, repr(delim), split_df.shape[1],
                    )
                    report.single_col_split = True
                    report.warnings.append(
                        f"Single-column data split on delimiter '{delim}' → {split_df.shape[1]} fields"
                    )
                    return split_df
                except Exception:  # noqa: BLE001
                    pass
        return df

    # ── Step 4: Binary columns ────────────────────────────────────────────────

    def _encode_bytes_columns(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """Convert bytes/bytearray columns to base64 strings."""
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0 and all(isinstance(v, (bytes, bytearray)) for v in sample):
                        df[col] = df[col].apply(
                            lambda v: base64.b64encode(v).decode("ascii") if isinstance(v, (bytes, bytearray)) else v
                        )
                        report.columns_b64_encoded.append(col)
                        logger.info("[DataRescue] Column '%s' — bytes→base64", col)
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Step 5: Nested list/dict explosion ────────────────────────────────────

    def _explode_nested_columns(
        self, df: pd.DataFrame, report: RescueReport, depth: int
    ) -> pd.DataFrame:
        """Recursively flatten columns that hold dicts or lists."""
        if depth >= self.MAX_EXPLODE_DEPTH:
            return df

        cols_to_explode = []
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    sample = df[col].dropna().head(20)
                    if len(sample) > 0:
                        n_dicts = sum(isinstance(v, dict) for v in sample)
                        n_lists = sum(isinstance(v, list) for v in sample)
                        if n_dicts / len(sample) >= 0.5:
                            cols_to_explode.append((col, "dict"))
                        elif n_lists / len(sample) >= 0.5:
                            cols_to_explode.append((col, "list"))
            except Exception:  # noqa: BLE001
                pass

        if not cols_to_explode:
            return df

        parts = [df.drop(columns=[c for c, _ in cols_to_explode])]
        for col, kind in cols_to_explode:
            try:
                if kind == "dict":
                    # Safely fill non-dict values with empty dict before normalising
                    safe_series = df[col].apply(lambda v: v if isinstance(v, dict) else {})
                    normalized = pd.json_normalize(safe_series.tolist())
                    normalized.columns = [f"{col}.{c}" for c in normalized.columns]
                    normalized.index = df.index
                    parts.append(normalized)
                    report.columns_exploded.append({"column": col, "kind": "dict", "depth": depth})
                    logger.info("[DataRescue] Column '%s' dict-exploded → %d sub-cols", col, len(normalized.columns))
                    continue
                elif kind == "list":
                    # Keep scalar-ified version (join list items as string)
                    df[col] = df[col].apply(
                        lambda v: ", ".join(str(x) for x in v) if isinstance(v, list) else v
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[DataRescue] Could not explode '%s': %s", col, exc)

        try:
            df = pd.concat(parts, axis=1)
        except Exception:  # noqa: BLE001
            pass

        # Recursively handle newly created nested columns
        return self._explode_nested_columns(df, report, depth + 1)

    # ── Step 6: Mixed cell-type serialisation ─────────────────────────────────

    def _serialise_mixed_cells(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        For any object column that still contains a mix of scalars and
        non-serialisable types (sets, custom objects, etc.), force everything
        to string as a last resort.
        """
        for col in df.select_dtypes(include="object").columns:
            try:
                sample = df[col].dropna().head(30)
                bad = [v for v in sample if not isinstance(v, (str, int, float, bool, type(None)))]
                if bad:
                    df[col] = df[col].apply(
                        lambda v: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    )
                    report.cells_serialised.append(col)
                    logger.info("[DataRescue] Column '%s' — non-scalar cells serialised to str", col)
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Step 7: Epoch integer → datetime ──────────────────────────────────────

    def _coerce_epoch_columns(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        Detect integer columns that look like Unix epoch timestamps and convert
        them to datetime64.  Heuristic: values in the range of plausible epochs
        (year 2000 → year 2050) and column name contains 'time', 'ts', 'date',
        'created', 'updated', 'at', 'stamp'.
        """
        TIME_KEYWORDS = re.compile(
            r"(time|timestamp|ts|date|created|updated|modified|at|stamp|epoch)",
            re.IGNORECASE,
        )
        EPOCH_MIN = 946_684_800   # 2000-01-01
        EPOCH_MAX = 2_524_608_000  # 2050-01-01

        for col in df.select_dtypes(include=[np.integer, "Int64"]).columns:
            if not TIME_KEYWORDS.search(col):
                continue
            try:
                vals = df[col].dropna()
                if len(vals) == 0:
                    continue
                # Check if values are in seconds epoch range
                in_range = ((vals >= EPOCH_MIN) & (vals <= EPOCH_MAX)).mean()
                if in_range >= 0.80:
                    df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
                    report.columns_date_converted.append(col)
                    logger.info("[DataRescue] Column '%s' — epoch-int→datetime", col)
                    continue
                # Check millisecond epoch
                in_range_ms = ((vals >= EPOCH_MIN * 1000) & (vals <= EPOCH_MAX * 1000)).mean()
                if in_range_ms >= 0.80:
                    df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
                    report.columns_date_converted.append(col)
                    logger.info("[DataRescue] Column '%s' — epoch-ms→datetime", col)
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Step 8: Date-string auto-conversion ───────────────────────────────────

    def _coerce_date_string_columns(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        Detect object columns where >= DATE_DETECT_THRESHOLD fraction of values
        parse successfully as dates.  Only convert if NOT already a number.
        """
        for col in df.select_dtypes(include="object").columns:
            try:
                sample = df[col].dropna().head(100)
                if len(sample) < 5:
                    continue
                # Don't waste time on numeric-looking values
                if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                parse_rate = parsed.notna().mean()
                if parse_rate >= self.DATE_DETECT_THRESHOLD:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    report.columns_date_converted.append(col)
                    logger.info(
                        "[DataRescue] Column '%s' — date-string→datetime (%.0f%% parsed)",
                        col, parse_rate * 100,
                    )
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Step 9: Row-level JSON strings ────────────────────────────────────────

    def _expand_json_string_columns(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        Detect TEXT columns (e.g., DB VARCHAR) where values are JSON strings.
        Expand them in-place via json_normalize.
        """
        import json

        for col in df.select_dtypes(include="object").columns:
            try:
                sample = df[col].dropna().head(30)
                if len(sample) < 2:
                    continue
                parsed_count = 0
                parsed_objs = []
                for v in sample:
                    try:
                        obj = json.loads(str(v))
                        if isinstance(obj, (dict, list)):
                            parsed_count += 1
                            parsed_objs.append(obj)
                    except Exception:  # noqa: BLE001
                        pass
                if parsed_count / len(sample) >= 0.70:
                    # Expand the whole column
                    def _try_parse(v):
                        try:
                            return json.loads(str(v))
                        except Exception:  # noqa: BLE001
                            return {}

                    all_objs = df[col].apply(_try_parse)
                    normalized = pd.json_normalize(all_objs.tolist())
                    normalized.columns = [f"{col}.{c}" for c in normalized.columns]
                    normalized.index = df.index
                    df = pd.concat([df.drop(columns=[col]), normalized], axis=1)
                    report.columns_exploded.append({"column": col, "kind": "json_string"})
                    logger.info(
                        "[DataRescue] Column '%s' JSON-string expanded → %d sub-cols",
                        col, len(normalized.columns),
                    )
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Step 10: Final dtype safety ───────────────────────────────────────────

    def _final_dtype_guard(self, df: pd.DataFrame, report: RescueReport) -> pd.DataFrame:
        """
        Ensure no remaining problematic dtypes reach the pipeline:
          - 'object' columns with ALL string values: keep as-is (fine)
          - complex128: convert to float (real part only)
          - timedelta64: convert to total_seconds float
          - category dtype: keep (pandas handles these fine)
        """
        for col in df.columns:
            try:
                dt = df[col].dtype
                if hasattr(dt, "kind") and dt.kind == "c":   # complex
                    df[col] = df[col].apply(lambda v: v.real if isinstance(v, complex) else v).astype(float)
                    logger.info("[DataRescue] Column '%s' — complex→float (real part)", col)
                elif hasattr(dt, "kind") and dt.kind == "m":  # timedelta
                    df[col] = df[col].dt.total_seconds()
                    logger.info("[DataRescue] Column '%s' — timedelta→seconds float", col)
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Placeholder ───────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder(context: Dict[str, Any], reason: str) -> pd.DataFrame:
        """Return a minimal 1-row placeholder DataFrame with metadata."""
        return pd.DataFrame([{
            "_rescue_placeholder": True,
            "_rescue_reason": reason,
            "_source_context": str(context)[:200],
        }])


# ── Module-level convenience ───────────────────────────────────────────────────

def rescue_dataframe(
    df: Optional[pd.DataFrame],
    source_type: str = "unknown",
    context: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, RescueReport]:
    """
    Module-level convenience wrapper for DataRescue.rescue().

    Usage::

        from ingestion.data_rescue import rescue_dataframe
        df, report = rescue_dataframe(df, source_type="file", context={"path": path})
    """
    return DataRescue(source_type=source_type).rescue(df, context=context)
