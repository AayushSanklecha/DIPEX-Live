"""
ingestion/normaliser.py
-------------------------
Source-agnostic normalisation pipeline.

Every reader (file, API, DB, stream) produces a raw DataFrame.
The Normaliser converts it into a clean, unified DataFrame + ColumnMeta list
ready for the ISSF snapshot.

Steps
-----
1. Column name normalisation   — snake_case, strip whitespace, deduplicate
2. Null unification            — None, NaN, 'NULL', 'N/A', '', 'nan' → pd.NA
3. Type inference & coercion   — numeric → datetime → bool → string fallback
4. Timestamp standardisation   — all datetime columns → ISO 8601 UTC
5. Primary key detection       — uniqueness rate > 0.99 and non-null
6. Column metadata extraction  — ColumnMeta list for ISSF
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.issf import ColumnMeta

logger = logging.getLogger("dipex.ingestion.normaliser")

# Values treated as NULL
_NULL_STRINGS = frozenset({
    "null", "none", "n/a", "na", "nan", "nil",
    "#n/a", "#na", "-", ".", "—", "undefined",
    "", " ",
})


class Normaliser:
    """
    Converts raw DataFrames from any source into normalised, ISSF-ready form.

    Parameters
    ----------
    max_sample_values : int
        Number of sample values to store per column in ColumnMeta.
    infer_types : bool
        If True, attempt type coercion on object columns.
    standardise_timestamps : bool
        If True, convert all detected datetime columns to UTC ISO strings.
    pk_uniqueness_threshold : float
        Uniqueness rate above which a column is a PK candidate.
    """

    def __init__(
        self,
        max_sample_values: int = 5,
        infer_types: bool = True,
        standardise_timestamps: bool = True,
        pk_uniqueness_threshold: float = 0.99,
    ) -> None:
        self.max_sample = max_sample_values
        self.infer_types = infer_types
        self.standardise_timestamps = standardise_timestamps
        self.pk_threshold = pk_uniqueness_threshold

    def normalise(
        self,
        df: pd.DataFrame,
        dataset_id: str = "",
    ) -> Tuple[pd.DataFrame, List[ColumnMeta]]:
        """
        Apply the full normalisation pipeline.

        Returns
        -------
        (clean_df, column_metadata_list)
        """
        if df is None or df.empty:
            logger.warning("[%s] Empty DataFrame passed to normaliser.", dataset_id)
            return pd.DataFrame(), []

        df = df.copy()

        # 1. Column name normalisation
        df = self._normalise_column_names(df)

        # 2. Null unification
        df = self._unify_nulls(df)

        # 2.5. Handle complex/unhashable types (Lists, Dicts)
        df = self._stringify_complex_columns(df)

        # 3. Type coercion
        if self.infer_types:
            df = self._coerce_types(df)

        # 4. Timestamp standardisation
        if self.standardise_timestamps:
            df = self._standardise_timestamps(df)

        # 5. Build column metadata
        column_meta = self._build_column_meta(df)

        logger.info(
            "[%s] Normalised: %d rows × %d cols. Types: %s",
            dataset_id, len(df), len(df.columns),
            {c.name: c.dtype for c in column_meta},
        )
        return df, column_meta

    # ── Step 1: Column names ──────────────────────────────────────────────────

    @staticmethod
    def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Convert column names to snake_case, strip specials, deduplicate."""
        seen: Dict[str, int] = {}
        new_cols: List[str] = []
        for col in df.columns:
            # Convert to snake_case
            name = str(col).strip()
            name = re.sub(r"[\s\-\.]+", "_", name)        # spaces/dashes/dots → _
            name = re.sub(r"[^\w]", "", name)              # remove non-word chars
            name = re.sub(r"_+", "_", name)                # collapse multiple _
            name = name.strip("_").lower()
            if not name:
                name = "col"
            # Deduplicate
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            new_cols.append(name)
        df.columns = new_cols
        return df

    # ── Step 2: Null unification ──────────────────────────────────────────────

    @staticmethod
    def _unify_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Replace all null sentinels with pd.NA."""
        def _is_null_sentinel(v):
            if v is None: return True
            if isinstance(v, (list, dict, set, np.ndarray, pd.Series)): return False
            if pd.isna(v): return True
            # Don't try to stringify lists/dicts here
            try:
                s = str(v).strip().lower()
                return s in _NULL_STRINGS
            except Exception:
                return False

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(lambda v: pd.NA if _is_null_sentinel(v) else v)
        return df

    @staticmethod
    def _stringify_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns containing lists/dicts to JSON strings so they are hashable."""
        for col in df.select_dtypes(include=["object"]).columns:
            # Check if any element is unhashable
            has_complex = False
            sample = df[col].dropna().head(10)
            for v in sample:
                if isinstance(v, (list, dict, set)):
                    has_complex = True
                    break
            
            if has_complex:
                try:
                    import json
                    # We use JSON stringification for lists/dicts
                    df[col] = df[col].apply(
                        lambda v: json.dumps(v) if isinstance(v, (list, dict, set)) else v
                    )
                except Exception:
                    # Fallback to plain string
                    df[col] = df[col].astype(str).replace("nan", pd.NA).replace("None", pd.NA)
        return df

    # ── Step 3: Type coercion ─────────────────────────────────────────────────

    @staticmethod
    def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        For each object column, try: numeric → datetime → boolean → keep string.
        Uses pd.to_numeric / pd.to_datetime with errors='coerce'.
        """
        for col in df.select_dtypes(include=["object"]).columns:
            series = df[col]
            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            # Try numeric
            try:
                converted = pd.to_numeric(series, errors="coerce")
                if converted.notna().sum() >= 0.9 * non_null.count():
                    df[col] = converted
                    continue
            except Exception:  # noqa: BLE001
                pass

            # Try datetime
            try:
                converted = pd.to_datetime(series, errors="coerce", utc=True)
                if converted.notna().sum() >= 0.85 * non_null.count():
                    df[col] = converted
                    continue
            except Exception:  # noqa: BLE001
                pass

            # Try boolean
            try:
                bool_map = {"true": True, "false": False, "yes": True, "no": False,
                            "1": True, "0": False, "t": True, "f": False}
                # Safe unique check for bools
                def _to_lower_safe(v):
                    if isinstance(v, str): return v.lower()
                    return str(v).lower()
                
                unique_vals = non_null.apply(_to_lower_safe).unique()
                if len(unique_vals) > 0 and set(unique_vals).issubset(set(bool_map.keys())):
                    df[col] = non_null.apply(_to_lower_safe).map(bool_map)
            except Exception:
                pass

        return df

    # ── Step 4: Timestamp standardisation ────────────────────────────────────

    @staticmethod
    def _standardise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime64 columns to UTC ISO-8601 strings."""
        for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            try:
                if hasattr(df[col].dt, "tz") and df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize("UTC")
                df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:  # noqa: BLE001
                pass
        return df

    # ── Step 5: Column metadata ───────────────────────────────────────────────

    def _build_column_meta(self, df: pd.DataFrame) -> List[ColumnMeta]:
        """Build ColumnMeta for every column — used in ISSF."""
        n = len(df)
        metas: List[ColumnMeta] = []
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_rate  = null_count / n if n > 0 else 0.0
            non_null   = series.dropna()
            
            try:
                unique_count = int(series.nunique(dropna=True))
            except Exception:
                # Robust fallback for unhashable types if they leaked through
                try:
                    unique_count = int(series.astype(str).nunique(dropna=True))
                except Exception:
                    unique_count = 0

            is_pk = (
                unique_count == n
                and null_count == 0
                and n > 0
            ) or (unique_count / max(n, 1) >= self.pk_threshold and null_count == 0)

            dtype_str = str(series.dtype)
            sample = list(non_null.head(self.max_sample).values)

            min_val: Any = None
            max_val: Any = None
            if pd.api.types.is_numeric_dtype(series):
                try:
                    min_val = float(non_null.min())
                    max_val = float(non_null.max())
                except Exception:  # noqa: BLE001
                    pass

            metas.append(ColumnMeta(
                name=col, dtype=dtype_str,
                null_count=null_count, null_rate=null_rate,
                unique_count=unique_count, is_pk_candidate=is_pk,
                sample_values=sample, min_val=min_val, max_val=max_val,
            ))
        return metas

    # ── Convenience: schema dict from DataFrame ───────────────────────────────

    @staticmethod
    def extract_schema(df: pd.DataFrame) -> Dict[str, str]:
        """Return {column_name: dtype_str} for schema registry."""
        return {col: str(df[col].dtype) for col in df.columns}
