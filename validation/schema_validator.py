"""
validation/schema_validator.py
-------------------------------
Validates a DataFrame against a schema definition.

Checks:
  1. Required columns presence
  2. Datatype enforcement (explicit dtype compatibility map)
  3. Timestamp consistency (future dates, epoch validity)
  4. Unique key constraints (single and compound keys)
  5. Null threshold (global, used by HardGate internally)

All state is local to each `validate()` call — thread-safe by design.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Maps schema type keywords to the pandas/numpy dtype strings they are compatible with.
_TYPE_COMPAT: Dict[str, List[str]] = {
    "int":      ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
    "float":    ["float16", "float32", "float64"],
    "bool":     ["bool"],
    "str":      ["object", "string"],
    "object":   ["object", "string"],
    "string":   ["object", "string"],
    "datetime": ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64"],
    "category": ["category"],
}

_NOW_UTC = datetime.now(timezone.utc)


class SchemaValidator:
    """
    Validates a DataFrame against schema definitions.

    Schema info dict keys (all optional):
      required_columns: List[str]
      types:            Dict[str, str]          — column → expected type keyword
      timestamp_columns: List[str]              — columns to check for future/invalid dates
      unique_keys:       List[str | List[str]]  — single column or compound key tuples
      ranges:           Dict[str, {min, max}]   — basic range bounds
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def validate(
        self,
        df: pd.DataFrame,
        schema_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Main entry point.

        Returns:
            List of error/warning dicts with keys:
            ``column``, ``severity``, ``type``, ``message``.
        """
        errors: List[Dict[str, Any]] = []

        # 1. Check for schema drift (missing / new / changed columns compared to last known schema)
        self._check_schema_drift(df, schema_info, errors)

        self._check_required_columns(df, schema_info.get("required_columns", []), errors)
        self._check_types(df, schema_info.get("types", {}), errors)
        # NOTE: Null threshold check intentionally removed — NullValidator is authoritative
        # and provides three-tier severity (CRITICAL/ERROR/WARNING) with per-column overrides.
        # Calling _check_nulls here caused double-counting of null violations.
        self._check_timestamps(df, schema_info.get("timestamp_columns", []), errors)
        self._check_unique_keys(df, schema_info.get("unique_keys", []), errors)
        self._check_ranges(df, schema_info.get("ranges", {}), errors)

        if errors:
            n_err = sum(1 for e in errors if e["severity"] in {"ERROR", "CRITICAL"})
            n_warn = len(errors) - n_err
            logger.info(
                "SchemaValidator: %d error(s), %d warning(s) found.", n_err, n_warn
            )
        return errors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_schema_drift(
        self,
        df: pd.DataFrame,
        schema_info: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> None:
        """Detects silent data contract violations: missing, new, or dtype-changed columns."""
        expected_types = schema_info.get("types", {})
        if not expected_types:
            return  # No baseline schema to compare against
            
        expected_cols = set(expected_types.keys())
        actual_cols = set(df.columns)
        
        # 1. Missing columns
        missing = expected_cols - actual_cols
        for col in missing:
            errors.append({
                "column": col,
                "severity": "ERROR",
                "type": "SCHEMA_DRIFT_MISSING_COLUMN",
                "message": f"Schema drift: expected column '{col}' has disappeared."
            })
            
        # 2. New columns
        new_cols = actual_cols - expected_cols
        for col in new_cols:
            errors.append({
                "column": col,
                "severity": "WARNING",
                "type": "SCHEMA_DRIFT_NEW_COLUMN",
                "message": f"Schema drift: unexpected new column '{col}' added."
            })
            
        # 3. Changed data types (caught partially by _check_types, but we explicitly flag drift here)
        for col in expected_cols.intersection(actual_cols):
            expected = expected_types[col].lower()
            actual_dtype = str(df[col].dtype)
            compatible = _TYPE_COMPAT.get(expected, [expected])
            
            # If the dtype completely shifted out of its compatible family
            if not any(actual_dtype.startswith(c) for c in compatible):
                # Don't flag int->float coercion as a drift shift if they are whole numbers
                # (handled gracefully by _check_types)
                if expected == "int" and actual_dtype.startswith("float"):
                     series = df[col].dropna()
                     if not series.empty and (series % 1 == 0).all():
                         continue
                
                errors.append({
                    "column": col,
                    "severity": "WARNING",
                    "type": "SCHEMA_DRIFT_DTYPE_CHANGE",
                    "message": f"Schema drift: column '{col}' shifted from '{expected}' to '{actual_dtype}'."
                })

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        required: List[str],
        errors: List[Dict[str, Any]],
    ) -> None:
        """Fail hard if any required column is absent from the DataFrame."""
        missing = [col for col in required if col not in df.columns]
        for col in missing:
            errors.append({
                "column": col,
                "severity": "CRITICAL",
                "type": "MISSING_REQUIRED_COLUMN",
                "message": (
                    f"Required column '{col}' is missing from the dataset. "
                    "Pipeline cannot continue without it."
                ),
            })
            logger.error("Required column '%s' is missing.", col)

    def _check_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, str],
        errors: List[Dict[str, Any]],
    ) -> None:
        for col, expected in expected_types.items():
            if col not in df.columns:
                logger.warning("Type check: column '%s' not in DataFrame — skipped.", col)
                continue

            actual_dtype = str(df[col].dtype)
            compatible = _TYPE_COMPAT.get(expected.lower(), [expected])
            if not any(actual_dtype.startswith(c) for c in compatible):
                # Fix 10: int declared as float64 is extremely common in CSVs with NaN rows.
                # pandas reads int columns as float64 when NaN is present. If all non-null
                # values are whole numbers, downgrade to WARNING instead of ERROR.
                severity = "ERROR"
                if expected.lower() == "int" and actual_dtype.startswith("float"):
                    series = df[col].dropna()
                    if not series.empty and (series % 1 == 0).all():
                        severity = "WARNING"
                        logger.warning(
                            "Type check: column '%s' declared int, found float64 with "
                            "whole-number values (likely NaN→float64 coercion) — WARNING only.",
                            col,
                        )
                errors.append({
                    "column": col,
                    "severity": severity,
                    "type": "TYPE_MISMATCH",
                    "message": (
                        f"Column '{col}' expected type '{expected}' "
                        f"(compatible with {compatible}), found '{actual_dtype}'. "
                        + ("Values are whole numbers — likely NaN coercion." if severity == "WARNING" else "")
                    ),
                })

    def _check_nulls(self, df: pd.DataFrame, errors: List[Dict[str, Any]]) -> None:
        """DEPRECATED: Null checking is now handled exclusively by NullValidator.

        NullValidator provides three-tier severity (CRITICAL/ERROR/WARNING),
        per-column threshold overrides, RL-dynamic thresholds, and critical-field
        zero-tolerance. Calling this method causes double-counting. It is retained
        as a no-op to avoid breaking any code that still calls it directly.
        """
        # Intentional no-op: see NullValidator.validate() for the authoritative check.

    def _check_timestamps(
        self,
        df: pd.DataFrame,
        timestamp_columns: List[str],
        errors: List[Dict[str, Any]],
    ) -> None:
        """
        Checks declared timestamp columns for:
          - Future timestamps (beyond current UTC time)
          - Negative epoch values (Unix timestamp < 0)
          - Columns that cannot be parsed as datetime at all
        """
        for col in timestamp_columns:
            if col not in df.columns:
                continue

            # Try to coerce to datetime if not already
            series = df[col]
            if not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    series = pd.to_datetime(series, utc=True, errors="raise")
                except Exception:
                    errors.append({
                        "column": col,
                        "severity": "ERROR",
                        "type": "TIMESTAMP_PARSE_FAILURE",
                        "message": (
                            f"Column '{col}' is declared as a timestamp column "
                            "but cannot be parsed as datetime."
                        ),
                    })
                    continue

            # Convert to UTC for comparison
            if series.dt.tz is None:
                series = series.dt.tz_localize("UTC")
            else:
                series = series.dt.tz_convert("UTC")

            now = pd.Timestamp.now(tz="UTC")
            future_count = (series > now).sum()
            if future_count > 0:
                errors.append({
                    "column": col,
                    "severity": "WARNING",
                    "type": "FUTURE_TIMESTAMP",
                    "message": (
                        f"Column '{col}': {future_count} timestamp(s) are in the future. "
                        f"Max observed: {series.max()}. "
                        "Possible clock skew or pre-dated record error."
                    ),
                })

            # Check for suspiciously early epoch (before 1970-01-01 UTC)
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
            pre_epoch_count = (series < epoch).sum()
            if pre_epoch_count > 0:
                errors.append({
                    "column": col,
                    "severity": "WARNING",
                    "type": "PRE_EPOCH_TIMESTAMP",
                    "message": (
                        f"Column '{col}': {pre_epoch_count} timestamp(s) "
                        "are before 1970-01-01 (Unix epoch). "
                        "Verify these are intentional historical records."
                    ),
                })

    def _check_unique_keys(
        self,
        df: pd.DataFrame,
        unique_keys: List[Any],
        errors: List[Dict[str, Any]],
    ) -> None:
        """
        Validates uniqueness constraints.

        ``unique_keys`` can contain:
          - A single column name (str): e.g. "transaction_id"
          - A list of column names (compound key): e.g. ["date", "customer_id"]
        """
        for key_def in unique_keys:
            cols: List[str] = [key_def] if isinstance(key_def, str) else list(key_def)

            # Skip if any column in the key is absent
            missing = [c for c in cols if c not in df.columns]
            if missing:
                logger.warning(
                    "Unique key check: column(s) %s not in DataFrame — skipped.", missing
                )
                continue

            subset = df[cols]
            dup_mask = subset.duplicated(keep=False)
            dup_count = dup_mask.sum()

            if dup_count > 0:
                key_label = cols[0] if len(cols) == 1 else f"({', '.join(cols)})"
                errors.append({
                    "column": key_label,
                    "severity": "ERROR",
                    "type": "UNIQUE_KEY_VIOLATION",
                    "message": (
                        f"Unique key violation on {key_label}: "
                        f"{dup_count} duplicate row(s) detected."
                    ),
                })
                logger.error(
                    "Unique key violation on %s: %d duplicate(s).", key_label, dup_count
                )

    def _check_ranges(
        self,
        df: pd.DataFrame,
        range_defs: Dict[str, Dict[str, float]],
        errors: List[Dict[str, Any]],
    ) -> None:
        for col, bounds in range_defs.items():
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            min_val = bounds.get("min")
            max_val = bounds.get("max")

            if min_val is not None and series.min() < min_val:
                errors.append({
                    "column": col,
                    "severity": "WARNING",
                    "type": "VALUE_OUT_OF_RANGE",
                    "message": (
                        f"Column '{col}' min {series.min():.4g} "
                        f"is below declared minimum {min_val}."
                    ),
                })
            if max_val is not None and series.max() > max_val:
                errors.append({
                    "column": col,
                    "severity": "WARNING",
                    "type": "VALUE_OUT_OF_RANGE",
                    "message": (
                        f"Column '{col}' max {series.max():.4g} "
                        f"is above declared maximum {max_val}."
                    ),
                })
