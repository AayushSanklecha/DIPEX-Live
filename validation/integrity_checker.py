"""
validation/integrity_checker.py
---------------------------------
Referential integrity and cross-column consistency checks for Hard Gate 1.

Checks:
  1. Duplicate rows detection
  2. ID uniqueness (single or compound)
  3. Referential integrity — foreign key values must exist in a reference set
  4. Cross-column conditional rules (if col_a == value then col_b must/must_not be null)
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class IntegrityChecker:
    """
    Checks structural and referential integrity of a DataFrame.

    Configuration example (from config.yaml):
      validation:
        integrity:
          check_duplicates: true
          id_columns: [transaction_id]
          referential:
            - column: status
              allowed_values: [ACTIVE, CLOSED, PENDING]
          cross_column_rules:
            - if_col: status
              if_value: CLOSED
              then_col: close_date
              then_condition: not_null
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        integ_cfg = config.get("validation", {}).get("integrity", {})
        self.check_duplicates: bool = integ_cfg.get("check_duplicates", True)
        self.id_columns: List[Any] = integ_cfg.get("id_columns", [])
        self.referential_rules: List[Dict[str, Any]] = integ_cfg.get("referential", [])
        self.cross_column_rules: List[Dict[str, Any]] = integ_cfg.get("cross_column_rules", [])

    def check(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run all integrity checks.

        Returns:
            List of error/warning dicts with keys:
            ``column``, ``severity``, ``type``, ``message``.
        """
        errors: List[Dict[str, Any]] = []

        if self.check_duplicates:
            self._check_full_duplicates(df, errors)

        self._check_id_uniqueness(df, errors)
        self._check_referential(df, errors)
        self._check_cross_column(df, errors)
        self._check_cardinality(df, errors)  # Fix 7

        return errors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_full_duplicates(
        self, df: pd.DataFrame, errors: List[Dict[str, Any]]
    ) -> None:
        """Reports completely duplicate rows (all columns identical)."""
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            errors.append({
                "column": "ALL",
                "severity": "WARNING",
                "type": "DUPLICATE_ROWS",
                "message": (
                    f"{dup_count} fully duplicate row(s) detected. "
                    "Consider deduplication before model training."
                ),
            })
            logger.warning("IntegrityChecker: %d duplicate row(s) found.", dup_count)

    def _check_id_uniqueness(
        self, df: pd.DataFrame, errors: List[Dict[str, Any]]
    ) -> None:
        """Validates ID columns contain no duplicates (supports compound keys)."""
        for key_def in self.id_columns:
            cols: List[str] = [key_def] if isinstance(key_def, str) else list(key_def)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                logger.warning(
                    "ID uniqueness check: column(s) %s not in DataFrame — skipped.", missing
                )
                continue

            dup_count = df[cols].duplicated(keep=False).sum()
            if dup_count > 0:
                label = cols[0] if len(cols) == 1 else f"({', '.join(cols)})"
                errors.append({
                    "column": label,
                    "severity": "ERROR",
                    "type": "ID_UNIQUENESS_VIOLATION",
                    "message": (
                        f"ID column(s) {label} contain {dup_count} duplicate value(s). "
                        "Each record must have a unique identifier."
                    ),
                })
                logger.error("ID uniqueness violation on %s: %d duplicate(s).", label, dup_count)

    def _check_referential(
        self, df: pd.DataFrame, errors: List[Dict[str, Any]]
    ) -> None:
        """
        Checks that column values belong to a declared allowed set.
        Equivalent to a soft FOREIGN KEY constraint.
        """
        for rule in self.referential_rules:
            col = rule.get("column")
            allowed = set(rule.get("allowed_values", []))
            if not col or not allowed or col not in df.columns:
                continue

            series = df[col].dropna()
            invalid = series[~series.isin(allowed)]
            if not invalid.empty:
                sample = invalid.unique()[:5].tolist()
                errors.append({
                    "column": col,
                    "severity": "ERROR",
                    "type": "REFERENTIAL_INTEGRITY_VIOLATION",
                    "message": (
                        f"Column '{col}': {len(invalid)} value(s) not in allowed set "
                        f"{sorted(allowed)}. "
                        f"Sample invalid: {sample}"
                    ),
                })

    def _check_cross_column(
        self, df: pd.DataFrame, errors: List[Dict[str, Any]]
    ) -> None:
        """
        Evaluates conditional cross-column consistency rules.

        Rule format:
          if_col:        column to filter on
          if_value:      value that triggers the rule
          then_col:      column to validate
          then_condition: "not_null" | "is_null" | "positive" | "non_negative"
        """
        for rule in self.cross_column_rules:
            if_col   = rule.get("if_col")
            if_value = rule.get("if_value")
            then_col = rule.get("then_col")
            condition = rule.get("then_condition", "not_null")

            if not all([if_col, then_col]):
                continue
            if if_col not in df.columns or then_col not in df.columns:
                continue

            subset = df[df[if_col] == if_value]
            if subset.empty:
                continue

            then_series = subset[then_col]

            if condition == "not_null":
                bad = then_series.isnull().sum()
                cond_desc = f"not null when {if_col}={if_value!r}"
            elif condition == "is_null":
                bad = then_series.notnull().sum()
                cond_desc = f"null when {if_col}={if_value!r}"
            elif condition == "positive":
                bad = (then_series.dropna() <= 0).sum()
                cond_desc = f"positive when {if_col}={if_value!r}"
            elif condition == "non_negative":
                bad = (then_series.dropna() < 0).sum()
                cond_desc = f"non-negative when {if_col}={if_value!r}"
            else:
                logger.warning(
                    "CrossColumnRule: unknown condition '%s' — skipped.", condition
                )
                continue

            if bad > 0:
                errors.append({
                    "column": then_col,
                    "severity": "ERROR",
                    "type": "CROSS_COLUMN_CONSISTENCY_VIOLATION",
                    "message": (
                        f"Column '{then_col}' must be {cond_desc}, "
                        f"but {bad} row(s) violate this rule."
                    ),
                })

    # Fix 7 — Cardinality / constant-column detection
    def _check_cardinality(
        self, df: pd.DataFrame, errors: List[Dict[str, Any]]
    ) -> None:
        """
        Detect columns with zero or near-zero information content.

        Three categories:
          - Constant column:      nunique == 1 → WARNING (zero ML signal)
          - Near-constant column: top_value_freq > 99% → WARNING (model bias risk)
          - Fully-unique column:  nunique == nrows AND not a declared ID → WARNING
                                  (model will memorize, not generalize)
        """
        n = len(df)
        id_cols = set()
        for key_def in self.id_columns:
            if isinstance(key_def, str):
                id_cols.add(key_def)
            else:
                id_cols.update(key_def)

        for col in df.columns:
            try:
                n_unique = df[col].nunique(dropna=True)

                # ── Constant column (zero information) ───────────────────────
                if n_unique <= 1:
                    errors.append({
                        "column": col,
                        "severity": "WARNING",
                        "type": "CONSTANT_COLUMN",
                        "message": (
                            f"Column '{col}' has only {n_unique} unique value(s) — "
                            "constant column carries zero information for ML. "
                            "Consider dropping it."
                        ),
                    })
                    logger.warning("Cardinality: constant column '%s'", col)
                    continue

                # ── Near-constant column (potential model bias) ───────────────
                top_freq = float(df[col].value_counts(normalize=True, dropna=True).iloc[0])
                if top_freq > 0.99:
                    top_val = df[col].value_counts(dropna=True).index[0]
                    errors.append({
                        "column": col,
                        "severity": "WARNING",
                        "type": "NEAR_CONSTANT_COLUMN",
                        "message": (
                            f"Column '{col}' has {top_freq:.1%} rows with value "
                            f"'{top_val}' — near-constant column may bias models. "
                            "Consider dropping or using as a filter."
                        ),
                    })
                    logger.warning(
                        "Cardinality: near-constant '%s' (top_freq=%.1%)", col, top_freq
                    )

                # ── Fully-unique non-ID column (memorization risk) ────────────
                elif n > 0 and n_unique == n and col not in id_cols:
                    errors.append({
                        "column": col,
                        "severity": "WARNING",
                        "type": "FULLY_UNIQUE_COLUMN",
                        "message": (
                            f"Column '{col}' has {n_unique} unique values across "
                            f"{n} rows (100% unique) and is not declared as an ID column. "
                            "Models will memorize this column without generalizing."
                        ),
                    })
                    logger.warning(
                        "Cardinality: fully-unique non-ID column '%s'", col
                    )

            except Exception as exc:  # noqa: BLE001
                logger.debug("Cardinality check failed for column '%s': %s", col, exc)

