"""
preprocessing/cleaner.py
------------------------
Enterprise-grade data cleaning engine.

Handles:
  - Missing value imputation (mean / median / mode / constant / KNN)
  - Outlier capping (IQR fence / z-score)
  - Type coercion (schema-driven)
  - Duplicate row removal
  - Whitespace normalisation (string columns)
  - Date parsing and validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("dipex.preprocessing.cleaner")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleaningReport:
    """Immutable record of all cleaning operations applied."""
    run_id: str
    rows_before: int
    rows_after: int
    duplicates_removed: int
    imputation_log: List[Dict[str, Any]] = field(default_factory=list)
    capping_log: List[Dict[str, Any]] = field(default_factory=list)
    coercion_log: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "duplicates_removed": self.duplicates_removed,
            "imputation_log": self.imputation_log,
            "capping_log": self.capping_log,
            "coercion_log": self.coercion_log,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataCleaner
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Stateless-first data cleaning engine.

    Configuration (all optional, with safe defaults):
      imputation_strategy : 'mean' | 'median' | 'mode' | 'constant' | 'knn'
      imputation_constant : value used when strategy == 'constant'
      outlier_capping     : 'iqr' | 'zscore' | None
      iqr_multiplier      : float (default 1.5)
      zscore_threshold    : float (default 3.0)
      remove_duplicates   : bool (default True)
      type_coercions      : dict {col: dtype}  e.g. {'age': 'int', 'price': 'float'}
      date_columns        : list of columns to parse as datetime
      string_strip        : bool — strip leading/trailing whitespace from str cols
    """

    VALID_STRATEGIES = {"mean", "median", "mode", "constant", "knn", "iterative", "none"}

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        pre = cfg.get("preprocessing", {})
        self.imputation_strategy: str = pre.get("imputation_strategy", "median").lower()
        self.imputation_constant: Any = pre.get("imputation_constant", 0)
        self.outlier_capping: Optional[str] = pre.get("outlier_capping", "iqr")
        self.iqr_multiplier: float = float(pre.get("iqr_multiplier", 1.5))
        self.zscore_threshold: float = float(pre.get("zscore_threshold", 3.0))
        self.remove_duplicates: bool = bool(pre.get("remove_duplicates", True))
        self.type_coercions: Dict[str, str] = pre.get("type_coercions", {})
        self.date_columns: List[str] = pre.get("date_columns", [])
        self.string_strip: bool = bool(pre.get("string_strip", True))

        if self.imputation_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid imputation_strategy '{self.imputation_strategy}'. "
                f"Choose from {sorted(self.VALID_STRATEGIES)}."
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DataCleaner":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame, run_id: str = "") -> tuple[pd.DataFrame, CleaningReport]:
        """
        Apply the full cleaning pipeline in a non-destructive manner.

        Returns:
            (cleaned_df, CleaningReport)
        """
        report = CleaningReport(
            run_id=run_id,
            rows_before=len(df),
            rows_after=len(df),
            duplicates_removed=0,
        )
        df = df.copy()

        # 1. String whitespace normalisation
        if self.string_strip:
            df = self._strip_strings(df)

        # 2. Type coercion
        df = self._coerce_types(df, report)

        # 3. Date parsing
        df = self._parse_dates(df, report)

        # 4. Duplicate removal
        if self.remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            report.duplicates_removed = before - len(df)
            if report.duplicates_removed:
                logger.info("Removed %d duplicate rows.", report.duplicates_removed)

        # 5. Missing value imputation
        df = self._impute(df, report)

        # 6. Outlier capping
        if self.outlier_capping and self.outlier_capping.lower() != "none":
            df = self._cap_outliers(df, report)

        report.rows_after = len(df)
        return df, report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _strip_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _coerce_types(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        for col, dtype in self.type_coercions.items():
            if col not in df.columns:
                report.warnings.append(f"Type coercion: column '{col}' not found.")
                continue
            try:
                df[col] = df[col].astype(dtype)
                report.coercion_log.append({"column": col, "dtype": dtype, "status": "OK"})
            except (ValueError, TypeError) as exc:
                msg = f"Could not coerce '{col}' to {dtype}: {exc}"
                report.warnings.append(msg)
                report.coercion_log.append({"column": col, "dtype": dtype, "status": "FAILED", "error": str(exc)})
                logger.warning(msg)
        return df

    def _parse_dates(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        for col in self.date_columns:
            if col not in df.columns:
                report.warnings.append(f"Date parse: column '{col}' not found.")
                continue
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                nat_count = df[col].isna().sum()
                report.coercion_log.append({
                    "column": col, "dtype": "datetime64[ns]",
                    "status": "OK", "nat_count": int(nat_count),
                })
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(f"Date parse failed for '{col}': {exc}")
        return df

    def _impute(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        if self.imputation_strategy == "none":
            return df

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        # [ML] Route to ML imputer if requested or if > 10 % nulls
        if self.imputation_strategy == "iterative":
            return self._impute_iterative(df, num_cols, report)
        if self.imputation_strategy == "knn":
            return self._impute_knn(df, num_cols, report)

        for col in num_cols:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue
            if self.imputation_strategy == "mean":
                fill_val = df[col].mean()
            elif self.imputation_strategy == "median":
                fill_val = df[col].median()
            elif self.imputation_strategy == "mode":
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            else:  # constant
                fill_val = self.imputation_constant
            df[col] = df[col].fillna(fill_val)
            report.imputation_log.append({
                "column": col, "strategy": self.imputation_strategy,
                "fill_value": float(fill_val) if isinstance(fill_val, (np.floating, float)) else fill_val,
                "nulls_filled": int(null_count),
            })

        for col in cat_cols:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue
            if self.imputation_strategy == "mode":
                fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN"
            else:
                fill_val = str(self.imputation_constant) if self.imputation_strategy == "constant" else "UNKNOWN"
            df[col] = df[col].fillna(fill_val)
            report.imputation_log.append({
                "column": col, "strategy": self.imputation_strategy,
                "fill_value": fill_val, "nulls_filled": int(null_count),
            })

        return df

    def _impute_knn(
        self, df: pd.DataFrame, num_cols: List[str], report: CleaningReport
    ) -> pd.DataFrame:
        """KNN imputation for numeric columns only."""
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            report.warnings.append("KNN imputation skipped — scikit-learn not installed.")
            return df

        if not num_cols:
            return df

        imputer = KNNImputer(n_neighbors=5)
        null_counts_before = df[num_cols].isna().sum()
        df[num_cols] = imputer.fit_transform(df[num_cols])
        for col in num_cols:
            nulls = int(null_counts_before[col])
            if nulls:
                report.imputation_log.append({
                    "column": col, "strategy": "knn",
                    "fill_value": "knn_inferred", "nulls_filled": nulls,
                })
        return df

    def _impute_iterative(
        self, df: pd.DataFrame, num_cols: List[str], report: CleaningReport
    ) -> pd.DataFrame:
        """
        [ML] Iterative (MICE) imputation using sklearn IterativeImputer.
        Falls back to KNN if IterativeImputer is not available.
        """
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
        except ImportError:
            report.warnings.append("IterativeImputer unavailable — falling back to KNN.")
            return self._impute_knn(df, num_cols, report)

        if not num_cols:
            return df

        null_counts_before = df[num_cols].isna().sum()
        total_nulls = null_counts_before.sum()
        if total_nulls == 0:
            return df

        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            verbose=0,
            sample_posterior=False,
        )
        try:
            df[num_cols] = imputer.fit_transform(df[num_cols])
            for col in num_cols:
                nulls = int(null_counts_before[col])
                if nulls:
                    report.imputation_log.append({
                        "column": col, "strategy": "iterative_mice",
                        "fill_value": "iterative_inferred", "nulls_filled": nulls,
                    })
            logger.info("[ML] IterativeImputer: filled %d null(s) across %d columns.",
                        int(total_nulls), len([c for c in num_cols if null_counts_before[c] > 0]))
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"IterativeImputer failed ({exc}) — trying KNN fallback.")
            return self._impute_knn(df, num_cols, report)
        return df

    def _cap_outliers(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        method = self.outlier_capping.lower()

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
            elif method == "zscore":
                mean, std = series.mean(), series.std()
                if std == 0:
                    continue
                lower = mean - self.zscore_threshold * std
                upper = mean + self.zscore_threshold * std
            else:
                continue

            capped_low = int((df[col] < lower).sum())
            capped_high = int((df[col] > upper).sum())
            if capped_low + capped_high == 0:
                continue

            df[col] = df[col].clip(lower=lower, upper=upper)
            report.capping_log.append({
                "column": col, "method": method,
                "lower_fence": float(lower), "upper_fence": float(upper),
                "capped_low": capped_low, "capped_high": capped_high,
            })

        return df
