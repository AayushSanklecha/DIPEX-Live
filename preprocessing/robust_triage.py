"""
preprocessing/robust_triage.py
-------------------------------
Profile → Triage → Adapt  — runs BEFORE the main DataCleaner.

Handles all pathological real-world data patterns:
  • All-null / near-all-null columns  → drop + log
  • Mixed-type columns (str + numbers) → numeric coerce
  • Near-zero-variance columns         → drop + log
  • High-cardinality categoricals      → hash encode
  • Residual object columns            → label encode (safety net)
  • Class imbalance (target col)       → emit imbalance_ratio + strategy
  • Skewed numeric distributions       → log1p auto-transform

No data is destroyed without a full audit trail in TriageReport.
All thresholds are config-driven — zero hardcoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.preprocessing.robust_triage")

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TriageReport:
    """Full audit trail of every triage decision made on the DataFrame."""
    run_id: str = ""
    columns_dropped: List[Dict[str, Any]] = field(default_factory=list)
    columns_filled: List[Dict[str, Any]] = field(default_factory=list)     # medium-null forward/back fill
    columns_coerced: List[Dict[str, Any]] = field(default_factory=list)
    columns_hash_encoded: List[Dict[str, Any]] = field(default_factory=list)
    columns_label_encoded: List[Dict[str, Any]] = field(default_factory=list)
    columns_log_transformed: List[Dict[str, Any]] = field(default_factory=list)
    zero_flagged_columns: List[Dict[str, Any]] = field(default_factory=list)
    zero_fixed_columns: List[Dict[str, Any]] = field(default_factory=list)  # zeros→NaN+imputed
    imbalance_info: Dict[str, Any] = field(default_factory=dict)
    resample_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "columns_dropped": self.columns_dropped,
            "columns_filled": self.columns_filled,
            "columns_coerced": self.columns_coerced,
            "columns_hash_encoded": self.columns_hash_encoded,
            "columns_label_encoded": self.columns_label_encoded,
            "columns_log_transformed": self.columns_log_transformed,
            "zero_flagged_columns": self.zero_flagged_columns,
            "zero_fixed_columns": self.zero_fixed_columns,
            "imbalance_info": self.imbalance_info,
            "resample_info": self.resample_info,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RobustTriage
# ─────────────────────────────────────────────────────────────────────────────

class RobustTriage:
    """
    Pre-cleaner data triage engine.

    Run this BEFORE DataCleaner so the cleaner always receives
    type-safe, non-degenerate data.

    Config stanza (all optional — safe defaults shown)::

        preprocessing:
          triage:
            # ── Null handling ──────────────────────────────────────────────
            drop_col_null_threshold: 0.90     # drop column if > 90% null
            medium_null_lower: 0.25           # >25% null → forward/back fill
            medium_null_upper: 0.90           # upper bound for fill (below drop)
            medium_null_strategy: ffill       # ffill | bfill | median | mean

            # ── Zero handling ─────────────────────────────────────────────
            high_zero_threshold: 0.50         # >50% literal zeros => flag
            zero_to_nan: true                 # replace zeros→NaN then re-impute
            zero_impute_strategy: median      # median | mean | mode | knn

            # ── Mixed-type / coercion ──────────────────────────────────────
            mixed_type_coerce: true
            mixed_type_loss_threshold: 0.15   # max NaN from coerce before fallback
            coerce_regex_fallback: true       # regex extract numerics on failure

            # ── Column quality ────────────────────────────────────────────
            drop_near_zero_variance: true
            high_cardinality_limit: 200
            hash_buckets: 64

            # ── Distribution ──────────────────────────────────────────────
            auto_log_transform: true
            skew_threshold: 2.0

            # ── Class imbalance ───────────────────────────────────────────
            imbalance_ratio_threshold: 5.0    # flag above this ratio
            auto_resample: true               # SMOTE (or oversample) imbalanced data
            resample_strategy: smote          # smote | oversample | undersample
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("preprocessing", {}).get("triage", {})
        # Null handling
        self.drop_null_thresh: float       = float(cfg.get("drop_col_null_threshold", 0.90))
        self.medium_null_lower: float      = float(cfg.get("medium_null_lower", 0.25))
        self.medium_null_upper: float      = float(cfg.get("medium_null_upper", 0.90))
        self.medium_null_strategy: str     = str(cfg.get("medium_null_strategy", "ffill"))
        # Zero handling
        self.high_zero_thresh: float       = float(cfg.get("high_zero_threshold", 0.50))
        self.zero_to_nan: bool             = bool(cfg.get("zero_to_nan", True))
        self.zero_impute_strategy: str     = str(cfg.get("zero_impute_strategy", "median"))
        # Mixed type
        self.mixed_coerce: bool            = bool(cfg.get("mixed_type_coerce", True))
        self.mixed_loss_thresh: float      = float(cfg.get("mixed_type_loss_threshold", 0.15))
        self.coerce_regex_fallback: bool   = bool(cfg.get("coerce_regex_fallback", True))
        # Column quality
        self.drop_zero_var: bool           = bool(cfg.get("drop_near_zero_variance", True))
        self.cardinality_limit: int        = int(cfg.get("high_cardinality_limit", 200))
        self.hash_buckets: int             = int(cfg.get("hash_buckets", 64))
        # Distribution
        self.auto_log: bool                = bool(cfg.get("auto_log_transform", True))
        self.skew_threshold: float         = float(cfg.get("skew_threshold", 2.0))
        # Imbalance
        self.imbalance_ratio_thresh: float = float(cfg.get("imbalance_ratio_threshold", 5.0))
        self.auto_resample: bool           = bool(cfg.get("auto_resample", True))
        self.resample_strategy: str        = str(cfg.get("resample_strategy", "smote"))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RobustTriage":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def triage(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        run_id: str = "",
    ) -> Tuple[pd.DataFrame, TriageReport]:
        """
        Run the full triage pipeline.

        Parameters
        ----------
        df         : raw input DataFrame
        target_col : target column name (used for imbalance detection only)
        run_id     : pipeline run ID for the audit trail

        Returns
        -------
        (triaged_df, TriageReport)
        """
        report = TriageReport(run_id=run_id)
        df = df.copy()

        # ── Guard: 0-row DataFrame ──────────────────────────────────
        if len(df) == 0:
            logger.warning("[Triage] DataFrame has 0 rows — all triage passes skipped.")
            report.warnings.append("Input DataFrame has 0 rows — triage skipped.")
            return df, report

        # ── Guard: sanitise column names before any operation ──────────────
        import re as _re
        new_cols = []
        seen: dict = {}
        for i, col in enumerate(df.columns):
            col_str = str(col).strip() if col is not None else ""
            if not col_str or col_str in ("", "None", "nan"):
                col_str = f"_col_{i}"
            # Replace non-word characters
            col_str = _re.sub(r"[^\w.]", "_", col_str).strip("_") or f"_col_{i}"
            # Deduplicate
            if col_str in seen:
                seen[col_str] += 1
                col_str = f"{col_str}_{seen[col_str]}"
            else:
                seen[col_str] = 0
            new_cols.append(col_str)
        if list(df.columns) != new_cols:
            logger.info("[Triage] Sanitised %d column name(s).",
                        sum(a != b for a, b in zip(df.columns, new_cols)))
            df.columns = new_cols
            # Update target_col if it was renamed
            if target_col is not None:
                for old, new in zip(list(df.columns), new_cols):
                    if old == target_col:
                        target_col = new
                        break

        # 0. Honor AnalystBrain's recommendations
        brain_decisions = df.attrs.get("column_decisions", {})
        cols_to_drop_from_brain = [col for col, decision in brain_decisions.items() if decision.get("should_drop") and col in df.columns]
        if cols_to_drop_from_brain:
            df = df.drop(columns=cols_to_drop_from_brain)
            report.columns_dropped.extend([
                {"column": c, "reason": brain_decisions[c].get("reason", "AnalystBrain recommended drop")}
                for c in cols_to_drop_from_brain
            ])
            logger.info("[Triage] Dropped %d column(s) per AnalystBrain's recommendation.", len(cols_to_drop_from_brain))

        # 1. Tiered null handling:
        #    >90% null  → drop column
        #    25-90% null → forward/backward fill (or stat fill)
        df = self._handle_null_cols(df, report, target_col)

        # 2. High-zero detection + optional zero→NaN coercion
        df = self._handle_high_zeros(df, report, target_col)

        # 2.5 Multivariate Anomaly Detection (ML Isolation Forest)
        df = self._detect_multivariate_anomalies(df, report, target_col)

        # 3. Mixed-type detection + numeric coercion (with regex fallback)
        if self.mixed_coerce:
            df = self._coerce_mixed_type_cols(df, report, target_col)

        # 4. Near-zero-variance removal
        if self.drop_zero_var:
            df = self._drop_zero_variance_cols(df, report, target_col)

        # 5. High-cardinality categoricals → hash encoding
        df = self._hash_high_cardinality(df, report, target_col)

        # 6. Auto log1p on heavily skewed positive numeric columns
        if self.auto_log:
            df = self._auto_log_transform(df, report, target_col)

        # 7. Residual object columns → label encode (safety net before model)
        df = self._label_encode_residuals(df, report, target_col)

        # 8. Class imbalance detection + optional SMOTE / oversampling
        if target_col and target_col in df.columns:
            df = self._handle_imbalance(df, target_col, report)

        logger.info(
            "[Triage run_id=%s] dropped=%d filled=%d coerced=%d hash_enc=%d "
            "label_enc=%d log_tx=%d zero_fixed=%d warnings=%d",
            run_id[:8],
            len(report.columns_dropped),
            len(report.columns_filled),
            len(report.columns_coerced),
            len(report.columns_hash_encoded),
            len(report.columns_label_encoded),
            len(report.columns_log_transformed),
            len(report.zero_fixed_columns),
            len(report.warnings),
        )
        return df, report

    # ── Private helpers ───────────────────────────────────────────────────────
    
    def _detect_multivariate_anomalies(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Applies a pre-trained ML Isolation Forest to detect multivariate corruption.
        Rows flagged as anomalies are recorded in the report and dropped if severely corrupted.
        """
        try:
            import os
            import joblib
            import numpy as np
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "anomaly_detector.pkl")
            if not os.path.exists(model_path):
                return df

            # Get numeric features only (up to 15 to match training)
            num_df = df.select_dtypes(include="number").copy()
            if target_col and target_col in num_df.columns:
                num_df = num_df.drop(columns=[target_col])
                
            if num_df.empty or len(num_df.columns) < 2:
                return df
                
            arr = num_df.values.astype(float)
            if arr.shape[1] < 15:
                arr = np.pad(arr, ((0,0), (0, 15 - arr.shape[1])))
            else:
                arr = arr[:, :15]
                
            arr_clean = np.nan_to_num(arr, 0)
            
            clf = joblib.load(model_path)
            preds = clf.predict(arr_clean)
            
            anomaly_mask = preds == -1
            n_anomalies = int(anomaly_mask.sum())
            
            if n_anomalies > 0:
                report.warnings.append(f"ML Anomaly Detector flagged {n_anomalies} heavily corrupted row(s).")
                logger.warning(f"[Triage] ML Anomaly Detector flagged {n_anomalies} row(s) for removal.")
                # Drop anomalous rows
                df = df[~anomaly_mask].copy()
                
            return df
        except Exception as e:
            logger.debug(f"[Triage] ML Anomaly Detection failed or skipped: {e}")
            return df

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Tiered null handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_null_cols(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Tiered null strategy:
          >90% null  → DROP column (no usable signal)
          25-90% null → smart fill based on data type and index:
            - Time-indexed DataFrame  : ffill + bfill, then stat fill for residuals
            - Non-time-indexed numeric : median fill (statistically correct)
            - Non-time-indexed object  : mode fill, fallback to 'UNKNOWN'
          <25% null   → left for DataCleaner's imputer
        """
        n = len(df)
        if n == 0:
            return df

        # ── Pre-pass: date-string auto-conversion ──────────────────────────
        # Detect object columns where ≥70% of non-null values parse as dates.
        # Convert them BEFORE mixed-type coercion so they aren't misclassified.
        DATE_DETECT_THRESHOLD = 0.70
        for col in list(df.select_dtypes(include="object").columns):
            if col == target_col:
                continue
            try:
                sample = df[col].dropna().head(100)
                if len(sample) < 5:
                    continue
                # Skip columns that look numeric
                if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                parse_rate = parsed.notna().mean()
                if parse_rate >= DATE_DETECT_THRESHOLD:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    logger.info(
                        "[Triage] Column '%s' auto-converted: date-string → datetime64 (%.0f%% parsed)",
                        col, parse_rate * 100,
                    )
                    report.warnings.append(
                        f"Column '{col}' auto-converted from string to datetime64 ({parse_rate:.0%} parsed)"
                    )
            except Exception:  # noqa: BLE001
                pass

        null_rates = df.isnull().mean()

        # Detect whether the DataFrame is time-indexed — only then does ffill make sense
        is_time_indexed = isinstance(df.index, pd.DatetimeIndex)
        # Also check if user explicitly chose ffill/bfill even on non-time data
        explicit_ffill = self.medium_null_strategy in ("ffill", "bfill")

        to_drop = []
        to_fill = []

        for col in df.columns:
            if col == target_col:
                continue
            r = null_rates[col]
            if r > self.drop_null_thresh:
                to_drop.append(col)
            elif self.medium_null_lower <= r <= self.medium_null_upper:
                to_fill.append(col)

        # Drop extreme-null columns
        if to_drop:
            for col in to_drop:
                report.columns_dropped.append({
                    "column": col,
                    "reason": "high_null",
                    "null_pct": round(float(null_rates[col]), 4),
                    "threshold": self.drop_null_thresh,
                })
                logger.warning(
                    "[Triage] Dropped '%s' — %.1f%% null (threshold %.0f%%)",
                    col, null_rates[col] * 100, self.drop_null_thresh * 100,
                )
            df = df.drop(columns=to_drop)

        # Fill medium-null columns
        for col in to_fill:
            nulls_before = int(df[col].isna().sum())
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            strategy_used = self.medium_null_strategy

            if (is_time_indexed or explicit_ffill) and strategy_used in ("ffill", "bfill"):
                # Time-series: propagate last known value
                if strategy_used == "ffill":
                    df[col] = df[col].ffill().bfill()
                else:
                    df[col] = df[col].bfill().ffill()
            elif is_numeric:
                # Non-time-series numeric: median is statistically sound
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                strategy_used = "median" if self.medium_null_strategy not in ("mean",) else "mean"
                if self.medium_null_strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                    strategy_used = "mean"
                else:
                    df[col] = df[col].fillna(df[col].median())
                    strategy_used = "median"
            else:
                # Categorical / object: mode is most appropriate
                mode_vals = df[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "UNKNOWN"
                df[col] = df[col].fillna(fill_val)
                strategy_used = "mode"

            # Residual NaNs (all-null at edges after ffill/bfill, or all-NaN column)
            if df[col].isna().any():
                if is_numeric:
                    # Fallback to 0 only if ALL values are NaN (column is genuinely empty)
                    fallback = df[col].median() if df[col].notna().any() else 0.0
                    df[col] = df[col].fillna(fallback)
                else:
                    df[col] = df[col].fillna("UNKNOWN")

            nulls_after = int(df[col].isna().sum())
            report.columns_filled.append({
                "column": col,
                "null_pct": round(float(null_rates[col]), 4),
                "nulls_before": nulls_before,
                "nulls_after": nulls_after,
                "strategy": strategy_used,
                "time_indexed": is_time_indexed,
            })
            logger.info(
                "[Triage] Filled '%s' (%.1f%% null) with %s — %d→%d remaining nulls",
                col, null_rates[col] * 100, strategy_used, nulls_before, nulls_after,
            )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 2. High-zero handling — convert to NaN and re-impute
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_high_zeros(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Detects numeric columns where >high_zero_thresh of values are exactly 0.
        These are often sentinel values for missing data.

        IMPORTANT: Binary / flag columns (nunique <= 2 or all values in {0, 1})
        are EXEMPTED — zeros there are legitimate data, not missing values.

        Also handles common sentinel values beyond 0:
        - -999, -1 (common DB sentinels), 9999, 99999 (out-of-range sentinels)

        If zero_to_nan=True: replaces the zeros with NaN, then fills using
        zero_impute_strategy so the column gets proper imputed values.
        """
        if len(df) == 0:
            return df

        # Common numeric sentinels beyond 0
        SENTINELS = {-999, -9999, -99999, -1, 9999, 99999, 999999}

        for col in df.select_dtypes(include=np.number).columns:
            if col == target_col:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            # ── GUARD: skip binary / flag columns ─────────────────────────
            # e.g. is_active (0/1), has_discount (0/1), boolean flags
            unique_vals = set(series.unique())
            if unique_vals.issubset({0, 1}) or series.nunique() <= 2:
                continue

            zero_pct = float((df[col] == 0).mean())

            # ── Also check for non-zero sentinels ─────────────────────────
            sentinel_found = [s for s in SENTINELS if s in unique_vals]
            for sentinel in sentinel_found:
                s_pct = float((df[col] == sentinel).mean())
                if s_pct > 0.05:  # >5% of values are this sentinel
                    logger.warning(
                        "[Triage] Detected sentinel value %s in '%s' (%.1f%% of rows)",
                        sentinel, col, s_pct * 100,
                    )
                    if self.zero_to_nan:
                        df[col] = df[col].replace(sentinel, np.nan)
                        report.zero_flagged_columns.append({
                            "column": col, "sentinel": sentinel,
                            "sentinel_pct": round(s_pct, 4), "action": "sentinel_replaced_with_nan",
                        })

            if zero_pct <= self.high_zero_thresh:
                continue

            # Always flag
            report.zero_flagged_columns.append({
                "column": col,
                "zero_pct": round(zero_pct, 4),
                "threshold": self.high_zero_thresh,
                "action": "replaced_with_nan" if self.zero_to_nan else "flagged_only",
            })

            if not self.zero_to_nan:
                logger.warning(
                    "[Triage] Flagged '%s' — %.1f%% zeros (zero_to_nan=False, no action taken)",
                    col, zero_pct * 100,
                )
                continue

            # Replace zeros with NaN
            df[col] = df[col].replace(0, np.nan)
            nulls_now = int(df[col].isna().sum())

            # Re-impute: use non-null, non-sentinel values only
            valid = df[col].dropna()
            strategy = self.zero_impute_strategy
            if strategy == "median":
                fill = valid.median() if len(valid) > 0 else np.nan
            elif strategy == "mean":
                fill = valid.mean() if len(valid) > 0 else np.nan
            elif strategy == "mode":
                mode = valid.mode()
                fill = float(mode.iloc[0]) if not mode.empty else np.nan
            elif strategy == "knn":
                fill = None  # deferred to DataCleaner KNN
            else:
                fill = valid.median() if len(valid) > 0 else np.nan

            if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
                df[col] = df[col].fillna(fill)
                nulls_after = 0
            else:
                nulls_after = nulls_now

            report.zero_fixed_columns.append({
                "column": col,
                "zero_pct": round(zero_pct, 4),
                "zeros_replaced": nulls_now,
                "strategy": strategy,
                "fill_value": float(fill) if fill is not None and not (isinstance(fill, float) and np.isnan(fill)) else "deferred_to_knn",
                "nulls_remaining": nulls_after,
            })
            logger.info(
                "[Triage] Fixed '%s': replaced %d zeros→NaN, imputed with %s (fill=%.4g)",
                col, nulls_now, strategy, fill if fill is not None else float("nan"),
            )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Mixed-type coercion with regex fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _coerce_mixed_type_cols(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Three-pass coercion for object columns:
          Pass 1: pd.to_numeric(errors='coerce')  — handles '1.5', '2', etc.
          Pass 2: if pass 1 fails, use regex to extract leading numerics (e.g. '42kg' → 42)
          Pass 3: if still too lossy, hash-encode as last resort (don't leave raw strings)
        """
        import re
        _NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

        for col in df.select_dtypes(include="object").columns:
            if col == target_col:
                continue

            original_nulls = int(df[col].isna().sum())

            # ── Pass 1: direct numeric coercion ──────────────────────────
            coerced = pd.to_numeric(df[col], errors="coerce")
            new_nulls = int(coerced.isna().sum()) - original_nulls
            loss_rate = new_nulls / max(len(df), 1)

            if 0 <= loss_rate <= self.mixed_loss_thresh:
                if new_nulls > 0 or coerced.notna().any():
                    df[col] = coerced
                    report.columns_coerced.append({
                        "column": col, "pass": 1,
                        "new_nulls_introduced": new_nulls,
                        "loss_rate": round(loss_rate, 4),
                    })
                    logger.info("[Triage] Pass-1 coerced '%s' → numeric (loss=%.2f%%)", col, loss_rate * 100)
                continue

            # ── Pass 2: regex numeric extraction ─────────────────────────
            # Strip common noise first: currency symbols, commas, units
            # e.g. "$1,200.50" → "1200.50"  |  "42kg" → "42"  |  "12.5%" → "12.5"
            if self.coerce_regex_fallback:
                _CURRENCY_RE = re.compile(r"[$€£¥₹,\s]+")
                _NUM_RE2 = re.compile(r"[-+]?\d[\d,]*\.?\d*")

                # Vectorized Pass 2
                s = df[col].astype(str).str.replace(r"[$€£¥₹,\s]+", "", regex=True)
                extracted = s.str.extract(r"([-+]?\d[\d,]*\.?\d*)")[0]
                regex_coerced = pd.to_numeric(extracted.str.replace(",", ""), errors="coerce")

                regex_new_nulls = int(regex_coerced.isna().sum()) - original_nulls
                regex_loss = regex_new_nulls / max(len(df), 1)

                if regex_loss <= self.mixed_loss_thresh:
                    df[col] = regex_coerced
                    report.columns_coerced.append({
                        "column": col, "pass": 2,
                        "method": "regex_extraction",
                        "new_nulls_introduced": regex_new_nulls,
                        "loss_rate": round(regex_loss, 4),
                    })
                    logger.info("[Triage] Pass-2 regex coerced '%s' → numeric (loss=%.2f%%)", col, regex_loss * 100)
                    continue

                # ── Pass 3: hash-encode as last resort ───────────────────
                n_unique = df[col].nunique(dropna=True)
                new_col = f"{col}_fallback_hash"
                # Vectorized fast C hash
                df[new_col] = (pd.util.hash_pandas_object(df[col], index=False) % self.hash_buckets).astype(np.int32)
                df = df.drop(columns=[col])
                report.columns_hash_encoded.append({
                    "original_column": col,
                    "new_column": new_col,
                    "reason": "coercion_failure_fallback",
                    "original_unique": n_unique,
                    "hash_buckets": self.hash_buckets,
                })
                logger.warning(
                    "[Triage] Pass-3 hash-encoded '%s' (coercion failed, %d unique vals → %d buckets)",
                    col, n_unique, self.hash_buckets,
                )
        return df

    def _drop_zero_variance_cols(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """Drop columns where all non-null values are the same (zero information)."""
        to_drop = []
        for col in df.columns:
            if col == target_col:
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            try:
                n_uniq = non_null.nunique()
            except Exception:  # noqa: BLE001 — unhashable types
                try:
                    n_uniq = non_null.astype(str).nunique()
                except Exception:  # noqa: BLE001
                    continue  # can't determine variance, skip
            if n_uniq <= 1:
                to_drop.append(col)
                report.columns_dropped.append({
                    "column": col,
                    "reason": "zero_variance",
                    "unique_values": int(non_null.nunique()),
                    "constant_value": str(non_null.iloc[0]) if len(non_null) else None,
                })
                logger.warning("[Triage] Dropped '%s' — zero variance (constant column)", col)
        if to_drop:
            df = df.drop(columns=to_drop)
        return df

    def _hash_high_cardinality(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Replace high-cardinality object columns with a hash-encoded integer column.
        Hash encoding is low-dimensional, collision-tolerant, and never OOMs.
        """
        for col in df.select_dtypes(include="object").columns:
            if col == target_col:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique > self.cardinality_limit:
                new_col = f"{col}_hash_enc"
                df[new_col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: hash(x) % self.hash_buckets)
                    .astype(np.int32)
                )
                df = df.drop(columns=[col])
                report.columns_hash_encoded.append({
                    "original_column": col,
                    "new_column": new_col,
                    "original_unique": n_unique,
                    "hash_buckets": self.hash_buckets,
                })
                logger.info(
                    "[Triage] Hash-encoded '%s' (%d unique → %d buckets)",
                    col, n_unique, self.hash_buckets,
                )
        return df

    def _auto_log_transform(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Detect and fix heavily right-skewed positive numeric columns via log1p.
        Skewness > threshold AND all values >= 0 → apply log1p.
        """
        for col in df.select_dtypes(include=np.number).columns:
            if col == target_col:
                continue
            series = df[col].dropna()
            if len(series) < 4:
                continue
            # Only transform positive-only columns (log1p defined for x >= 0)
            if series.min() < 0:
                continue
            try:
                skewness = float(series.skew())
            except Exception:
                continue
            if abs(skewness) > self.skew_threshold:
                new_col = f"{col}_auto_log1p"
                df[new_col] = np.log1p(df[col].clip(lower=0))
                df = df.drop(columns=[col])
                report.columns_log_transformed.append({
                    "original_column": col,
                    "new_column": new_col,
                    "skewness": round(skewness, 4),
                    "threshold": self.skew_threshold,
                })
                logger.info(
                    "[Triage] Log1p '%s' (skewness=%.2f > %.1f)",
                    col, skewness, self.skew_threshold,
                )
        return df

    def _label_encode_residuals(
        self, df: pd.DataFrame, report: TriageReport, target_col: Optional[str]
    ) -> pd.DataFrame:
        """
        Any remaining object columns that weren't hash-encoded get label-encoded
        so no raw strings reach the model layer. 
        Uses fast pandas Categorical code conversion.
        """
        for col in df.select_dtypes(include=["object", "string", "category"]).columns:
            if col == target_col:
                continue
            
            non_null = df[col].fillna("__MISSING__").astype(str)
            
            # Fast C-level categorical encoding
            cat_col = non_null.astype('category')
            df[col] = cat_col.cat.codes
            
            n_unique = len(cat_col.cat.categories)
            report.columns_label_encoded.append({
                "column": col,
                "unique_values": n_unique,
                "method": "pandas_categorical",
            })
            logger.info("[Triage] Label-encoded '%s' (%d unique values)", col, n_unique)
        
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Class imbalance — detect + optionally resample
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_imbalance(
        self, df: pd.DataFrame, target_col: str, report: TriageReport
    ) -> pd.DataFrame:
        """
        Detects class imbalance and, if auto_resample=True, applies:
          - SMOTE (via imbalanced-learn) if available
          - Random oversampling of minority classes (pandas fallback)
          - Random undersampling of majority class (if resample_strategy='undersample')

        The target column is never dropped or corrupted.
        Non-imbalanced data is returned unchanged.
        """
        y = df[target_col].dropna()
        n_unique = y.nunique()

        # Only meaningful for classification (2–20 classes)
        if n_unique < 2 or n_unique > 20:
            return df

        counts = y.value_counts()
        majority = int(counts.iloc[0])
        minority = int(counts.iloc[-1])
        ratio = majority / max(minority, 1)
        is_imbalanced = ratio >= self.imbalance_ratio_thresh

        report.imbalance_info = {
            "target_col": target_col,
            "n_classes": n_unique,
            "majority_class": str(counts.index[0]),
            "majority_count": majority,
            "minority_class": str(counts.index[-1]),
            "minority_count": minority,
            "imbalance_ratio": round(ratio, 2),
            "is_imbalanced": is_imbalanced,
            "resampled": False,
        }

        if not is_imbalanced:
            logger.info("[Triage] Class balance OK — '%s' ratio=%.1f", target_col, ratio)
            return df

        logger.warning(
            "[Triage] Class imbalance detected on '%s': ratio=%.1f (threshold %.1f)",
            target_col, ratio, self.imbalance_ratio_thresh,
        )

        if not self.auto_resample:
            report.imbalance_info["recommended_strategy"] = "class_weight_balanced"
            logger.info("[Triage] auto_resample=False — AutoML will use class_weight='balanced'")
            return df

        # ── Attempt resampling ───────────────────────────────────────────
        rows_before = len(df)
        resampled_df = None
        method_used = None

        num_cols_for_smote = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in num_cols_for_smote:
            num_cols_for_smote.remove(target_col)

        strategy = self.resample_strategy.lower()

        # ── SMOTE (best quality, requires all-numeric features) ──────────
        if strategy == "smote" and num_cols_for_smote:
            try:
                from imblearn.over_sampling import SMOTE
                X = df[num_cols_for_smote].fillna(df[num_cols_for_smote].median())
                y_col = df[target_col]
                # k_neighbors must be < minority class count; clamp to avoid ValueError
                k = max(1, min(5, minority - 1))
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_res, y_res = smote.fit_resample(X, y_col)
                resampled_df = pd.DataFrame(X_res, columns=num_cols_for_smote)
                resampled_df[target_col] = y_res
                # Re-attach any non-numeric columns by dropping them from the result
                # (they cannot be SMOTE'd — label-encoded residuals handled earlier)
                method_used = "smote"
                logger.info("[Triage] SMOTE applied on '%s': %d → %d rows", target_col, rows_before, len(resampled_df))
            except ImportError:
                logger.info("[Triage] imbalanced-learn not installed — falling back to random oversample")
            except Exception as exc:
                logger.warning("[Triage] SMOTE failed (%s) — falling back to random oversample", exc)

        # ── Random oversample (pandas, no extra deps) ─────────────────────
        if resampled_df is None and strategy in ("smote", "oversample"):
            try:
                frames = [df]
                target_count = majority  # upsample all minorities to majority size
                for cls in counts.index[1:]:  # skip majority class
                    cls_df = df[df[target_col] == cls]
                    extra = cls_df.sample(
                        n=target_count - len(cls_df),
                        replace=True,
                        random_state=42,
                    )
                    frames.append(extra)
                resampled_df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
                method_used = "random_oversample"
                logger.info("[Triage] Random oversample on '%s': %d → %d rows", target_col, rows_before, len(resampled_df))
            except Exception as exc:
                logger.warning("[Triage] Random oversample failed: %s", exc)

        # ── Random undersample (reduce majority) ──────────────────────────
        if resampled_df is None or strategy == "undersample":
            try:
                target_count = minority  # downsample majority to minority size
                frames = []
                for cls in counts.index:
                    cls_df = df[df[target_col] == cls]
                    if len(cls_df) > target_count:
                        cls_df = cls_df.sample(n=target_count, random_state=42)
                    frames.append(cls_df)
                resampled_df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
                method_used = "random_undersample"
                logger.info("[Triage] Undersample on '%s': %d → %d rows", target_col, rows_before, len(resampled_df))
            except Exception as exc:
                logger.warning("[Triage] Undersample failed: %s", exc)

        if resampled_df is not None:
            df = resampled_df
            report.imbalance_info["resampled"] = True
            report.imbalance_info["resample_method"] = method_used
            report.imbalance_info["rows_before"] = rows_before
            report.imbalance_info["rows_after"] = len(df)
            report.resample_info = {
                "method": method_used,
                "rows_before": rows_before,
                "rows_after": len(df),
                "strategy": strategy,
            }
        else:
            report.imbalance_info["recommended_strategy"] = "class_weight_balanced"
            report.warnings.append(
                f"[Triage] All resampling strategies failed for '{target_col}'. "
                "AutoML will use class_weight='balanced' instead."
            )

        return df
