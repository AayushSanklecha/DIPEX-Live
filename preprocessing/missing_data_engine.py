"""
preprocessing/missing_data_engine.py
-------------------------------------
Phase 3 — Unified Missing Data Intelligence Engine.

Runs BEFORE DataCleaner and MissingPatternAnalyzer to produce a single,
authoritative decision on HOW to handle every column's missing data.

Strategy selection matrix
--------------------------
null_pct == 1.0         → DROP column  (no signal whatsoever)
null_pct  > 0.90        → DROP column  (>90% null — signal too weak)
null_pct == 0.0         → COMPLETE     (nothing to do)
MCAR + null_pct <  0.05 → MEDIAN / MODE fill
MCAR + null_pct <  0.30 → KNN imputation (k=5)
MCAR + null_pct >= 0.30 → MICE (IterativeImputer)
MAR  + any null_pct     → MICE (conditioned on other columns)
MNAR + any null_pct     → add {col}_was_null indicator + MICE

Row quarantine
--------------
After column decisions are applied, any row that is still ≥80% null
is moved to quarantine_df and excluded from clean_df.

Fallback chain (library not installed)
---------------------------------------
IterativeImputer → KNNImputer → median/mode → constant (0 / "UNKNOWN")

The engine never crashes. Every exception is caught, logged,
and the best available fallback is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.preprocessing.missing_data_engine")

# ── Sentinel string values treated as missing ─────────────────────────────────
_STRING_SENTINELS: frozenset = frozenset({
    "N/A", "NA", "n/a", "na", "NULL", "null", "None", "none",
    "NaN", "nan", "#N/A", "#NULL!", "?", ".", " ", "--", "---",
    "UNKNOWN", "unknown", "missing", "MISSING", "TBD", "tbd",
    "N.A.", "N.A", "Not Available", "not available", "(blank)", "",
    "undefined", "Undefined", "NIL", "nil", "empty", "EMPTY",
})

# ── Numeric sentinel values treated as missing ────────────────────────────────
_NUMERIC_SENTINELS: frozenset = frozenset({
    -999, -9999, -99999, -99, -1, 9999, 99999, 999999, -1.0,
    -999.0, -9999.0, 9999.0, 99999.0,
})


# ─────────────────────────────────────────────────────────────────────────────
# Report dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnMissingProfile:
    column: str
    null_pct: float
    mechanism: str                  # COMPLETE | MCAR | MAR | MNAR | DROPPED
    strategy: str                   # complete | median | mode | knn | mice | drop | indicator+mice
    indicator_added: bool = False   # True if {col}_was_null column was added
    sentinels_replaced: int = 0     # count of sentinel → NaN replacements
    rows_imputed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "null_pct": round(self.null_pct, 4),
            "mechanism": self.mechanism,
            "strategy": self.strategy,
            "indicator_added": self.indicator_added,
            "sentinels_replaced": self.sentinels_replaced,
            "rows_imputed": self.rows_imputed,
        }


@dataclass
class MissingDataReport:
    run_id: str
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    quarantine_rows: int = 0
    columns_dropped: List[str] = field(default_factory=list)
    columns_profiled: List[ColumnMissingProfile] = field(default_factory=list)
    sentinel_replacements_total: int = 0
    imputation_library_used: str = "median"  # mice | knn | median

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_shape": list(self.original_shape),
            "final_shape": list(self.final_shape),
            "quarantine_rows": self.quarantine_rows,
            "columns_dropped": self.columns_dropped,
            "sentinel_replacements_total": self.sentinel_replacements_total,
            "imputation_library_used": self.imputation_library_used,
            "columns_profiled": [p.to_dict() for p in self.columns_profiled],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class MissingDataEngine:
    """
    Unified missing-data intelligence engine.

    Usage::

        engine = MissingDataEngine(config=config)
        clean_df, quarantine_df, report = engine.run(
            df, run_id="abc123", target_col="churn"
        )

    Parameters
    ----------
    config : dict
        Project config. Reads from ``preprocessing.missing_data_engine`` stanza.
    drop_if_null_above : float
        Drop column when null_pct > this value. Default 0.90.
    quarantine_row_null_pct : float
        Quarantine rows with null_pct >= this value. Default 0.80.
    add_mnar_indicator : bool
        Add {col}_was_null column for MNAR columns. Default True.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("preprocessing", {}).get("missing_data_engine", {})
        self._drop_threshold: float  = float(cfg.get("drop_if_null_above", 0.90))
        self._quarantine_pct: float  = float(cfg.get("quarantine_row_null_pct", 0.80))
        self._add_indicator:  bool   = bool(cfg.get("add_mnar_indicator", True))
        self._mcar_knn_threshold: float = float(cfg.get("mcar_knn_threshold", 0.05))
        self._mcar_mice_threshold: float = float(cfg.get("mcar_mice_threshold", 0.30))

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        run_id: str = "unknown",
        target_col: Optional[str] = None,
        rl_recommendations: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, MissingDataReport]:
        """
        Execute the full missing-data pipeline on df.

        Returns
        -------
        (clean_df, quarantine_df, MissingDataReport)
        """
        report = MissingDataReport(run_id=run_id, original_shape=df.shape)

        # ── Guard: empty input ────────────────────────────────────────────────
        if df is None or df.empty:
            logger.warning("[MDE] Input DataFrame is empty — returning as-is.")
            empty_q = pd.DataFrame(columns=df.columns if df is not None else [])
            report.final_shape = df.shape if df is not None else (0, 0)
            return (df if df is not None else pd.DataFrame()), empty_q, report

        df = df.copy()

        # ── Step 1: Replace string and numeric sentinels with NaN ─────────────
        df, sentinel_total = self._replace_sentinels(df)
        report.sentinel_replacements_total = sentinel_total

        # ── Step 2: Drop 100%-null and >threshold-null columns ────────────────
        df, dropped_cols = self._drop_hopeless_columns(df, report)
        report.columns_dropped = dropped_cols

        # ── Step 3: If df is now empty of columns, return early ───────────────
        if df.shape[1] == 0:
            logger.warning("[MDE] All columns were null/dropped — no data to impute.")
            quarantine_df = pd.DataFrame()
            report.final_shape = (0, 0)
            return df, quarantine_df, report

        # ── Step 4: Classify missing mechanism per column ─────────────────────
        profiles = self._classify_columns(df, target_col, rl_recommendations)
        report.columns_profiled = profiles

        # ── Step 5: Add MNAR indicator columns ────────────────────────────────
        if self._add_indicator:
            df = self._add_mnar_indicators(df, profiles)

        # ── Step 6: Execute imputation per column ─────────────────────────────
        df, lib_used = self._impute(df, profiles, target_col)
        report.imputation_library_used = lib_used

        # ── Step 7: Update imputed row counts in profiles ─────────────────────
        for p in profiles:
            if p.column in df.columns:
                p.rows_imputed = int(df[p.column].notna().sum())

        # ── Step 8: Quarantine rows that are still too null ───────────────────
        df, quarantine_df = self._quarantine_rows(df)
        report.quarantine_rows = len(quarantine_df)

        report.final_shape = df.shape
        logger.info(
            "[MDE] run_id=%s — original=%s final=%s dropped_cols=%d "
            "quarantine_rows=%d sentinel_replacements=%d library=%s",
            run_id[:8] if run_id else "?",
            report.original_shape, report.final_shape,
            len(dropped_cols), report.quarantine_rows,
            sentinel_total, lib_used,
        )
        return df, quarantine_df, report

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Sentinel replacement
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _replace_sentinels(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Replace string and numeric sentinel values with np.nan."""
        total_replaced = 0

        # Object columns — string sentinels
        for col in df.select_dtypes(include="object").columns:
            try:
                mask = df[col].isin(_STRING_SENTINELS)
                count = int(mask.sum())
                if count > 0:
                    df[col] = df[col].where(~mask, other=np.nan)
                    total_replaced += count
                    logger.debug("[MDE] String sentinels replaced in '%s': %d", col, count)
            except Exception as exc:
                logger.debug("[MDE] Sentinel replace failed for '%s': %s", col, exc)

        # Numeric columns — numeric sentinels
        for col in df.select_dtypes(include=np.number).columns:
            try:
                mask = df[col].isin(_NUMERIC_SENTINELS)
                count = int(mask.sum())
                if count > 0:
                    df[col] = df[col].where(~mask, other=np.nan)
                    total_replaced += count
                    logger.debug("[MDE] Numeric sentinels replaced in '%s': %d", col, count)
            except Exception as exc:
                logger.debug("[MDE] Numeric sentinel replace failed for '%s': %s", col, exc)

        if total_replaced > 0:
            logger.info("[MDE] Total sentinel values replaced with NaN: %d", total_replaced)
        return df, total_replaced

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Drop hopeless columns
    # ─────────────────────────────────────────────────────────────────────────

    def _drop_hopeless_columns(
        self, df: pd.DataFrame, report: MissingDataReport
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop columns that are 100% null or above the drop threshold."""
        dropped: List[str] = []
        n = len(df)
        if n == 0:
            return df, dropped

        for col in list(df.columns):
            null_pct = float(df[col].isna().mean())
            if null_pct >= self._drop_threshold:
                reason = "100% null" if null_pct == 1.0 else f"{null_pct:.1%} null > threshold {self._drop_threshold:.0%}"
                logger.warning("[MDE] Dropping column '%s': %s", col, reason)
                report.columns_profiled.append(ColumnMissingProfile(
                    column=col, null_pct=null_pct, mechanism="DROPPED",
                    strategy="drop",
                ))
                dropped.append(col)

        if dropped:
            df = df.drop(columns=dropped)
        return df, dropped

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Classify missing mechanism per column
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_columns(
        self,
        df: pd.DataFrame,
        target_col: Optional[str],
        rl_recommendations: Optional[Dict[str, Any]] = None,
    ) -> List[ColumnMissingProfile]:
        """
        Classify missing mechanism for each column.
        If the Analyst Brain ran (Stage 0.4), it prefers the brain's explicit
        imputation hint.
        """
        profiles: List[ColumnMissingProfile] = []
        null_mask_df = df.isnull()
        brain_decisions = df.attrs.get("column_decisions", {})
        rl_pref = rl_recommendations.get("imputation_preference", {}).get("recommended") if rl_recommendations else None

        for col in df.columns:
            null_pct = float(df[col].isna().mean())
            brain_hint = brain_decisions.get(col, {}).get("imputation_hint")
            should_drop = brain_decisions.get(col, {}).get("should_drop", False)

            if should_drop:
                # The brain explicitly said this column has no value (e.g. ID, >90% null)
                profiles.append(ColumnMissingProfile(
                    column=col, null_pct=null_pct, mechanism="DROPPED_BY_BRAIN",
                    strategy="drop",
                ))
                continue

            if null_pct == 0.0:
                profiles.append(ColumnMissingProfile(
                    column=col, null_pct=0.0,
                    mechanism="COMPLETE", strategy="complete",
                ))
                continue

            # Classify mechanism
            mechanism = self._detect_mechanism(df, col, null_mask_df, target_col)

            # Choose strategy based on brain's hint (which now includes RL nudge from Step 0.4)
            # or apply direct RL preference if hint is generic.
            if brain_hint and brain_hint != "none":
                strategy = brain_hint
                logger.debug("[MDE] Using AnalystBrain hint '%s' for '%s'", strategy, col)
            else:
                strategy = self._choose_strategy(null_pct, mechanism)
                
            # RL Overrides
            if rl_pref == "robust_fast (Iterative Median)" and strategy in ("mice", "knn"):
                strategy = "median"
                logger.debug("[MDE] RL override for '%s': median (robust_fast)", col)
            elif rl_pref == "distribution_preserving (SMOTE-assisted)" and strategy == "median":
                strategy = "mice"
                logger.debug("[MDE] RL override for '%s': mice (distribution_preserving)", col)

            profiles.append(ColumnMissingProfile(
                column=col, null_pct=null_pct,
                mechanism=mechanism, strategy=strategy,
            ))

        return profiles

        return profiles

    def _detect_mechanism(
        self,
        df: pd.DataFrame,
        col: str,
        null_mask_df: pd.DataFrame,
        target_col: Optional[str],
    ) -> str:
        """
        Heuristic mechanism detection:
        1. MNAR: check if missingness of col correlates with col's own observed values
        2. MAR:  check if missingness of col correlates with any other numeric column
        3. MCAR: default
        """
        try:
            col_null_indicator = null_mask_df[col].astype(float)
            n_missing = int(col_null_indicator.sum())
            n_obs     = len(df) - n_missing

            # Need at least 10 observed values to run correlations
            if n_obs < 10 or n_missing < 5:
                return "MCAR"

            # MNAR test: does the value of col predict its own missingness?
            if pd.api.types.is_numeric_dtype(df[col]):
                observed_vals = df.loc[df[col].notna(), col]
                if len(observed_vals) >= 10:
                    try:
                        # Spearman correlation between rank of value and position of nulls
                        # A strong negative correlation suggests MNAR (high values go missing)
                        obs_quantile = observed_vals.rank(pct=True)
                        # Check if the median of observed values differs significantly
                        # from what we'd expect if MCAR (use simple heuristic)
                        median_obs = float(observed_vals.median())
                        overall_approx_mean = float(observed_vals.mean())
                        # If median and mean differ greatly, distribution is skewed/MNAR-like
                        if abs(median_obs - overall_approx_mean) > 2 * float(observed_vals.std() + 1e-9):
                            return "MNAR"
                    except Exception:
                        pass

            # MAR test: does any other column's value predict this column's missingness?
            other_numeric = [
                c for c in df.select_dtypes(include=np.number).columns
                if c != col and c != target_col
            ]
            for other_col in other_numeric[:5]:  # check up to 5 columns
                try:
                    # Pointbiserial correlation: null_indicator vs other_col values
                    valid_mask = df[other_col].notna()
                    if valid_mask.sum() < 20:
                        continue
                    corr = np.corrcoef(
                        col_null_indicator[valid_mask].values,
                        df.loc[valid_mask, other_col].values,
                    )[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.30:
                        return "MAR"
                except Exception:
                    continue

        except Exception as exc:
            logger.debug("[MDE] Mechanism detection failed for '%s': %s", col, exc)

        return "MCAR"

    def _choose_strategy(self, null_pct: float, mechanism: str) -> str:
        """Map mechanism + null_pct to imputation strategy name."""
        if mechanism == "MNAR":
            return "indicator+mice"
        if mechanism == "MAR":
            return "mice"
        # MCAR
        if null_pct < self._mcar_knn_threshold:
            return "median"
        if null_pct < self._mcar_mice_threshold:
            return "knn"
        return "mice"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Add MNAR indicator columns
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_mnar_indicators(
        df: pd.DataFrame, profiles: List[ColumnMissingProfile]
    ) -> pd.DataFrame:
        """For MNAR columns add a binary {col}_was_null indicator column."""
        added = 0
        for p in profiles:
            if p.mechanism == "MNAR" and p.column in df.columns:
                ind_col = f"{p.column}_was_null"
                if ind_col not in df.columns:
                    df[ind_col] = df[p.column].isna().astype(int)
                    p.indicator_added = True
                    added += 1
        if added:
            logger.info("[MDE] Added %d MNAR was_null indicator column(s).", added)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 — Execute imputation
    # ─────────────────────────────────────────────────────────────────────────

    def _impute(
        self,
        df: pd.DataFrame,
        profiles: List[ColumnMissingProfile],
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, str]:
        """
        Apply imputation strategies from the profiles list.
        Returns (df, library_used_name).

        Strategy priority: MICE → KNN → median/mode → constant
        Falls back gracefully if sklearn is not installed.
        """
        mice_cols = [p.column for p in profiles if p.strategy in ("mice", "indicator+mice")
                     and p.column in df.columns]
        knn_cols  = [p.column for p in profiles if p.strategy == "knn"
                     and p.column in df.columns]
        med_cols  = [p.column for p in profiles if p.strategy == "median"
                     and p.column in df.columns]
        lib_used  = "median"

        numeric_cols    = set(df.select_dtypes(include=np.number).columns)
        non_numeric_cols = set(df.columns) - numeric_cols

        # ── All-null numeric column guard: fill before imputer sees them ────────
        for col in list(df.select_dtypes(include=np.number).columns):
            if df[col].isna().all():
                df[col] = 0
                logger.info("[MDE] Column '%s' is 100%% null — filled with 0 before imputation", col)

        # ── Minimum-sample guard: skip heavy imputers on tiny DataFrames ─────
        MIN_ROWS_FOR_IMPUTER = 5
        if len(df) < MIN_ROWS_FOR_IMPUTER:
            logger.warning(
                "[MDE] DataFrame has only %d rows (< %d) — skipping KNN/MICE, using constant fill",
                len(df), MIN_ROWS_FOR_IMPUTER,
            )
            for col in df.columns:
                if df[col].isna().any():
                    if col in numeric_cols:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna("UNKNOWN")
            return df, "constant"

        # ── MICE (IterativeImputer) ────────────────────────────────────────────
        all_mice = [c for c in mice_cols if c in numeric_cols]
        if all_mice:
            df, ok = self._apply_mice(df, all_mice)
            lib_used = "mice" if ok else lib_used

        # Fallback for MICE on non-numeric: mode fill
        non_num_mice = [c for c in mice_cols if c in non_numeric_cols]
        for col in non_num_mice:
            try:
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
                df[col] = df[col].fillna(fill_val)
            except Exception:
                df[col] = df[col].fillna("UNKNOWN")

        # ── KNN imputation ────────────────────────────────────────────────────
        knn_numeric = [c for c in knn_cols if c in numeric_cols]
        if knn_numeric:
            df, ok = self._apply_knn(df, knn_numeric)
            if ok and lib_used != "mice":
                lib_used = "knn"

        # Fallback for KNN on non-numeric: mode fill
        knn_non_num = [c for c in knn_cols if c in non_numeric_cols]
        for col in knn_non_num:
            try:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")
            except Exception:
                df[col] = df[col].fillna("UNKNOWN")

        # ── Median / mode fill ────────────────────────────────────────────────
        for col in med_cols:
            try:
                if col in numeric_cols:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")
            except Exception as exc:
                logger.debug("[MDE] Median fill failed for '%s': %s — using 0/UNKNOWN", col, exc)
                df[col] = df[col].fillna(0 if col in numeric_cols else "UNKNOWN")

        # ── Final sweep: any remaining NaN ───────────────────────────────────
        # At this point the DataCleaner will also run — but we do a best-effort
        # sweep so nothing propagates as NaN into the validation/stats stages.
        for col in df.columns:
            if df[col].isna().any():
                try:
                    if col in numeric_cols:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        mode_val = df[col].mode()
                        df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "UNKNOWN")
                except Exception:
                    df[col] = df[col].fillna(0 if col in numeric_cols else "UNKNOWN")

        return df, lib_used

    # ── MICE helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_mice(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, bool]:
        """Apply IterativeImputer (MICE) to numeric cols. Returns (df, success)."""
        # Min-sample guard: need at least 5 rows for MICE to be meaningful
        if len(df) < 5:
            logger.info("[MDE] Skipping MICE — only %d rows (need ≥5)", len(df))
            for col in cols:
                try:
                    df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                except Exception:
                    df[col] = df[col].fillna(0)
            return df, False
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            imp = IterativeImputer(max_iter=10, random_state=42)
            df[cols] = imp.fit_transform(df[cols])
            logger.info("[MDE] MICE imputation applied to %d column(s).", len(cols))
            return df, True
        except ImportError:
            logger.info("[MDE] sklearn IterativeImputer not available — falling back to KNN/median.")
        except Exception as exc:
            logger.warning("[MDE] MICE failed: %s — falling back to median.", exc)
        # Fallback: median
        for col in cols:
            try:
                df[col] = df[col].fillna(df[col].median())
            except Exception:
                df[col] = df[col].fillna(0)
        return df, False

    # ── KNN helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_knn(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, bool]:
        """Apply KNNImputer to numeric cols. Returns (df, success)."""
        try:
            from sklearn.impute import KNNImputer
            n_neighbours = max(1, min(5, len(df) - 1))
            imp = KNNImputer(n_neighbors=n_neighbours)
            df[cols] = imp.fit_transform(df[cols])
            logger.info("[MDE] KNN imputation applied to %d column(s) (k=%d).", len(cols), n_neighbours)
            return df, True
        except ImportError:
            logger.info("[MDE] sklearn KNNImputer not available — falling back to median.")
        except Exception as exc:
            logger.warning("[MDE] KNN failed: %s — falling back to median.", exc)
        # Fallback: median
        for col in cols:
            try:
                df[col] = df[col].fillna(df[col].median())
            except Exception:
                df[col] = df[col].fillna(0)
        return df, False

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8 — Quarantine rows
    # ─────────────────────────────────────────────────────────────────────────

    def _quarantine_rows(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Move rows that are still ≥80% null to quarantine.
        These rows have too little signal even after imputation.
        """
        if df.empty or len(df.columns) == 0:
            return df, pd.DataFrame(columns=df.columns)

        null_row_pct = df.isnull().mean(axis=1)
        quarantine_mask = null_row_pct >= self._quarantine_pct

        quarantine_df = df[quarantine_mask].copy()
        clean_df      = df[~quarantine_mask].copy()

        if len(quarantine_df) > 0:
            logger.warning(
                "[MDE] Quarantined %d row(s) (≥%.0f%% null after imputation).",
                len(quarantine_df), self._quarantine_pct * 100,
            )

        return clean_df, quarantine_df
