"""
profiling/profiler.py
----------------------
Step 3 — Data Profiling Engine: Distribution & Column Statistics.

For every column the Profiler answers:
  - What type of data is this?
  - What are its central tendency and dispersion?  (mean, median, std, IQR, percentiles)
  - What is its distributional shape?              (skewness, kurtosis, normality flag)
  - Are there anomalies?                           (IQR Tukey fence + Z-score dual method)
  - How repetitive / unique is it?                 (cardinality tier)
  - How complete is it?                            (null count / pct)

This simulates the first 15 minutes of a senior quant analyst looking at new data.

Usage::

    profiler = Profiler(config)
    report   = profiler.profile(df)
    flags    = report["analyst_flags"]   # flat list of auto-detected concerns
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants (overridden by config)
# ──────────────────────────────────────────────────────────────────────────────

_IQR_MULTIPLIER_DEFAULT       = 1.5
_ZSCORE_THRESHOLD_DEFAULT     = 3.0
_SKEW_FLAG_THRESHOLD          = 1.0   # |skewness| > this → HIGH_SKEW flag
_KURTOSIS_FLAG_THRESHOLD      = 3.0   # excess kurtosis > this → HEAVY_TAILS flag
_HIGH_OUTLIER_PCT_THRESHOLD   = 0.03  # IQR outlier pct > 3% → flag

# Cardinality tier boundaries (fraction of total rows)
_CARD_LOW_DEFAULT             = 0.05   # < 5 % unique → low
_CARD_HIGH_DEFAULT            = 0.50   # > 50% unique → high
_CARD_UNIQUE_DEFAULT          = 0.95   # > 95% unique → unique


# ──────────────────────────────────────────────────────────────────────────────
# Profiler
# ──────────────────────────────────────────────────────────────────────────────

class Profiler:
    """
    Computes per-column statistical profiles and dataset-level analyst flags.

    Args:
        config: Project config dict.  The ``profiling`` sub-key is used for
                threshold overrides.  Passing ``None`` uses safe defaults.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("profiling", {})

        out_cfg = cfg.get("outlier", {})
        self._iqr_mult:      float = float(out_cfg.get("iqr_multiplier",   _IQR_MULTIPLIER_DEFAULT))
        self._zscore_thresh: float = float(out_cfg.get("zscore_threshold", _ZSCORE_THRESHOLD_DEFAULT))

        card_cfg = cfg.get("cardinality", {})
        self._card_low:    float = float(card_cfg.get("low_pct",    _CARD_LOW_DEFAULT))
        self._card_high:   float = float(card_cfg.get("high_pct",   _CARD_HIGH_DEFAULT))
        self._card_unique: float = float(card_cfg.get("unique_pct", _CARD_UNIQUE_DEFAULT))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Profiles every column and returns a structured report dict.

        Returns:
            {
              "row_count":      int,
              "column_count":   int,
              "columns":        { col_name: {column_profile} },
              "analyst_flags":  [ {flag_dict}, ... ],
            }
        """
        if df is None or df.empty:
            logger.warning("Profiler: received empty DataFrame — skipping.")
            return {"row_count": 0, "column_count": 0, "columns": {}, "analyst_flags": []}

        n_rows = len(df)
        columns: Dict[str, Any] = {}
        flags:   List[Dict[str, Any]] = []

        # [RL] Profiling strategy: decide full vs. basic per column
        dataset_id = getattr(df, "attrs", {}).get("dataset_id", "unknown")
        try:
            import time as _time
            from profiling.rl_profiling_strategy import get_rl_strategy
            _rl = get_rl_strategy()
            _use_rl = True
        except Exception:  # noqa: BLE001
            _use_rl = False

        for col in df.columns:
            if _use_rl:
                t0       = _time.perf_counter()
                _action  = _rl.get_action(dataset_id, col)
                _full    = (_action == "full")
            else:
                _full    = True

            col_profile = self._profile_column(df[col], n_rows, full=_full)
            columns[col] = col_profile
            flags_before = len(flags)
            self._collect_flags(col, col_profile, flags)
            flags_after  = len(flags)

            if _use_rl:
                _ms = (_time.perf_counter() - t0) * 1000
                _rl.record_outcome(
                    dataset_id, col, _action,
                    flags_raised=flags_after - flags_before,
                    compute_ms=_ms,
                )

        logger.info(
            "Profiler: %d columns profiled, %d analyst flag(s) raised.",
            len(df.columns), len(flags),
        )
        return {
            "row_count":     n_rows,
            "column_count":  len(df.columns),
            "columns":       columns,
            "analyst_flags": flags,
        }

    # ------------------------------------------------------------------
    # Column dispatch
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series, n_rows: int, full: bool = True) -> Dict[str, Any]:
        """Profiles a single column; `full` controls whether expensive stats are computed."""
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique(dropna=True))
        null_pct = null_count / n_rows if n_rows > 0 else 0.0

        profile: Dict[str, Any] = {
            "dtype":        str(series.dtype),
            "null_count":   null_count,
            "null_pct":     round(null_pct, 6),
            "unique_count": unique_count,
            "cardinality_tier": self._cardinality_tier(unique_count, n_rows),
        }

        clean = series.dropna()

        # Early-exit: all values are null — write note regardless of dtype
        if clean.empty:
            profile["note"] = "All values null — no statistics available."
            return profile

        if pd.api.types.is_numeric_dtype(series):
            self._profile_numeric(clean, profile, full=full)
        elif pd.api.types.is_datetime64_any_dtype(series):
            self._profile_datetime(clean, profile)
        elif isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
            self._profile_categorical(series, profile, n_rows)

        return profile

    # ------------------------------------------------------------------
    # Numeric profile (the richest sub-profile)
    # ------------------------------------------------------------------

    def _profile_numeric(
        self, clean: pd.Series, profile: Dict[str, Any], full: bool = True
    ) -> None:
        if clean.empty:
            profile["note"] = "All values null — no numeric stats available."
            return

        values = clean.to_numpy(dtype=np.float64)
        q25, q75 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        iqr = q75 - q25

        # ── Central tendency & dispersion ──────────────────────────────
        profile.update({
            "min":      float(values.min()),
            "max":      float(values.max()),
            "mean":     float(values.mean()),
            "median":   float(np.median(values)),
            "std":      float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "variance": float(values.var(ddof=1)) if len(values) > 1 else 0.0,
            "q05":      float(np.percentile(values, 5)),
            "q25":      q25,
            "q75":      q75,
            "q95":      float(np.percentile(values, 95)),
            "iqr":      round(iqr, 6),
        })

        # ── Distribution shape (only for full profile) ─────────────────────────
        if full:
            skew_val = float(scipy_stats.skew(values)) if len(values) >= 3 else 0.0
            kurt_val = float(scipy_stats.kurtosis(values)) if len(values) >= 4 else 0.0
            profile.update({
                "skewness":    round(skew_val, 6),
                "kurtosis":    round(kurt_val, 6),
                "high_skew":   abs(skew_val) > _SKEW_FLAG_THRESHOLD,
                "heavy_tails": kurt_val > _KURTOSIS_FLAG_THRESHOLD,
            })
            # Normality (Shapiro-Wilk; reliable for n ≤ 5000)
            if 3 <= len(values) <= 5000:
                _, sw_p = scipy_stats.shapiro(values)
                profile["shapiro_p"]   = round(float(sw_p), 6)
                profile["normal_dist"] = sw_p >= 0.05
            else:
                profile["shapiro_p"]   = None
                profile["normal_dist"] = None
        else:
            # Basic mode — populate keys with neutral sentinel values
            profile.update({
                "skewness": None, "kurtosis": None,
                "high_skew": False, "heavy_tails": False,
                "shapiro_p": None, "normal_dist": None,
                "profiling_mode": "basic",
            })

        # ── Outlier detection — dual method ───────────────────────────
        iqr_outliers = self._outliers_iqr(values, q25, q75, iqr)
        z_outliers   = self._outliers_zscore(values)

        profile["outliers"] = {
            "iqr": {
                "count":     int(iqr_outliers.sum()),
                "pct":       round(float(iqr_outliers.mean()), 6),
                "fence_low":  round(q25 - self._iqr_mult * iqr, 6),
                "fence_high": round(q75 + self._iqr_mult * iqr, 6),
            },
            "zscore": {
                "count":     int(z_outliers.sum()),
                "pct":       round(float(z_outliers.mean()), 6),
                "threshold": self._zscore_thresh,
            },
        }

    def _outliers_iqr(
        self,
        values: np.ndarray,
        q25: float,
        q75: float,
        iqr: float,
    ) -> np.ndarray:
        """Boolean mask: True where value is outside the Tukey IQR fence."""
        fence_low  = q25 - self._iqr_mult * iqr
        fence_high = q75 + self._iqr_mult * iqr
        return (values < fence_low) | (values > fence_high)

    def _outliers_zscore(self, values: np.ndarray) -> np.ndarray:
        """Boolean mask: True where |z-score| > configured threshold."""
        if values.std() == 0:
            return np.zeros(len(values), dtype=bool)
        z = np.abs((values - values.mean()) / values.std(ddof=1))
        return z > self._zscore_thresh

    # ------------------------------------------------------------------
    # Datetime profile
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_datetime(clean: pd.Series, profile: Dict[str, Any]) -> None:
        if clean.empty:
            profile["note"] = "All values null — no datetime stats available."
            return
        profile.update({
            "min":        str(clean.min()),
            "max":        str(clean.max()),
            "range_days": int((clean.max() - clean.min()).days),
            "median":     str(clean.sort_values().iloc[len(clean) // 2]),
        })

    # ------------------------------------------------------------------
    # Categorical profile
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_categorical(
        series: pd.Series, profile: Dict[str, Any], n_rows: int
    ) -> None:
        vc = series.value_counts(dropna=False)
        non_null_vc = series.value_counts(dropna=True)

        profile["top_values"] = {
            str(k): int(v) for k, v in non_null_vc.head(10).items()
        }
        if not non_null_vc.empty:
            profile["mode"] = str(non_null_vc.index[0])
            profile["mode_freq"] = round(float(non_null_vc.iloc[0] / n_rows), 6)

        # Shannon entropy (bits) — measures uniformity of distribution
        freqs = non_null_vc.values / non_null_vc.sum()
        if len(freqs) > 1:
            profile["entropy_bits"] = round(
                float(-np.sum(freqs * np.log2(freqs + 1e-12))), 6
            )
        else:
            profile["entropy_bits"] = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cardinality_tier(self, unique_count: int, n_rows: int) -> str:
        """
        Classifies cardinality into four tiers:
          low    — < 5 % unique (e.g. status codes, boolean-like)
          medium — 5–50% unique (e.g. categorical with many values)
          high   — 50–95% unique (e.g. names, free-text)
          unique — > 95% unique (e.g. primary keys, UUIDs)
        """
        if n_rows == 0:
            return "unknown"
        ratio = unique_count / n_rows
        if ratio < self._card_low:
            return "low"
        elif ratio < self._card_high:
            return "medium"
        elif ratio < self._card_unique:
            return "high"
        else:
            return "unique"

    @staticmethod
    def _collect_flags(
        col: str,
        col_profile: Dict[str, Any],
        flags: List[Dict[str, Any]],
    ) -> None:
        """Appends analyst flags from a column profile into the shared flags list."""
        # High skewness
        if col_profile.get("high_skew"):
            flags.append({
                "column": col,
                "flag":   "HIGH_SKEW",
                "detail": f"skewness={col_profile['skewness']:.4f}. "
                          "Consider log/Box-Cox transform before modelling.",
            })

        # Heavy tails
        if col_profile.get("heavy_tails"):
            flags.append({
                "column": col,
                "flag":   "HEAVY_TAILS",
                "detail": f"excess kurtosis={col_profile['kurtosis']:.4f}. "
                          "Distribution has heavier tails than normal.",
            })

        # Outliers (IQR method > 5% threshold)
        outlier_info = col_profile.get("outliers", {}).get("iqr", {})
        if outlier_info.get("pct", 0) > _HIGH_OUTLIER_PCT_THRESHOLD:
            flags.append({
                "column": col,
                "flag":   "HIGH_OUTLIER_RATE_IQR",
                "detail": f"{outlier_info['pct']:.2%} of values are outside IQR fence "
                          f"[{outlier_info['fence_low']:.4g}, {outlier_info['fence_high']:.4g}].",
            })

        # Unique-cardinality identifier (likely a primary key — may inflate model)
        if col_profile.get("cardinality_tier") == "unique":
            flags.append({
                "column": col,
                "flag":   "HIGH_CARDINALITY",
                "detail": f"Column appears to be a unique identifier "
                          f"({col_profile['unique_count']} unique values). "
                          "Exclude from feature set unless intended.",
            })

        # Not normally distributed
        sw_p = col_profile.get("shapiro_p")
        if sw_p is not None and sw_p < 0.05:
            flags.append({
                "column": col,
                "flag":   "NON_NORMAL_DISTRIBUTION",
                "detail": f"Shapiro-Wilk p={sw_p:.4g} (<0.05). "
                          "Parametric tests assume normality — consider non-parametric alternatives.",
            })
