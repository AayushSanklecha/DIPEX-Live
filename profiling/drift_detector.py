"""
profiling/drift_detector.py
-----------------------------
Step 3 — Data Profiling Engine: Distribution Drift Detection.

Drift signals that the real-world process generating the data has changed,
invalidating any model trained on historical data.  A senior analyst uses
multiple complementary tests because no single test is sufficient:

  PSI (Population Stability Index)
    Industry standard in credit scoring.  Compares bin frequencies.
    Interpretable thresholds: <0.10 stable | 0.10–0.25 watch | >0.25 investigate.
    Weakness: sensitive to binning strategy; can miss shape changes.

  Kolmogorov-Smirnov (KS) two-sample test
    Distribution-free test comparing CDFs of two samples.
    Returns a statistic D ∈ [0,1] and a p-value.
    p < 0.05 → reject null hypothesis (distributions are the same) → drift likely.

  Jensen-Shannon Divergence (JSD)
    Symmetric, bounded [0,1] generalisation of KL divergence.
    More stable than PSI on small samples.
    JSD > 0.10 is conventionally treated as significant.

  Temporal drift (time-series aware)
    Slices data by rolling time windows and tracks per-window statistics.
    Flags windows where the rolling mean deviates > N standard deviations
    from the global baseline mean.

Usage::

    dd = DriftDetector(config)
    result = dd.detect(baseline_df, current_df)
    temporal = dd.detect_temporal_drift(df, timestamp_col="event_time")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (overridden by config)
# ──────────────────────────────────────────────────────────────────────────────

_MIN_BIN_FREQ:          float = 1e-6   # Avoid log(0) in PSI
_PSI_STABLE:            float = 0.10
_PSI_WATCH:             float = 0.25
_KS_P_THRESHOLD:        float = 0.05
_JS_THRESHOLD:          float = 0.10
_TEMPORAL_ZSCORE_ALERT: float = 2.0
_MIN_SAMPLES:           int   = 5      # Minimum non-null values per column


class DriftDetector:
    """
    Multi-method data drift detector comparing a baseline distribution to a
    current distribution across all shared numeric columns.

    Args:
        config: Project config dict.  The ``profiling.drift`` sub-key
                controls thresholds.  ``None`` uses safe defaults.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("profiling", {}).get("drift", {})
        self._ks_p_thresh:   float = float(cfg.get("ks_p_value_threshold",    _KS_P_THRESHOLD))
        self._js_thresh:     float = float(cfg.get("js_divergence_threshold",  _JS_THRESHOLD))
        self._psi_watch:     float = float(cfg.get("psi_watch_threshold",      _PSI_WATCH))
        self._psi_stable:    float = float(cfg.get("psi_stable_threshold",     _PSI_STABLE))

        temp_cfg = cfg.get("temporal", {})
        self._temporal_enabled: bool  = bool(temp_cfg.get("enabled", False))
        self._temporal_ts_col:  Optional[str] = temp_cfg.get("timestamp_column")
        self._temporal_window:  str   = str(temp_cfg.get("window", "7D"))
        self._temporal_zscore:  float = float(temp_cfg.get("zscore_alert_threshold", _TEMPORAL_ZSCORE_ALERT))

    # ------------------------------------------------------------------
    # Public API — cross-batch drift
    # ------------------------------------------------------------------

    def detect(
        self,
        baseline_df: pd.DataFrame,
        current_df:  pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Computes PSI, KS-test and Jensen-Shannon divergence for all shared
        numeric columns between baseline and current DataFrames.

        Returns:
            {
              "columns": {
                col: { "psi": float, "psi_status": str,
                       "ks_statistic": float, "ks_p_value": float, "ks_drifted": bool,
                       "js_divergence": float, "js_drifted": bool,
                       "drifted": bool }
              },
              "analyst_flags": [ {...}, ... ],
              "drifted_columns": [col, ...]
            }
        """
        baseline_num = set(baseline_df.select_dtypes(include=[np.number]).columns)
        current_num  = set(current_df.select_dtypes(include=[np.number]).columns)
        common_cols  = sorted(baseline_num & current_num)

        column_results: Dict[str, Any] = {}
        analyst_flags:  List[Dict[str, Any]] = []

        for col in common_cols:
            b_vals = baseline_df[col].dropna().to_numpy(dtype=np.float64)
            c_vals = current_df[col].dropna().to_numpy(dtype=np.float64)

            if len(b_vals) < _MIN_SAMPLES or len(c_vals) < _MIN_SAMPLES:
                logger.debug(
                    "DriftDetector: column '%s' skipped — insufficient samples.", col
                )
                continue

            psi       = self.calculate_psi(b_vals, c_vals)
            ks_stat, ks_p, ks_drifted = self._ks_test(b_vals, c_vals)
            js_div, js_drifted        = self._js_divergence(b_vals, c_vals)

            # Consensus drift: any two methods agree → drift detected
            drifted = int(ks_drifted) + int(js_drifted) + int(psi > self._psi_watch) >= 2

            column_results[col] = {
                "psi":          round(psi, 6),
                "psi_status":   self._psi_status(psi),
                "ks_statistic": round(ks_stat, 6),
                "ks_p_value":   round(ks_p, 6),
                "ks_drifted":   ks_drifted,
                "js_divergence":round(js_div, 6),
                "js_drifted":   js_drifted,
                "drifted":      drifted,
            }

            if drifted:
                analyst_flags.append({
                    "column": col,
                    "flag":   "DRIFT_DETECTED",
                    "detail": (
                        f"PSI={psi:.4f} ({self._psi_status(psi)}), "
                        f"KS p={ks_p:.4f} ({'drifted' if ks_drifted else 'stable'}), "
                        f"JS={js_div:.4f} ({'drifted' if js_drifted else 'stable'}). "
                        "Model retraining strongly recommended."
                    ),
                })

        drifted_cols = [c for c, r in column_results.items() if r["drifted"]]
        logger.info(
            "DriftDetector: %d column(s) checked — %d drifted.",
            len(column_results), len(drifted_cols),
        )

        # [ML] Multivariate drift via Autoencoder reconstruction error
        multivariate = self.detect_multivariate_drift(baseline_df, current_df, common_cols)
        if multivariate.get("drifted", False):
            analyst_flags.append({
                "column": "[MULTIVARIATE]",
                "flag":   "MULTIVARIATE_DRIFT_DETECTED",
                "detail": (
                    f"[ML:Autoencoder] {multivariate.get('drifted_ratio', 0):.1%} of current rows "
                    f"exceed the 95th-percentile reconstruction error threshold "
                    f"({multivariate.get('threshold_error', 0):.4f}). "
                    "Structural multivariate shift detected. Retraining strongly recommended."
                ),
            })

        return {
            "columns":          column_results,
            "drifted_columns":  drifted_cols,
            "multivariate":     multivariate,
            "analyst_flags":    analyst_flags,
        }

    # ------------------------------------------------------------------
    # [ML] Autoencoder multivariate drift detection
    # ------------------------------------------------------------------

    def detect_multivariate_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df:  pd.DataFrame,
        shared_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect structural multivariate drift using an Autoencoder.

        Strategy (tiered):
          1. Load a Colab-trained MLPRegressor from ``models/drift_autoencoder.pkl``
             (preferred — higher quality, domain-aware).
          2. If artifact absent, fit a lightweight in-memory autoencoder on the
             current baseline batch (still statistically sound for > 50 rows).
          3. Compute reconstruction error on current_df; flag if > 10 % of rows
             exceed the 95th-percentile baseline error.

        Returns
        -------
        dict with keys: drifted, drifted_ratio, threshold_error,
                        mean_current_error, method
        """
        _ARTIFACT  = os.path.join(
            os.path.dirname(__file__), "..", "models", "drift_autoencoder.pkl"
        )
        _SCALER_A  = os.path.join(
            os.path.dirname(__file__), "..", "models", "drift_scaler.pkl"
        )
        _PCA_PATH  = os.path.join(
            os.path.dirname(__file__), "..", "models", "drift_pca.pkl"
        )

        # ── Resolve shared numeric columns ────────────────────────────────────
        if shared_cols is None:
            bn = set(baseline_df.select_dtypes(include="number").columns)
            cn = set(current_df.select_dtypes(include="number").columns)
            shared_cols = sorted(bn & cn)

        if len(shared_cols) < 2 or len(baseline_df) < 30 or len(current_df) < 10:
            return {
                "drifted": False,
                "reason":  "insufficient numeric columns or sample size",
                "method":  "skipped",
            }

        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("DriftDetector: scikit-learn absent — multivariate check skipped.")
            return {"drifted": False, "method": "skipped_no_sklearn"}

        base_X = baseline_df[shared_cols].fillna(0)
        curr_X = current_df[shared_cols].fillna(0)

        try:
            # ── Tier 1: Load pre-trained artifact (PCA + autoencoder) ─────────
            #
            #  Design: the artifact is distribution-agnostic because we use a
            #  LOCAL StandardScaler fitted on the baseline batch (not the saved
            #  scaler) for data normalisation.  The saved scaler only carries
            #  the n_features_in_ metadata (= N_DRIFT_FEATURES = 15) so we
            #  know how wide to pad/truncate the runtime data before projecting
            #  through the pre-trained PCA → autoencoder.
            #
            #  This means the artifact works on ANY dataset regardless of which
            #  columns are present — no column-name alignment needed.
            # ─────────────────────────────────────────────────────────────────
            if os.path.exists(_ARTIFACT) and os.path.exists(_SCALER_A):
                import joblib  # type: ignore
                ae            = joblib.load(_ARTIFACT)
                meta_scaler   = joblib.load(_SCALER_A)   # metadata only
                pca           = joblib.load(_PCA_PATH) if os.path.exists(_PCA_PATH) else None
                method        = "pretrained_pca_ae" if pca is not None else "pretrained_ae"
                logger.debug("DriftDetector: loaded pre-trained artifact (method=%s).", method)

                # ── Pad / truncate to the training width ──────────────────────
                n_target = (
                    meta_scaler.n_features_in_
                    if hasattr(meta_scaler, "n_features_in_")
                    else base_X.shape[1]
                )

                def _pad_cols(arr: np.ndarray, target: int) -> np.ndarray:
                    if arr.shape[1] == target:
                        return arr
                    if arr.shape[1] > target:
                        return arr[:, :target]
                    return np.hstack([arr, np.zeros((arr.shape[0], target - arr.shape[1]))])

                base_X_arr = _pad_cols(base_X.values.astype(np.float64), n_target)
                curr_X_arr = _pad_cols(curr_X.values.astype(np.float64), n_target)

                # ── LOCAL StandardScaler fitted on baseline only ───────────────
                local_sc = StandardScaler()
                base_X_arr = local_sc.fit_transform(base_X_arr)
                curr_X_arr = local_sc.transform(curr_X_arr)

                # ── Project through pre-trained PCA (if available) ────────────
                if pca is not None:
                    base_X_arr = pca.transform(base_X_arr)
                    curr_X_arr = pca.transform(curr_X_arr)

            else:
                # ── Tier 2: In-memory fit (no artifact) ───────────────────────
                dim    = len(shared_cols)
                h      = (max(dim // 2, 2), max(dim // 4, 1), max(dim // 2, 2))
                local_sc = StandardScaler()
                ae     = MLPRegressor(
                    hidden_layer_sizes=h,
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    random_state=42,
                    warm_start=False,
                )
                base_X_arr = local_sc.fit_transform(base_X.values.astype(np.float64))
                curr_X_arr = local_sc.transform(curr_X.values.astype(np.float64))
                ae.fit(base_X_arr, base_X_arr)
                method = "in_memory_fit"
                logger.info(
                    "DriftDetector: no pre-trained artifact found — fit in-memory "
                    "autoencoder on %d rows × %d features.", len(base_X), dim
                )

            # ── Score baseline to set 95th-pct threshold ──────────────────────
            # base_X_arr / curr_X_arr are already normalised + PCA-projected.
            base_err = np.mean(np.square(base_X_arr - ae.predict(base_X_arr)), axis=1)
            p95      = float(np.percentile(base_err, 95))

            # ── Score current batch ───────────────────────────────────────────
            curr_err      = np.mean(np.square(curr_X_arr - ae.predict(curr_X_arr)), axis=1)
            drifted_ratio = float(np.mean(curr_err > p95))
            drifted       = drifted_ratio > 0.10

            logger.info(
                "DriftDetector [ML]: multivariate drift=%s (ratio=%.2f%%, threshold=%.4f) via %s.",
                drifted, drifted_ratio * 100, p95, method,
            )
            return {
                "drifted":            drifted,
                "drifted_ratio":      round(drifted_ratio, 4),
                "threshold_error":    round(p95, 6),
                "mean_current_error": round(float(curr_err.mean()), 6),
                "method":             method,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("DriftDetector: multivariate drift check failed: %s", exc)
            return {"drifted": False, "method": "exception", "error": str(exc)}

    # ------------------------------------------------------------------
    # Public API — temporal drift
    # ------------------------------------------------------------------

    def detect_temporal_drift(
        self,
        df:            pd.DataFrame,
        timestamp_col: Optional[str] = None,
        window:        Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detects drift within a single DataFrame by slicing into time windows.

        For each numeric column, computes per-window mean/median/std and flags
        windows where the rolling mean deviates > N×σ from the column's
        overall mean.

        Args:
            df:            DataFrame with a datetime column.
            timestamp_col: Name of the datetime column.  Falls back to config.
            window:        Pandas offset alias (e.g. "7D", "1ME").  Falls back to config.

        Returns:
            {
              col: {
                "global_mean": float,
                "global_std":  float,
                "windows": [ {"period": str, "mean": float, "std": float,
                              "median": float, "n": int, "alert": bool} ],
              }
            }
        """
        ts_col = timestamp_col or self._temporal_ts_col
        win    = window or self._temporal_window

        if not ts_col:
            logger.info(
                "Temporal drift: no timestamp_column configured — skipping."
            )
            return {}

        if ts_col not in df.columns:
            logger.warning(
                "Temporal drift: column '%s' not found in DataFrame — skipping.", ts_col
            )
            return {}

        # Coerce timestamp column
        df_local = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_local[ts_col]):
            df_local[ts_col] = pd.to_datetime(df_local[ts_col], errors="coerce", utc=True)

        # Set index
        df_local = df_local.set_index(ts_col)

        # Strip timezone for resample compatibility (pandas <2.2 resample tz issues)
        if df_local.index.tzinfo is not None:
            df_local.index = df_local.index.tz_convert("UTC").tz_localize(None)

        df_local = df_local.sort_index()
        num_cols  = df_local.select_dtypes(include=[np.number]).columns.tolist()

        results: Dict[str, Any] = {}
        for col in num_cols:
            series = df_local[col].dropna()
            if len(series) < _MIN_SAMPLES:
                continue

            global_mean = float(series.mean())
            global_std  = float(series.std(ddof=1)) if len(series) > 1 else 0.0

            windows_out: List[Dict[str, Any]] = []
            for period, group in series.resample(win):
                if group.empty:
                    continue
                w_mean   = float(group.mean())
                w_std    = float(group.std(ddof=1)) if len(group) > 1 else 0.0
                w_median = float(group.median())
                # z-score of this window's mean relative to global distribution
                z = abs(w_mean - global_mean) / (global_std + 1e-12)
                windows_out.append({
                    "period": str(period.date()),
                    "mean":   round(w_mean, 6),
                    "std":    round(w_std, 6),
                    "median": round(w_median, 6),
                    "n":      int(len(group)),
                    "alert":  z > self._temporal_zscore,
                })

            results[col] = {
                "global_mean": round(global_mean, 6),
                "global_std":  round(global_std, 6),
                "windows":     windows_out,
                "alert_windows": sum(1 for w in windows_out if w["alert"]),
            }

        logger.info(
            "Temporal drift: %d column(s) analysed across %s windows.",
            len(results), win,
        )
        return results

    # ------------------------------------------------------------------
    # PSI
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual:   np.ndarray,
        buckets:  int = 10,
    ) -> float:
        """
        Population Stability Index between two numeric arrays.

        Returns 0.0 when either array is empty or lacks enough unique values.
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        combined    = np.concatenate([expected, actual])
        breakpoints = np.histogram_bin_edges(combined, bins=buckets)

        exp_counts, _ = np.histogram(expected, bins=breakpoints)
        act_counts, _ = np.histogram(actual,   bins=breakpoints)

        exp_pct = np.clip(exp_counts / len(expected), _MIN_BIN_FREQ, None)
        act_pct = np.clip(act_counts / len(actual),   _MIN_BIN_FREQ, None)

        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def _psi_status(self, psi: float) -> str:
        if psi < self._psi_stable:
            return "STABLE"
        elif psi < self._psi_watch:
            return "WATCH"
        else:
            return "SIGNIFICANT_DRIFT"

    # ------------------------------------------------------------------
    # KS two-sample test
    # ------------------------------------------------------------------

    def _ks_test(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float, bool]:
        """Returns (statistic, p_value, drifted)."""
        result   = scipy_stats.ks_2samp(baseline, current)
        drifted  = bool(result.pvalue < self._ks_p_thresh)
        return float(result.statistic), float(result.pvalue), drifted

    # ------------------------------------------------------------------
    # Jensen-Shannon divergence
    # ------------------------------------------------------------------

    def _js_divergence(
        self, baseline: np.ndarray, current: np.ndarray, bins: int = 20
    ) -> Tuple[float, bool]:
        """Returns (js_divergence, drifted)."""
        combined    = np.concatenate([baseline, current])
        breakpoints = np.histogram_bin_edges(combined, bins=bins)

        b_hist, _ = np.histogram(baseline, bins=breakpoints, density=True)
        c_hist, _ = np.histogram(current,  bins=breakpoints, density=True)

        # Smooth to avoid zeros
        b_hist = b_hist + 1e-12
        c_hist = c_hist + 1e-12

        js  = float(jensenshannon(b_hist / b_hist.sum(), c_hist / c_hist.sum()))
        return round(js, 6), js > self._js_thresh

    # ------------------------------------------------------------------
    # Legacy compat
    # ------------------------------------------------------------------

    def detect_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Backwards-compatible wrapper — returns raw PSI scores only."""
        result = self.detect(baseline_df, current_df)
        return {col: info["psi"] for col, info in result["columns"].items()}
