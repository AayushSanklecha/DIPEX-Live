"""
feature_engineering/engineer.py
---------------------------------
AI & ANALYTICS SERVICE LAYER — Feature Engineering

FeatureEngineer.transform(df, target_col, config) generates new features:
  - Lag features for time-series-like numeric columns
  - Polynomial / interaction terms (degree-2 pairs)
  - Frequency encoding for high-cardinality categoricals
  - Equal-width / quantile binning for numeric columns
  - RL-guided feature pruning via existing RLFeatureSelector

Returns an EngineeredFeatures object with the enriched DataFrame and
a feature manifest documenting what was added and why.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.feature_engineering.engineer")

_MAX_INTERACTION_COLS = 8   # cap to avoid O(n²) explosion
_MAX_LAG_STEPS        = 3   # number of lag steps for time-series features
_HIGH_CARDINALITY_THR = 20  # unique values → frequency encoding


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class EngineeredFeatures:
    """Result of feature engineering transformation."""
    original_shape: Tuple[int, int] = (0, 0)
    final_shape: Tuple[int, int] = (0, 0)
    features_added: List[str] = field(default_factory=list)
    features_pruned: List[str] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)
    df: Optional[pd.DataFrame] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "original_shape": list(self.original_shape),
            "final_shape": list(self.final_shape),
            "features_added": self.features_added,
            "features_pruned": self.features_pruned,
            "net_features_added": len(self.features_added) - len(self.features_pruned),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ── Feature Engineer ───────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Automated feature generation with RL-guided pruning.

    Usage::

        fe = FeatureEngineer(config=config)
        result = fe.transform(df, target_col="churn")
        enriched_df = result.df
        print(result.manifest)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        enable_lag: bool = True,
        enable_interactions: bool = True,
        enable_freq_encoding: bool = True,
        enable_binning: bool = True,
        enable_rl_pruning: bool = True,
    ):
        self.config = config or {}
        self.enable_lag = enable_lag
        self.enable_interactions = enable_interactions
        self.enable_freq_encoding = enable_freq_encoding
        self.enable_binning = enable_binning
        self.enable_rl_pruning = enable_rl_pruning

    def transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> EngineeredFeatures:
        """
        Apply all configured feature engineering steps.

        Parameters
        ----------
        df         : input DataFrame (after preprocessing)
        target_col : target column name (excluded from feature generation)

        Returns
        -------
        EngineeredFeatures with .df being the enriched DataFrame
        """
        t0 = time.perf_counter()
        if df is None or df.empty:
            return EngineeredFeatures()

        result = EngineeredFeatures(original_shape=df.shape)
        df = df.copy()
        manifest: Dict = {}
        added: List[str] = []

        numeric_cols = [c for c in df.select_dtypes(include="number").columns
                        if c != target_col]
        cat_cols = [c for c in df.select_dtypes(exclude="number").columns
                    if c != target_col]

        # ── 1. Lag features ─────────────────────────────────────────────────
        if self.enable_lag and numeric_cols:
            df, lag_added = self._add_lag_features(df, numeric_cols, target_col)
            added.extend(lag_added)
            manifest["lag_features"] = lag_added

        # ── 2. Interaction / polynomial terms ────────────────────────────────
        if self.enable_interactions and len(numeric_cols) >= 2:
            df, inter_added = self._add_interaction_terms(df, numeric_cols, target_col)
            added.extend(inter_added)
            manifest["interaction_features"] = inter_added

        # ── 3. Frequency encoding ─────────────────────────────────────────────
        if self.enable_freq_encoding and cat_cols:
            df, freq_added = self._add_frequency_encoding(df, cat_cols)
            added.extend(freq_added)
            manifest["frequency_encoded"] = freq_added

        # ── 4. Binning ────────────────────────────────────────────────────────
        if self.enable_binning and numeric_cols:
            df, bin_added = self._add_binning(df, numeric_cols, target_col)
            added.extend(bin_added)
            manifest["binned_features"] = bin_added

        # ── 5. RL-guided pruning ───────────────────────────────────────────────
        pruned: List[str] = []
        if self.enable_rl_pruning and added:
            df, pruned = self._rl_prune(df, added, target_col)
            manifest["pruned_features"] = pruned

        result.df = df
        result.final_shape = df.shape
        result.features_added = added
        result.features_pruned = pruned
        result.manifest = manifest
        result.elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[FeatureEngineer] %d→%d cols, added=%d pruned=%d elapsed=%.0fms",
            result.original_shape[1], result.final_shape[1],
            len(added), len(pruned), result.elapsed_ms,
        )
        return result

    # ── Feature generation methods ─────────────────────────────────────────────

    def _add_lag_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Shift-based lag features (useful for ordered / time-series data)."""
        added = []
        # Only use columns that look like they could be time-ordered (use all numeric)
        cols_to_lag = numeric_cols[:6]  # cap to avoid bloat
        for col in cols_to_lag:
            for lag in range(1, _MAX_LAG_STEPS + 1):
                feat_name = f"{col}_lag{lag}"
                if feat_name not in df.columns:
                    df[feat_name] = df[col].shift(lag)
                    added.append(feat_name)
        return df, added

    def _add_interaction_terms(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Degree-2 interaction features for top numeric columns."""
        added = []
        cols = numeric_cols[:_MAX_INTERACTION_COLS]
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1 :]:
                feat_name = f"{col_a}_x_{col_b}"
                if feat_name not in df.columns:
                    df[feat_name] = df[col_a] * df[col_b]
                    added.append(feat_name)
        return df, added

    def _add_frequency_encoding(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Replace high-cardinality categoricals with their frequency counts."""
        added = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > _HIGH_CARDINALITY_THR:
                feat_name = f"{col}_freq_enc"
                if feat_name not in df.columns:
                    freq_map = df[col].value_counts(normalize=True)
                    df[feat_name] = df[col].map(freq_map).fillna(0.0)
                    added.append(feat_name)
        return df, added

    def _add_binning(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Equal-width binning (4 bins) for numeric columns."""
        added = []
        n_bins = self.config.get("feature_engineering", {}).get("n_bins", 4)
        for col in numeric_cols[:8]:  # cap to top-8
            feat_name = f"{col}_bin{n_bins}"
            if feat_name not in df.columns:
                try:
                    df[feat_name] = pd.cut(
                        df[col], bins=n_bins, labels=False, duplicates="drop"
                    ).astype("float32")
                    added.append(feat_name)
                except Exception:
                    pass
        return df, added

    def _rl_prune(
        self,
        df: pd.DataFrame,
        added: List[str],
        target_col: Optional[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Use existing RLFeatureSelector to prune low-value engineered features.
        Falls back gracefully if selector is unavailable.
        """
        try:
            from preprocessing.rl_feature_selector import RLFeatureSelector
            all_cols = [c for c in df.columns if c != target_col]
            selector = RLFeatureSelector(all_cols, max_features=len(all_cols))
            active = set(selector.get_active_features())
            to_drop = [c for c in added if c not in active and c in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
                logger.info("[FeatureEngineer] RL pruned %d features", len(to_drop))
            return df, to_drop
        except Exception as exc:
            logger.debug("RL pruning unavailable (non-fatal): %s", exc)
            return df, []
