"""
feature_engineering/engineer.py
---------------------------------
LEGACY PROXY — AI & ANALYTICS SERVICE LAYER — Feature Engineering

This file is now a thin proxy to preprocessing/feature_engineer.py to ensure
all enterprise robustness fixes are applied centrally.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from preprocessing.feature_engineer import FeatureEngineer as RobustFeatureEngineer

logger = logging.getLogger("dipex.feature_engineering.engineer")

class EngineeredFeatures:
    """Legacy Result object for backward compatibility."""
    def __init__(self, df: pd.DataFrame, report_dict: Dict[str, Any], original_df: Optional[pd.DataFrame] = None):
        self.df = df
        self.manifest = report_dict
        self.features_added:   List[str] = report_dict.get("features_added", [])
        # B10 fix: surface features_pruned from report (columns dropped during encoding / cardinality pruning)
        self.features_pruned:  List[str] = report_dict.get("features_pruned", [])
        self.elapsed_ms:       float     = float(report_dict.get("elapsed_ms", 0.0))
        self.original_shape    = original_df.shape if original_df is not None else (0, 0)
        self.final_shape       = df.shape if df is not None else (0, 0)

    def to_dict(self) -> Dict:
        return {
            "features_added":       self.features_added,
            "features_pruned":      self.features_pruned,
            "net_features_added":   len(self.features_added) - len(self.features_pruned),
            "original_shape":       list(self.original_shape),
            "final_shape":          list(self.final_shape),
            **self.manifest
        }

class FeatureEngineer:
    """Legacy Proxy for FeatureEngineer — delegates to RobustFeatureEngineer."""
    def __init__(self, config: Optional[Dict] = None, **kwargs):
        self.robust_fe = RobustFeatureEngineer(config=config)

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> EngineeredFeatures:
        """Apply the robust feature engineering via the proxy."""
        original_df   = df.copy()
        df_out, report = self.robust_fe.engineer(df, target_col=target_col)

        report_dict = report.to_dict()

        # B10 fix: derive features_pruned = columns in original that aren't in output,
        # EXCLUDING columns that were simply replaced by engineered versions.
        original_cols = set(original_df.columns)
        final_cols    = set(df_out.columns) if df_out is not None else set()
        engineered_set = set(report_dict.get("features_added", []))

        # A column is "pruned" if it disappeared AND we didn't explicitly add it back under a new name
        pruned = sorted(original_cols - final_cols)
        report_dict["features_pruned"] = pruned

        logger.debug(
            "[FeatureEngineer proxy] added=%d, pruned=%d, net=%d",
            len(engineered_set), len(pruned), len(engineered_set) - len(pruned)
        )
        return EngineeredFeatures(df_out, report_dict, original_df=original_df)
