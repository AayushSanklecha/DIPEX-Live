"""
feature_engineering/engineer.py
---------------------------------
LEGACY PROXY — AI & ANALYTICS SERVICE LAYER — Feature Engineering

This file is now a thin proxy to preprocessing/feature_engineer.py to ensure
all enterprise robustness fixes are applied centrally.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import pandas as pd
from preprocessing.feature_engineer import FeatureEngineer as RobustFeatureEngineer

logger = logging.getLogger("dipex.feature_engineering.engineer")

class EngineeredFeatures:
    """Legacy Result object for backward compatibility."""
    def __init__(self, df: pd.DataFrame, report_dict: Dict[str, Any]):
        self.df = df
        self.manifest = report_dict
        self.features_added = report_dict.get("features_added", [])
        self.features_pruned = []
        self.elapsed_ms = 0.0
        self.original_shape = (0, 0)
        self.final_shape = df.shape if df is not None else (0, 0)

    def to_dict(self) -> Dict:
        return {
            "features_added": self.features_added,
            "net_features_added": len(self.features_added),
            "final_shape": list(self.final_shape),
            **self.manifest
        }

class FeatureEngineer:
    """Legacy Proxy for FeatureEngineer."""
    def __init__(self, config: Optional[Dict] = None, **kwargs):
        self.robust_fe = RobustFeatureEngineer(config=config)

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> EngineeredFeatures:
        """Apply the robust feature engineering via the proxy."""
        df_out, report = self.robust_fe.engineer(df, target_col=target_col)
        return EngineeredFeatures(df_out, report.to_dict())
