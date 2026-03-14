import sys
import os
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
from preprocessing.feature_engineer import FeatureEngineer

class GuardedFE(FeatureEngineer):
    def engineer(self, df, run_id="", target_col=None):
        print("START engineer")
        stages = [
            ("_lag_features", self._lag_features),
            ("_rolling_features", self._rolling_features),
            ("_calendar_features", self._calendar_features),
            ("_frequency_encode", self._frequency_encode),
            ("_target_encode", lambda d, r: self._target_encode(d, r, target_col)),
            ("_log_transform", self._log_transform),
            ("_auto_log_correction", self._auto_log_correction),
            ("_polynomial_features", self._polynomial_features),
            ("_binning", self._binning),
            ("_interactions", self._interactions),
            ("_zscore_scale", self._zscore_scale),
            ("_minmax_scale", self._minmax_scale),
            ("_synthesize_features", lambda d, r: self._synthesize_features(d, r, target_col)),
            ("_encode_remaining_objects", self._encode_remaining_objects),
            ("_handle_class_imbalance", lambda d, r: self._handle_class_imbalance(d, r, target_col))
        ]
        
        report = type('Report', (), {'features_added': [], 'transformations_applied': [], 'warnings': []})()
        current_df = df.copy()
        for name, fn in stages:
            before = current_df.columns.tolist()
            current_df = fn(current_df, report)
            after = current_df.columns.tolist()
            dropped = [c for c in before if c not in after]
            if dropped:
                print(f"!!! DROP in {name}: {dropped}")
            else:
                pass
                # print(f"Stage {name} passed.")
        return current_df, report

df = pd.DataFrame({
    "skewed": np.random.lognormal(0, 1.5, 100),
    "target": np.random.randint(0, 2, 100)
})

fe = GuardedFE({'preprocessing': {'auto_log_skew_threshold': 0.1, 'handle_class_imbalance': False}})
fe.engineer(df, target_col='target')
print("DONE")
