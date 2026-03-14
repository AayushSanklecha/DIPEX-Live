import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, r"c:\Users\sankl\Desktop\dipex_project")

from main import load_config
from ingestion.pipeline_bridge import PipelineBridge
from ingestion.universal_intake import ISSFSnapshot

def test_messy_data_robustness():
    """Validates the 5 real-world robustness features added in v1.3"""
    config = load_config("config.yaml")

    # Define a highly pathological dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(100),
        "target": np.random.choice([0, 1], size=100, p=[0.9, 0.1]),  # Imbalanced
        "all_null_col": [np.nan] * 100,                              # High-null drop
        "mixed_obj_col": ["1.1", "2.2", "A", "4.4"] * 25,            # Coercion
        "constant_col": [42] * 100,                                  # Zero variance
        "high_cardinality_cat": [f"val_{i}" for i in range(100)],    # Hash encode
        "normal_num": np.random.randn(100)
    })

    # Wrap in ISSF
    snapshot = ISSFSnapshot.from_dataframe(
        df=df,
        dataset_id="test_messy",
        snapshot_id="snap_1",
        source_type="file",
        schema_version="1.0"
    )

    # Make gating non-strict so we downgrade rather than reject
    config["validation"]["strict_mode"] = False
    
    # Configure feature engineering to target the categorical column
    if "preprocessing" not in config:
        config["preprocessing"] = {}
    config["preprocessing"]["frequency_encode"] = ["high_cardinality_cat"]
    config["preprocessing"]["high_cardinality_limit"] = 50
    config["preprocessing"]["mixed_type_coerce_threshold"] = 0.30
    
    # 1. Test the new Robust Modules directly first
    from preprocessing.cleaner import DataCleaner
    from preprocessing.feature_engineer import FeatureEngineer
    
    cleaner = DataCleaner(config)
    fe = FeatureEngineer(config)
    
    # Clean the data
    df_clean, clean_report = cleaner.clean(df.copy(), run_id="test_messy")
    
    # Assert drop behavior
    assert "all_null_col" not in df_clean.columns, "Failed to drop high-null column"
    assert "constant_col" not in df_clean.columns, "Failed to drop constant column"
    
    # Assert coercion
    if "mixed_obj_col" in df_clean.columns:
        assert pd.api.types.is_numeric_dtype(df_clean["mixed_obj_col"]), "Failed to coerce mixed type"
        
    # Feature Engineering
    df_engineered, fe_report = fe.engineer(df_clean, run_id="snap_1", target_col="target")
    
    # Assert hash encoding fallback
    assert "high_cardinality_cat_hash_enc" in df_engineered.columns, "Failed to apply hash encoding fallback"
    assert "high_cardinality_cat" not in df_engineered.columns, "Original high cardinality column not dropped"
    
    # Assert SMOTE (if target was processed and we passed the limit)
    final_target_counts = df_engineered["target"].value_counts()
    if len(final_target_counts) > 1:
        assert final_target_counts.min() > 10, "SMOTE should have increased minority class representation"

    # 2. Test PipelineBridge (End-to-End no-crash guarantee)
    bridge = PipelineBridge(config)
    result = bridge.run(snapshot, target_col="target")
    
    assert result.gate_decision in ["PASS", "WARN"], "Pipeline was unfairly rejected!"
    print("ALL ROBUSTNESS TESTS PASSED SUCCESSFULLY.")
    
if __name__ == "__main__":
    test_messy_data_robustness()
