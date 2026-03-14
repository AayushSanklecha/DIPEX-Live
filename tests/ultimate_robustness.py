import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import logging

# Enable debug logging to see internal FeatureEngineer prints
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineer import FeatureEngineer
from ingestion.pipeline_bridge import PipelineBridge
from ingestion.universal_intake import ISSFSnapshot

def run_ultimate_test():
    print("--- STARTING ULTIMATE ROBUSTNESS TEST ---")
    
    # 1. GENERATE PATHOLOGICAL DATA
    np.random.seed(42)
    rows = 200
    
    # Target with 95% imbalance (10 rows minority)
    target = np.array([0] * 190 + [1] * 10)
    np.random.shuffle(target)
    
    df = pd.DataFrame({
        "all_null": [np.nan] * rows,
        "constant": [42.0] * rows,
        "mixed_type": ["1.0", "2.0", "BAD", "4.0"] * (rows // 4), # 25% bad strings
        "high_card": [f"user_{i}" for i in range(rows)], # 200 unique
        "skewed": np.random.lognormal(mean=0, sigma=2, size=rows), # Skewed Lognormal
        "near_zero_var": [0] * (rows - 5) + [1, 2, 3, 4, 5], # 97.5% same
        "normal_num": np.random.randn(rows),
        "target": target
    })
    
    config = {
        "preprocessing": {
            "drop_col_null_threshold": 0.8,
            "mixed_type_coerce_threshold": 0.3,
            "high_cardinality_limit": 50,
            "auto_log_skew_threshold": 0.5,
            "handle_class_imbalance": True,
            "imbalance_ratio_threshold": 4.0,
            "handle_outliers": False,
            "drop_near_zero_variance": True
        },
        "validation": {
            "strict_mode": False
        },
        "pipeline": {
            "domain": "default",
            "confidence": {"threshold": 0.5} # Ensure it passes
        }
    }
    
    # START TRACING
    cleaner = DataCleaner(config)
    fe = FeatureEngineer(config)
    
    print("\n[STEP 1] DATA CLEANER")
    df_clean, c_report = cleaner.clean(df.copy(), run_id="ultimate")
    print(f"DEBUG: Columns after Cleaner: {df_clean.columns.tolist()}")
    assert "all_null" not in df_clean.columns, "FAILED: all_null not dropped"
    assert "constant" not in df_clean.columns, "FAILED: constant not dropped"
    assert "mixed_type" in df_clean.columns
    assert pd.api.types.is_numeric_dtype(df_clean["mixed_type"]), f"FAILED: mixed_type not coerced. Dtype: {df_clean['mixed_type'].dtype}"
    print("PASSED: Cleaner.")

    print("\n[STEP 2] FEATURE ENGINEER")
    print(f"DEBUG: 'skewed' col skew before FE: {df_clean['skewed'].skew()}")
    df_eng, f_report = fe.engineer(df_clean, run_id="ultimate", target_col="target")
    print(f"DEBUG: Columns after FE: {df_eng.columns.tolist()}")
    assert "high_card_hash_enc" in df_eng.columns, "FAILED: hash encoding not applied"
    assert "skewed_auto_log1p" in df_eng.columns, "FAILED: auto-skewness correction not applied"
    
    # SMOTE CHECK
    final_minority = df_eng["target"].value_counts().min()
    assert final_minority > 20, f"FAILED: SMOTE didn't work (minority count {final_minority})"
    print("PASSED: FeatureEngineer.")

    print("\n[STEP 3] PIPELINE BRIDGE (END-TO-END)")
    snapshot = ISSFSnapshot.from_dataframe(df, dataset_id="ultimate", source_type="file", data_mode="batch")
    bridge = PipelineBridge(config)
    result = bridge.run(snapshot, target_col="target")
    
    assert result.gate1_decision == "PASS"
    assert result.gate_decision in ["PASS", "WARN"]
    print("PASSED: PipelineBridge.")
    
    print("\n--- ULTIMATE TEST COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    try:
        run_ultimate_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
