import os
import pandas as pd
import numpy as np
import pytest
from validation.drift_detector import SchemaDriftDetector, DriftReport

@pytest.fixture
def base_df():
    # Initial steady-state data
    np.random.seed(42)
    return pd.DataFrame({
        "revenue": np.random.normal(1000, 100, 1000),
        "age": np.random.normal(35, 5, 1000),
        "category": ["A"] * 500 + ["B"] * 500
    })

@pytest.fixture
def drifted_df():
    # Data with a huge mathematical drift in 'revenue' and a deleted column
    np.random.seed(42)
    return pd.DataFrame({
        "revenue": np.random.normal(50000, 10000, 1000),  # Massive drift
        "category": ["A"] * 500 + ["B"] * 500
        # "age" column removed (schema contract violation)
    })

def test_retrain_required_trigger(base_df, drifted_df, tmp_path):
    detector = SchemaDriftDetector(config={"validation": {"drift": {"registry_dir": str(tmp_path)}}})
    
    # 1. Run the base data (First run creates fingerprint)
    report_base = detector.detect(base_df, dataset_id="test_drift_retrain")
    
    assert report_base.is_first_run is True
    assert report_base.retrain_required is False
    assert len(report_base.violations) == 0
    
    # 2. Run the drifted data against the fingerprint
    report_drifted = detector.detect(drifted_df, dataset_id="test_drift_retrain")
    
    # Assertions
    assert report_drifted.is_first_run is False
    
    # We expect 2 violations: 'age' removed, and 'revenue' drifted
    assert len(report_drifted.violations) >= 2
    
    drift_types = [v.drift_type for v in report_drifted.violations]
    assert "column_removed" in drift_types
    assert "distribution_drift" in drift_types
    
    # Crucially, the flag should be True
    assert report_drifted.retrain_required is True
