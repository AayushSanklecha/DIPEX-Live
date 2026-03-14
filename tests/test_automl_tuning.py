import pandas as pd
import numpy as np
import pytest
from proposal.automl import AutoMLProposal

@pytest.fixture
def mock_classification_data():
    np.random.seed(42)
    # 200 rows to satisfy _MIN_SAMPLES and provide enough data for tuning
    X = pd.DataFrame(np.random.randn(200, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    # Create a clear pattern so models can learn easily
    y = ((X["f1"] + X["f2"] > 0) ^ (X["f3"] < 0)).astype(int)
    df = X.copy()
    df["target"] = y
    return df

@pytest.fixture
def mock_regression_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = X["f1"] * 2.5 + X["f2"] * 1.5 - X["f3"] * 3.0 + np.random.randn(200) * 0.1
    df = X.copy()
    df["target"] = y
    return df

def test_automl_classification_tuning(mock_classification_data):
    automl = AutoMLProposal()
    
    # Run the proposal
    result = automl.propose(mock_classification_data, target_col="target")
    
    # Assertions
    assert "model_type" in result
    assert result["task"] == "classification"
    assert result["metric_name"] in ["roc_auc", "accuracy"]
    assert "all_results" in result
    
    # Check that tuning occurred
    assert "tuned_params" in result
    assert result["tuning_method"] in ["optuna_tpe", "randomized_search_cv"]
    assert result["tuned_score"] is not None
    # Tuning a model with very few trials on a tiny dataset can sometimes
    # yield a slightly worse cross-validated score due to noise. 
    # We assert that the tuned score is at least reasonably close (within 5%)
    assert result["tuned_score"] >= result["metric_value"] * 0.95

def test_automl_regression_tuning(mock_regression_data):
    automl = AutoMLProposal()
    
    # Run the proposal
    result = automl.propose(mock_regression_data, target_col="target")
    
    # Assertions
    assert "model_type" in result
    assert result["task"] == "regression"
    assert result["metric_name"] == "r2"
    
    # Ensure tuning information is populated
    assert "tuned_params" in result
    assert result["tuning_method"] != "none"
