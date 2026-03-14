import numpy as np
import pandas as pd
import pytest

from preprocessing.anomaly_scorer import AnomalyScorer, AnomalyReport
from preprocessing.robust_triage import RobustTriage

@pytest.fixture
def anomaly_df():
    # 100 normal rows
    np.random.seed(42)
    normal_data = {
        "age": np.random.normal(35, 5, 100),
        "income": np.random.normal(60000, 10000, 100),
        "credit_score": np.random.normal(700, 50, 100)
    }
    df = pd.DataFrame(normal_data)
    
    # Add 5 highly anomalous rows
    anomalies = pd.DataFrame({
        "age": [120, 18, 99, 15, 110],
        "income": [1000000, 10, 5000000, 0, 2000000],
        "credit_score": [300, 850, 400, 300, 850]
    })
    
    df = pd.concat([df, anomalies], ignore_index=True)
    return df

@pytest.fixture
def zero_heavy_df():
    # Column with 60% zeros
    np.random.seed(42)
    revenue = np.random.normal(100, 10, 100)
    revenue[:60] = 0.0  # 60 zeros out of 100
    
    # Column with 10% zeros
    age = np.random.normal(40, 5, 100)
    age[:10] = 0.0 # 10 zeros
    
    return pd.DataFrame({
        "revenue": revenue,
        "age": age,
        "target": np.random.randint(0, 2, 100)
    })

def test_anomaly_scorer(anomaly_df):
    scorer = AnomalyScorer()
    scored_df, report = scorer.score(anomaly_df, run_id="test1")
    
    assert "anomaly_score" in scored_df.columns
    assert "anomaly_flag" in scored_df.columns
    
    assert report.severity in ["OK", "WARNING"]
    
    # The last 5 rows should generally have a lower score than the rest
    normal_scores = scored_df.iloc[:100]["anomaly_score"].mean()
    anomaly_scores = scored_df.iloc[100:]["anomaly_score"].mean()
    
    assert anomaly_scores < normal_scores
    
    # Ensure at least some anomalies are flagged
    assert sum(scored_df["anomaly_flag"] == -1) > 0


def test_zero_value_detection(zero_heavy_df):
    triage = RobustTriage()  # Defaults: high_zero_threshold=0.50
    triaged_df, report = triage.triage(zero_heavy_df, target_col="target")
    
    # The 'revenue' column has 60% zeros, so it should be flagged
    flagged = [x["column"] for x in report.zero_flagged_columns]
    
    assert "revenue" in flagged
    assert "age" not in flagged # only 10% zeros
