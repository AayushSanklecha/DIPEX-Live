"""
tests/test_proposal_layer.py
-----------------------------
Comprehensive test suite for Step 4 — Proposal Layer.
"""

import pytest
import pandas as pd
import numpy as np
import os
import shutil
from proposal.proposal_engine import ProposalEngine
from proposal.proposers.automl_proposer import AutoMLProposer
from proposal.proposers.anomaly_proposer import AnomalyProposer
from proposal.proposers.ranker_proposer import RankerProposer
from proposal.proposers.bandit_proposer import BanditProposer
from proposal.proposers.aggregation_proposer import AggregationProposer
from proposal.proposers.transformation_proposer import TransformationProposer
from proposal.proposers.encoding_proposer import EncodingProposer
from proposal.proposers.window_proposer import WindowProposer
from proposal.rag.experience_recall import ExperienceRecall

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n) * 10,
        "feature_3": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n)
    })
    # Add some outliers to feature_2
    df.loc[:4, "feature_2"] = 500.0
    return df

@pytest.fixture
def sample_config():
    return {
        "pipeline": {"target_column": "target"},
        "proposal": {
            "anomaly": {"n_estimators": 10, "contamination": 0.05},
            "rag": {
                "db_path": "data/test_chroma_db",   # RAGRetriever uses db_path
                "storage_path": "data/test_experience.json",  # kept for compatibility
            },
            "bandit": {"storage_path": "data/test_bandit.json"}
        }
    }

def test_automl_proposer(sample_df):
    proposer = AutoMLProposer()
    res = proposer.propose(sample_df, target_col="target")
    assert "error" not in res
    assert res["primary_task"] == "classification"
    assert len(res["candidates"]) > 0
    assert res["candidates"][0]["model_type"] == "RandomForest"

def test_anomaly_proposer(sample_df, sample_config):
    proposer = AnomalyProposer(sample_config)
    res = proposer.propose(sample_df)
    assert "error" not in res
    candidates = res["anomaly_candidates"]
    assert candidates["detected_outlier_count"] > 0
    assert "suggested_threshold" in candidates

def test_ranker_proposer(sample_df):
    proposer = RankerProposer()
    res = proposer.propose(sample_df, target_col="target")
    assert "error" not in res
    rankings = res["feature_importance_candidates"]
    assert len(rankings) > 0
    assert rankings[0]["column"] in ["feature_1", "feature_2"]

def test_bandit_proposer(sample_df, sample_config):
    # Ensure clean state
    if os.path.exists("data/test_bandit.json"):
        os.remove("data/test_bandit.json")
    
    proposer = BanditProposer(sample_config)
    res = proposer.propose(sample_df, contexts=["retry_strategy"])
    assert "strategy_candidates" in res
    assert "retry_strategy" in res["strategy_candidates"]
    assert "recommendation" in res["strategy_candidates"]["retry_strategy"]


def test_aggregation_proposer(sample_df):
    proposer = AggregationProposer()
    res = proposer.propose(sample_df)
    assert "error" not in res
    assert "aggregation_candidates" in res


def test_transformation_proposer(sample_df):
    proposer = TransformationProposer()
    res = proposer.propose(sample_df)
    assert "error" not in res
    # Some numeric columns should have at least one suggested transform
    assert "transformation_candidates" in res


def test_encoding_proposer(sample_df):
    proposer = EncodingProposer()
    res = proposer.propose(sample_df)
    assert "error" not in res
    assert "encoding_candidates" in res


def test_window_proposer(sample_df):
    # Add a synthetic timestamp column to drive window suggestions
    df = sample_df.copy()
    df["event_time"] = pd.date_range("2024-01-01", periods=len(df), freq="min")
    proposer = WindowProposer()
    res = proposer.propose(df)
    assert "error" not in res
    assert "window_candidates" in res

def test_experience_recall(tmp_path, sample_df, sample_config):
    """Test ExperienceRecall using RAGRetriever (ChromaDB-backed semantic store)."""
    import shutil
    # Use tmp_path so each test run gets a clean ChromaDB store
    db_path = str(tmp_path / "test_chroma_db")
    config = dict(sample_config)
    config["proposal"] = dict(config["proposal"])
    config["proposal"]["rag"] = {"db_path": db_path}

    recall = ExperienceRecall(config)

    # Store an experience — must complete without error
    recall.store_experience("run_1", sample_df, {
        "accuracy": 0.95,
        "dataset_id": "test_ds",
        "confidence_score": 0.95,
    })

    # Recall — same DataFrame should return a highly-similar result
    results = recall.recall(sample_df, top_k=1)
    assert len(results) > 0, "ExperienceRecall.recall should return at least 1 result after storing one run"

    # Validate result structure from RAGRetriever
    result = results[0]
    assert "id" in result,               "Result must have 'id' key"
    assert "relevance_score" in result,  "Result must have 'relevance_score' key"
    assert result["id"] == "run_1",     f"Expected id=run_1, got {result['id']}"
    assert result["relevance_score"] > 0.5, (
        f"Same-DataFrame recall should be highly similar, got {result['relevance_score']}"
    )

def test_proposal_engine_full(sample_df, sample_config):
    engine = ProposalEngine(sample_config)
    res = engine.generate_proposals(sample_df, run_id="test_run")
    assert res["run_id"] == "test_run"
    assert "automl" in res["candidates"]
    assert "anomaly" in res["candidates"]
    assert "ranker" in res["candidates"]
    assert "bandit" in res["candidates"]
    assert "aggregation" in res["candidates"]
    assert "transformation" in res["candidates"]
    assert "encoding" in res["candidates"]
    assert "window" in res["candidates"]
    assert "historical_precedents" in res
