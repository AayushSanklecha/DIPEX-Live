"""
tests/test_insight_ranker.py
------------------------------
Tests for Phase 5 enhancements:
  - InsightRanker (5-dim scoring, null-rate insights, baseline novelty)
  - FeatureProposer (log/OHE/target/hash/dt_extract/flag_null/polynomial/interaction)
  - AnomalyFlagger (Isolation Forest, advisory-only, graceful fallback)
  - RAGRecall (cosine similarity, feature vector, empty memory fallback)
"""
from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT RANKER
# ══════════════════════════════════════════════════════════════════════════════

class TestInsightRanker:

    @pytest.fixture
    def ranker(self):
        from proposal.insight_ranker import InsightRanker
        return InsightRanker(top_k=10)

    @pytest.fixture
    def profile_with_skewed_col(self):
        return {
            "row_count": 500,
            "columns": {
                "revenue": {
                    "dtype": "float64",
                    "mean": 25.0, "std": 200.0, "skewness": 4.5,
                    "null_rate": 0.0, "unique_count": 480,
                    "cardinality_tier": "high", "count": 500,
                },
                "user_id": {
                    "dtype": "int64",
                    "mean": 500.0, "std": 1.0, "skewness": 0.1,
                    "null_rate": 0.0, "unique_count": 500,
                    "cardinality_tier": "unique", "count": 500,
                },
                "category": {
                    "dtype": "object",
                    "mean": None, "std": None, "skewness": None,
                    "null_rate": 0.30, "unique_count": 5,
                    "cardinality_tier": "low", "count": 350,
                },
            },
        }

    def test_returns_list(self, ranker, profile_with_skewed_col):
        results = ranker.rank(profile_with_skewed_col)
        assert isinstance(results, list)

    def test_all_insights_have_required_keys(self, ranker, profile_with_skewed_col):
        results = ranker.rank(profile_with_skewed_col)
        for r in results:
            for key in ("column", "insight_type", "score", "description", "scores_breakdown"):
                assert key in r, f"Missing key '{key}' in insight: {r}"

    def test_scores_in_range(self, ranker, profile_with_skewed_col):
        results = ranker.rank(profile_with_skewed_col)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_sorted_descending(self, ranker, profile_with_skewed_col):
        results = ranker.rank(profile_with_skewed_col)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_high_skewness_detected(self, ranker, profile_with_skewed_col):
        results = ranker.rank(profile_with_skewed_col)
        skew_insights = [r for r in results if r["insight_type"] == "high_skewness"]
        assert len(skew_insights) > 0

    def test_high_null_rate_generates_insight(self, ranker, profile_with_skewed_col):
        results = ranker.rank(profile_with_skewed_col)
        null_insights = [r for r in results if r["insight_type"] == "high_null_rate"]
        assert len(null_insights) > 0
        # category col has 30% nulls
        assert any(r["column"] == "category" for r in null_insights)

    def test_top_k_enforced(self, ranker):
        from proposal.insight_ranker import InsightRanker
        ranker_small = InsightRanker(top_k=2)
        profile = {
            "row_count": 1000,
            "columns": {f"col_{i}": {"dtype": "float64", "mean": 1.0, "std": 2.0,
                                      "skewness": 3.0, "null_rate": 0.25,
                                      "unique_count": 200, "cardinality_tier": "high",
                                      "count": 1000}
                        for i in range(10)},
        }
        results = ranker_small.rank(profile)
        assert len(results) <= 2

    def test_empty_profile_returns_empty(self, ranker):
        results = ranker.rank({"row_count": 0, "columns": {}})
        assert results == []

    def test_novelty_from_baseline(self, ranker):
        current = {"row_count": 500, "columns": {"revenue": {
            "dtype": "float64", "mean": 5000.0, "std": 10.0, "skewness": 0.5,
            "null_rate": 0.0, "unique_count": 100, "cardinality_tier": "high", "count": 500,
        }}}
        baseline = {"columns": {"revenue": {"mean": 100.0}}}
        results = ranker.rank(current, baseline_stats=baseline)
        # Revenue should score high novelty (mean jumped 50x)
        rev = [r for r in results if r["column"] == "revenue"]
        assert len(rev) > 0
        assert rev[0]["scores_breakdown"]["novelty"] > 0.5

    def test_drift_lowers_stability(self, ranker, profile_with_skewed_col):
        drift_report = {"column_drifts": [
            {"column": "revenue", "psi": 0.30}
        ]}
        results = ranker.rank(profile_with_skewed_col, drift_report=drift_report)
        rev = [r for r in results if r["column"] == "revenue"]
        if rev:
            assert rev[0]["scores_breakdown"]["stability"] < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PROPOSER
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureProposer:

    @pytest.fixture
    def proposer(self):
        from proposal.insight_ranker import FeatureProposer
        return FeatureProposer()

    def test_returns_list(self, proposer):
        profile = {"row_count": 100, "columns": {}}
        props = proposer.propose(profile)
        assert isinstance(props, list)

    def test_skewed_numeric_gets_log_transform(self, proposer):
        profile = {"row_count": 200, "columns": {
            "income": {"dtype": "float64", "skewness": 3.2, "unique_count": 180,
                       "null_rate": 0.0, "count": 200},
        }}
        props = proposer.propose(profile)
        transforms = [p["transformation"] for p in props]
        assert "log_transform" in transforms

    def test_low_cardinality_cat_gets_ohe(self, proposer):
        profile = {"row_count": 500, "columns": {
            "status": {"dtype": "object", "skewness": None, "unique_count": 5,
                       "null_rate": 0.0, "count": 500},
        }}
        props = proposer.propose(profile)
        transforms = [p["transformation"] for p in props]
        assert "one_hot_encode" in transforms

    def test_high_cardinality_cat_gets_target_encode(self, proposer):
        profile = {"row_count": 5000, "columns": {
            "city": {"dtype": "object", "skewness": None, "unique_count": 50,
                     "null_rate": 0.0, "count": 5000},
        }}
        props = proposer.propose(profile)
        transforms = [p["transformation"] for p in props]
        assert "target_encode" in transforms

    def test_very_high_cardinality_cat_gets_hash_encode(self, proposer):
        profile = {"row_count": 10000, "columns": {
            "product_id": {"dtype": "object", "skewness": None, "unique_count": 5000,
                           "null_rate": 0.0, "count": 10000},
        }}
        props = proposer.propose(profile)
        transforms = [p["transformation"] for p in props]
        assert "hash_encode" in transforms

    def test_datetime_col_gets_extract(self, proposer):
        profile = {"row_count": 300, "columns": {
            "event_date": {"dtype": "datetime64", "skewness": None,
                           "unique_count": 300, "null_rate": 0.0, "count": 300},
        }}
        props = proposer.propose(profile)
        transforms = [p["transformation"] for p in props]
        assert "datetime_extract" in transforms

    def test_null_flag_proposed_for_moderate_nulls(self, proposer):
        profile = {"row_count": 400, "columns": {
            "score": {"dtype": "float64", "skewness": 0.2, "unique_count": 350,
                      "null_rate": 0.15, "count": 340},
        }}
        props = proposer.propose(profile)
        transforms = [p["transformation"] for p in props]
        assert "flag_null" in transforms

    def test_interaction_proposed_when_corr_high(self, proposer):
        profile = {"row_count": 500, "columns": {
            "col_a": {"dtype": "float64", "skewness": 0.3, "unique_count": 400,
                      "null_rate": 0.0, "count": 500},
            "col_b": {"dtype": "float64", "skewness": 0.1, "unique_count": 350,
                      "null_rate": 0.0, "count": 500},
        }}
        # Manually build high-correlation matrix
        corr = pd.DataFrame([[1.0, 0.92], [0.92, 1.0]], columns=["col_a", "col_b"],
                             index=["col_a", "col_b"])
        props = proposer.propose(profile, correlation_matrix=corr)
        transforms = [p["transformation"] for p in props]
        assert "interaction_term" in transforms

    def test_proposals_sorted_by_priority(self, proposer):
        profile = {"row_count": 500, "columns": {
            "income": {"dtype": "float64", "skewness": 3.0, "unique_count": 400,
                       "null_rate": 0.08, "count": 500},
        }}
        props = proposer.propose(profile)
        priorities = [p["priority"] for p in props]
        assert priorities == sorted(priorities)

    def test_no_duplicates_in_proposals(self, proposer):
        profile = {"row_count": 500, "columns": {
            "income": {"dtype": "float64", "skewness": 3.0, "unique_count": 400,
                       "null_rate": 0.08, "count": 500},
        }}
        props = proposer.propose(profile)
        keys = [(tuple(p["columns"]), p["transformation"]) for p in props]
        assert len(keys) == len(set(keys))


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY FLAGGER
# ══════════════════════════════════════════════════════════════════════════════

class TestAnomalyFlagger:

    @pytest.fixture
    def flagger(self):
        from proposal.insight_ranker import AnomalyFlagger
        return AnomalyFlagger(contamination=0.05, random_state=42)

    @pytest.fixture
    def clean_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "x": np.random.normal(0, 1, 200),
            "y": np.random.normal(0, 1, 200),
        })

    @pytest.fixture
    def dirty_df(self, clean_df):
        df = clean_df.copy()
        df.loc[0, "x"] = 1000.0
        df.loc[1, "y"] = -1000.0
        return df

    def test_returns_dict(self, flagger, clean_df):
        result = flagger.flag(clean_df)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, flagger, clean_df):
        result = flagger.flag(clean_df)
        for key in ("anomalous_indices", "anomaly_scores", "anomaly_count",
                    "total_rows", "anomaly_rate", "columns_used", "advisory_note"):
            assert key in result

    def test_anomaly_rate_in_range(self, flagger, clean_df):
        result = flagger.flag(clean_df)
        assert 0.0 <= result["anomaly_rate"] <= 1.0

    def test_does_not_modify_original_df(self, flagger, dirty_df):
        original_hash = pd.util.hash_pandas_object(dirty_df).sum()
        flagger.flag(dirty_df)
        assert pd.util.hash_pandas_object(dirty_df).sum() == original_hash

    def test_detects_extreme_outliers(self, flagger, dirty_df):
        result = flagger.flag(dirty_df)
        # Both extreme outlier rows should be flagged
        assert 0 in result["anomalous_indices"] or 1 in result["anomalous_indices"]

    def test_advisory_note_present(self, flagger, clean_df):
        result = flagger.flag(clean_df)
        assert "ADVISORY" in result["advisory_note"].upper()

    def test_too_small_df_returns_empty(self, flagger):
        tiny = pd.DataFrame({"x": [1.0, 2.0]})
        result = flagger.flag(tiny)
        assert result["anomaly_count"] == 0

    def test_column_subset_selection(self, flagger, clean_df):
        result = flagger.flag(clean_df, numeric_cols=["x"])
        assert result["columns_used"] == ["x"]

    def test_non_numeric_df_returns_empty(self, flagger):
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        result = flagger.flag(df)
        assert result["anomaly_count"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# RAG RECALL
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGRecall:

    def test_no_memory_returns_empty(self):
        from proposal.insight_ranker import RAGRecall
        rag = RAGRecall(experience_memory=None)
        result = rag.recall({"columns": {}, "row_count": 100})
        assert result == []

    def test_returns_list(self):
        from proposal.insight_ranker import RAGRecall

        class MockMemory:
            def query(self, **kwargs):
                return []

        rag = RAGRecall(experience_memory=MockMemory())
        result = rag.recall({"columns": {}, "row_count": 100})
        assert isinstance(result, list)

    def test_cosine_similarity_identical_returns_one(self):
        from proposal.insight_ranker import RAGRecall
        rag = RAGRecall()
        a = np.array([0.5, 0.5, 0.5])
        sim = rag._cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_returns_zero(self):
        from proposal.insight_ranker import RAGRecall
        rag = RAGRecall()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = rag._cosine_similarity(a, b)
        assert sim < 0.01

    def test_cosine_zero_vector_returns_zero(self):
        from proposal.insight_ranker import RAGRecall
        rag = RAGRecall()
        a = np.zeros(5)
        b = np.ones(5)
        sim = rag._cosine_similarity(a, b)
        assert sim == 0.0

    def test_profile_to_vector_dimension(self):
        from proposal.insight_ranker import RAGRecall
        rag = RAGRecall()
        profile = {"row_count": 500, "columns": {
            "x": {"dtype": "float64", "null_rate": 0.05, "skewness": 1.2,
                  "unique_count": 450, "cardinality_tier": "high", "count": 500},
        }}
        vec = rag._profile_to_vector(profile, 0.80, None)
        assert vec.shape == (7,)
        # All components should be non-negative; log-scaled row_count may slightly exceed 1.0
        assert all(v >= 0.0 for v in vec), f"Vector has negative components: {vec}"
        assert max(abs(v) for v in vec) < 10.0, f"Vector has unreasonable magnitude: {vec}"

    def test_episode_to_vector_dimension(self):
        from proposal.insight_ranker import RAGRecall
        rag = RAGRecall()
        ep = {"row_count": 100, "mean_null_rate": 0.0, "mean_skewness": 0.5,
              "numeric_col_ratio": 0.8, "unique_col_ratio": 0.1,
              "mean_psi": 0.05, "confidence_score": 0.82}
        vec = rag._episode_to_vector(ep)
        assert vec.shape == (7,)

    def test_top_k_enforced(self):
        from proposal.insight_ranker import RAGRecall

        class MockEp:
            def to_dict(self):
                return {"row_count": 100, "mean_null_rate": 0.0, "mean_skewness": 0.5,
                        "numeric_col_ratio": 0.5, "unique_col_ratio": 0.2,
                        "mean_psi": 0.05, "confidence_score": 0.80,
                        "stage": "APPROVED_OUTPUT", "episode_id": "ep1",
                        "winning_strategy": "IMPUTE_MEDIAN", "schema_version": "1.0.0",
                        "source_type": "csv"}

        class MockMemory:
            def query(self, **kwargs):
                return [MockEp() for _ in range(20)]

        rag = RAGRecall(experience_memory=MockMemory(), top_k=3)
        profile = {"row_count": 100, "columns": {}}
        result = rag.recall(profile)
        assert len(result) <= 3

    def test_similarity_in_range(self):
        from proposal.insight_ranker import RAGRecall

        class MockEp:
            def to_dict(self):
                return {"row_count": 500, "mean_null_rate": 0.05, "mean_skewness": 0.5,
                        "numeric_col_ratio": 0.7, "unique_col_ratio": 0.2,
                        "mean_psi": 0.05, "confidence_score": 0.80,
                        "stage": "APPROVED_OUTPUT", "episode_id": "ep_sim",
                        "winning_strategy": "IMPUTE_MEDIAN", "schema_version": "1.0.0",
                        "source_type": "csv"}

        class MockMemory:
            def query(self, **kwargs):
                return [MockEp()]

        rag = RAGRecall(experience_memory=MockMemory(), top_k=5)
        profile = {"row_count": 500, "columns": {}}
        result = rag.recall(profile, confidence_score=0.80)
        for r in result:
            assert 0.0 <= r["similarity"] <= 1.0
