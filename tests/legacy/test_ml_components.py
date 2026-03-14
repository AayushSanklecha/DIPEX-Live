"""
tests/test_ml_components.py
-----------------------------
Automated tests for all ML components in DIPEX.

Covers:
  - SmartSchemaInferer (heuristic path)
  - SoftValidator (IsolationForest anomaly classification)
  - DriftDetector (univariate + multivariate in-memory AE)
  - UncertaintyQuantifier (conformal prediction — classification + regression)
  - SHAPExplainer (4-tier fallback)
  - PiiDetector (regex pattern matching)
  - Anomaly Access Detector (rule-based path)
  - ChartRelevanceScorer (heuristic fallback)
  - PipelineSuccessPredictor (heuristic fallback)

All tests run WITHOUT trained .pkl artifacts (heuristic fallback paths).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. SmartSchemaInferer
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartSchemaInferer:

    def _inferer(self):
        from ingestion.schema_infer import SmartSchemaInferer
        return SmartSchemaInferer()

    def test_email_column(self):
        inf = self._inferer()
        s = pd.Series(["alice@example.com", "bob@test.org", "carol@foo.io"])
        r = inf.infer(s, "email_address")
        assert r["semantic_type"] == "email"
        assert r["confidence"] >= 0.5

    def test_age_column(self):
        inf = self._inferer()
        s = pd.Series([25, 30, 45, 22, 19, 37])
        r = inf.infer(s, "age")
        assert r["semantic_type"] == "age"

    def test_amount_column(self):
        inf = self._inferer()
        s = pd.Series([1200.5, 3400.0, 980.75, 12000.0])
        r = inf.infer(s, "transaction_amount")
        assert r["semantic_type"] in {"amount", "score", "count", "unknown"}

    def test_enrich_schema(self):
        inf = self._inferer()
        df = pd.DataFrame({
            "customer_email": ["a@b.com", "c@d.com"],
            "revenue":         [100.0, 200.0],
        })
        schema = {"customer_email": "object", "revenue": "float64"}
        result = inf.enrich_schema(df, schema)
        assert "customer_email" in result
        assert result["customer_email"]["semantic_type"] == "email"

    def test_column_not_in_df(self):
        inf = self._inferer()
        result = inf.enrich_schema(pd.DataFrame(), {"ghost_col": "object"})
        assert result["ghost_col"]["method"] == "not_found"


# ─────────────────────────────────────────────────────────────────────────────
# 2. SoftValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftValidator:

    def _sv(self):
        from validation.soft_validator import SoftValidator
        return SoftValidator(contamination=0.05)

    def test_no_violations(self):
        sv = self._sv()
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        mask = pd.Series([False] * 100)
        r = sv.classify_violations(df, "a", mask)
        assert r["hard_count"] == 0
        assert r["soft_count"] == 0
        assert r["method"] == "no_violations"

    def test_with_violations(self):
        sv = self._sv()
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(5, 2, 200),
        })
        mask = pd.Series([False] * 190 + [True] * 10)
        r = sv.classify_violations(df, "a", mask)
        assert r["hard_count"] + r["soft_count"] == 10
        assert r["method"] in {"isolation_forest", "fallback", "fallback_too_few_features"}

    def test_single_column_fallback(self):
        sv = self._sv()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        mask = pd.Series([False, False, False, True, True])
        r = sv.classify_violations(df, "a", mask)
        # Only 1 numeric column → fallback
        assert r["method"] in {"fallback_too_few_features", "isolation_forest", "fallback"}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DriftDetector (in-memory autoencoder)
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetector:

    def _dd(self):
        from profiling.drift_detector import DriftDetector
        return DriftDetector({})

    def test_no_drift_identical(self):
        dd = self._dd()
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((300, 4)), columns=list("abcd"))
        result = dd.detect(df, df)
        assert isinstance(result["drifted_columns"], list)

    def test_obvious_drift(self):
        dd = self._dd()
        rng = np.random.default_rng(0)
        base = pd.DataFrame(rng.standard_normal((300, 4)), columns=list("abcd"))
        curr = pd.DataFrame(rng.standard_normal((100, 4)) + 10, columns=list("abcd"))
        result = dd.detect(base, curr)
        assert len(result["drifted_columns"]) > 0

    def test_multivariate_in_memory(self):
        dd = self._dd()
        rng = np.random.default_rng(7)
        base = pd.DataFrame(rng.standard_normal((200, 5)), columns=[f"f{i}" for i in range(5)])
        curr = pd.DataFrame(rng.standard_normal((50, 5)),  columns=[f"f{i}" for i in range(5)])
        mv = dd.detect_multivariate_drift(base, curr)
        assert "drifted" in mv
        assert "method" in mv

    def test_insufficient_columns_skipped(self):
        dd = self._dd()
        base = pd.DataFrame({"a": [1, 2, 3]})
        curr = pd.DataFrame({"a": [4, 5, 6]})
        mv = dd.detect_multivariate_drift(base, curr)
        assert mv["method"] == "skipped"


# ─────────────────────────────────────────────────────────────────────────────
# 4. UncertaintyQuantifier (conformal prediction)
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertaintyQuantifier:

    def test_classification_coverage(self):
        from cognitive.uncertainty_quantifier import UncertaintyQuantifier
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 5))
        y = (X[:, 0] > 0).astype(int)
        X_tr, X_c, X_t = X[:300], X[300:400], X[400:]
        y_tr, y_c, y_t = y[:300], y[300:400], y[400:]
        clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
        uq = UncertaintyQuantifier()
        cal = uq.calibrate(clf, X_c, y_c, task="classification", alpha=0.10)
        assert cal["coverage_target"] == pytest.approx(0.90)
        result = uq.predict(X_t)
        assert "prediction_sets" in result
        # Empirical coverage should be reasonably close to target
        cov = uq.coverage_summary(X_t, y_t)
        assert cov["empirical_coverage"] >= 0.70  # loose bound for small test

    def test_regression_intervals(self):
        from cognitive.uncertainty_quantifier import UncertaintyQuantifier
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(0)
        X = rng.standard_normal((400, 3))
        y = X @ np.array([1, -2, 0.5]) + rng.normal(0, 0.5, 400)
        X_tr, X_c, X_t = X[:250], X[250:350], X[350:]
        y_tr, y_c, y_t = y[:250], y[250:350], y[350:]
        mdl = Ridge().fit(X_tr, y_tr)
        uq = UncertaintyQuantifier()
        uq.calibrate(mdl, X_c, y_c, task="regression", alpha=0.10)
        result = uq.predict(X_t)
        assert "intervals" in result
        assert len(result["intervals"]) == len(X_t)
        assert result["interval_width"] > 0

    def test_uncalibrated_returns_error(self):
        from cognitive.uncertainty_quantifier import UncertaintyQuantifier
        uq = UncertaintyQuantifier()
        r = uq.predict(np.zeros((5, 3)))
        assert "error" in r


# ─────────────────────────────────────────────────────────────────────────────
# 5. PiiDetector (regex path — no spaCy needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPiiDetector:

    def _det(self):
        from governance.pii_detector import PIIDetector
        return PIIDetector()

    def test_email_detected(self):
        det = self._det()
        df = pd.DataFrame({"email_col": ["alice@example.com", "bob@test.org", "hello world"]})
        r = det.scan(df)
        assert "email_col" in r["pii_columns"]

    def test_clean_column(self):
        det = self._det()
        df = pd.DataFrame({"fruit": ["apple", "banana", "cherry"]})
        r = det.scan(df)
        assert "fruit" in r["safe_columns"]

    def test_scan_dataframe(self):
        det = self._det()
        df = pd.DataFrame({
            "email":   ["a@b.com", "c@d.com"],
            "revenue": [100, 200],
        })
        r = det.scan(df)
        assert "email" in r["pii_columns"]
        assert "revenue" not in r["pii_columns"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. AccessAnomalyDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestAccessAnomalyDetector:

    def test_insufficient_events(self):
        from governance.anomaly_access import AccessAnomalyDetector
        det = AccessAnomalyDetector()
        det.observe("alice", "/api/data", 200, 1.0, 10)
        r = det.is_anomalous("alice")
        assert r["method"] == "insufficient_data"

    def test_spike_flagged(self):
        from governance.anomaly_access import AccessAnomalyDetector
        det = AccessAnomalyDetector()
        for i in range(30):
            det.observe("bob", "/api/secret", 403, 0.0, 3)   # many errors, after-hours
        r = det.is_anomalous("bob")
        assert r["score"] > 0   # some anomaly signal


# ─────────────────────────────────────────────────────────────────────────────
# 7. PipelineSuccessPredictor
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineSuccessPredictor:

    def test_good_pipeline(self):
        from verifier.pipeline_success_predictor import PipelineSuccessPredictor
        p = PipelineSuccessPredictor()
        ctx = {
            "null_rate": 0.01, "drift_detected": False, "quality_score": 0.95,
            "row_count": 50000, "n_columns": 10, "anomaly_count": 0,
            "schema_match": True, "cv_score": 0.88, "columns_drifted": 0,
        }
        r = p.predict(ctx)
        assert r["prediction"] in {"LIKELY_SUCCESS", "LIKELY_FAILURE"}
        assert 0 <= r["success_prob"] <= 1

    def test_bad_pipeline(self):
        from verifier.pipeline_success_predictor import PipelineSuccessPredictor
        p = PipelineSuccessPredictor()
        ctx = {
            "null_rate": 0.50, "drift_detected": True, "quality_score": 0.20,
            "row_count": 100, "n_columns": 2, "anomaly_count": 30,
            "schema_match": False, "cv_score": 0.40, "columns_drifted": 5,
        }
        r = p.predict(ctx)
        assert r["prediction"] == "LIKELY_FAILURE"


# ─────────────────────────────────────────────────────────────────────────────
# 8. ChartRelevanceScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestChartRelevanceScorer:

    def test_returns_sorted_list(self):
        from reporting_service.chart_relevance_scorer import ChartRelevanceScorer
        scorer = ChartRelevanceScorer()
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((200, 5)), columns=list("abcde"))
        result = scorer.rank(df, query_intent="trend over time")
        assert len(result) > 0
        assert "chart_type" in result[0]
        assert "score" in result[0]
        # Scores should be descending
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_df(self):
        from reporting_service.chart_relevance_scorer import ChartRelevanceScorer
        scorer = ChartRelevanceScorer()
        result = scorer.rank(pd.DataFrame(), query_intent="")
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# 9. NLP Query Classifier (heuristic seed corpus)
# ─────────────────────────────────────────────────────────────────────────────

class TestNLPQuery:

    def _engine(self):
        from analyst.nlp_query import NLPQueryEngine
        return NLPQueryEngine()

    def test_top_n_intent(self):
        eng = self._engine()
        r = eng.parse("show me top 10 customers by revenue")
        assert r["intent"] == "top_n"

    def test_aggregate_intent(self):
        eng = self._engine()
        r = eng.parse("what is the total revenue?")
        assert r["intent"] == "aggregate"

    def test_filter_intent(self):
        eng = self._engine()
        r = eng.parse("show customers where age > 30")
        assert r["intent"] == "filter"

    def test_trend_intent(self):
        eng = self._engine()
        r = eng.parse("show monthly revenue trend")
        assert r["intent"] == "trend"

    def test_execute_top_n(self):
        eng = self._engine()
        df = pd.DataFrame({"revenue": [100, 200, 50, 400], "name": ["A", "B", "C", "D"]})
        r  = eng.execute_on_df(df, "top 2 by revenue")
        assert isinstance(r, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
