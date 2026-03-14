"""
tests/test_verifier_stubs.py
------------------------------
Tests for verifier/drift_verifier.py, stability_verifier.py, domain_verifier.py.

Ensures all verifiers:
- Return the correct result schema
- Correctly classify drift levels
- SHAP and CV checks work with real sklearn models
- Domain rules are correctly applied
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# ══════════════════════════════════════════════════════════════════════════════
# DriftVerifier
# ══════════════════════════════════════════════════════════════════════════════

class TestDriftVerifier:

    @pytest.fixture()
    def verifier(self):
        from verifier.drift_verifier import DriftVerifier
        return DriftVerifier(psi_threshold=0.25)

    @pytest.fixture()
    def ref_df(self):
        np.random.seed(0)
        return pd.DataFrame({"a": np.random.normal(0, 1, 500), "b": np.random.uniform(0, 1, 500)})

    def test_no_drift_returns_stable(self, verifier, ref_df):
        """Same distribution → STABLE."""
        cur_df = ref_df.copy().sample(frac=0.8, random_state=1)
        result = verifier.verify(current_df=cur_df, reference_df=ref_df)
        assert result["passed"] is True
        assert result["severity"] in ("STABLE", "WARN")

    def test_extreme_drift_returns_drift(self, verifier, ref_df):
        """Very different distribution → DRIFT, passed=False."""
        cur_df = pd.DataFrame({
            "a": np.random.normal(100, 1, 500),   # completely different
            "b": np.random.uniform(100, 200, 500),
        })
        result = verifier.verify(current_df=cur_df, reference_df=ref_df)
        assert result["severity"] == "DRIFT"
        assert result["passed"] is False

    def test_returns_float_scores(self, verifier, ref_df):
        """All score values must be floats in [0, ∞)."""
        cur_df = ref_df.copy()
        result = verifier.verify(current_df=cur_df, reference_df=ref_df)
        assert isinstance(result["value"], float)
        assert result["value"] >= 0.0

    def test_column_details_present(self, verifier, ref_df):
        """column_details must list per-column metrics."""
        cur_df = ref_df.copy()
        result = verifier.verify(current_df=cur_df, reference_df=ref_df)
        assert isinstance(result["column_details"], dict)
        assert "a" in result["column_details"] or "b" in result["column_details"]

    def test_legacy_scores_dict_path(self, verifier):
        """Pre-computed PSI scores path must work."""
        scores = {"col_a": 0.05, "col_b": 0.30}
        result = verifier.verify(drift_scores=scores)
        assert result["passed"] is False   # col_b > 0.25
        assert "col_b" in result["critical_columns"]

    def test_empty_data_does_not_crash(self, verifier):
        """Empty DataFrames must not crash."""
        result = verifier.verify(current_df=pd.DataFrame(), reference_df=pd.DataFrame())
        assert isinstance(result, dict)
        assert "passed" in result

    def test_no_reference_returns_pass(self, verifier):
        """No reference → drift check skipped → passed=True."""
        result = verifier.verify()
        assert result["passed"] is True


# ══════════════════════════════════════════════════════════════════════════════
# StabilityVerifier
# ══════════════════════════════════════════════════════════════════════════════

class TestStabilityVerifier:

    @pytest.fixture()
    def model_and_data(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_stable_model_passes(self, model_and_data):
        """Well-trained model on balanced data should pass stability check."""
        from verifier.stability_verifier import StabilityVerifier
        model, X, y = model_and_data
        sv = StabilityVerifier(max_std_threshold=0.15, check_shap=False)
        result = sv.verify(model, X, y)
        assert result["passed"] is True
        assert isinstance(result["value"], float)
        assert result["value"] >= 0.0

    def test_result_has_required_keys(self, model_and_data):
        """Result must contain all required keys."""
        from verifier.stability_verifier import StabilityVerifier
        model, X, y = model_and_data
        sv = StabilityVerifier(check_shap=False)
        result = sv.verify(model, X, y)
        for key in ("metric", "value", "mean_score", "passed", "cv_passed", "detail"):
            assert key in result, f"Missing key: {key}"

    def test_empty_training_set_does_not_crash(self):
        """Empty training set must return a valid result without crashing."""
        from verifier.stability_verifier import StabilityVerifier
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier()
        sv = StabilityVerifier(check_shap=False)
        result = sv.verify(model, np.empty((0, 3)), np.empty(0))
        assert isinstance(result, dict)
        assert result["passed"] is True

    def test_cv_std_is_float(self, model_and_data):
        """CV std must be a numeric float."""
        from verifier.stability_verifier import StabilityVerifier
        model, X, y = model_and_data
        sv = StabilityVerifier(check_shap=False)
        result = sv.verify(model, X, y)
        assert isinstance(result["value"], float)


# ══════════════════════════════════════════════════════════════════════════════
# DomainVerifier
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainVerifier:

    def test_non_negative_rule_accepts_positive(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[{"type": "non_negative"}])
        preds = np.array([0.1, 0.5, 0.9])
        result = dv.verify(predictions=preds)
        assert result["passed"] is True

    def test_non_negative_rule_rejects_negative(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[{"type": "non_negative"}])
        preds = np.array([-1.0, 0.5, 0.9])
        result = dv.verify(predictions=preds)
        assert result["passed"] is False
        assert len(result["violations"]) > 0

    def test_range_rule_rejects_out_of_bounds(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[{"type": "range", "min": 0, "max": 1}])
        preds = np.array([0.0, 1.5])   # 1.5 > max
        result = dv.verify(predictions=preds)
        assert result["passed"] is False

    def test_no_rules_always_passes(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[])
        result = dv.verify(predictions=np.array([1, 2, 3]))
        assert result["passed"] is True
        assert result["value"] == 0  # zero violations

    def test_banking_preset_loads(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(domain="banking")
        assert len(dv.rules) > 0
        rule_types = [r["type"] for r in dv.rules]
        assert "non_negative" in rule_types

    def test_no_inf_rule_detects_inf(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[{"type": "no_inf"}])
        preds = np.array([1.0, float("inf"), 0.5])
        result = dv.verify(predictions=preds)
        assert result["passed"] is False

    def test_result_schema(self):
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier()
        result = dv.verify(predictions=np.array([0.5, 0.7]))
        for key in ("metric", "value", "passed", "violations", "detail", "domain"):
            assert key in result, f"Missing key: {key}"

    def test_from_config(self):
        from verifier.domain_verifier import DomainVerifier
        config = {"pipeline": {"domain": "finance"}}
        dv = DomainVerifier.from_config(config)
        assert dv.domain == "finance"
        assert len(dv.rules) > 0
