"""
tests/test_insight_narrator.py
---------------------------------
Tests for the InsightNarrator engine.
Verifies that each sub-narrator returns well-formed, non-empty, meaningful text.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reporting_service.insight_narrator import (
    ColumnInterpreter,
    CorrelationNarrator,
    MissingValueAdvisor,
    OutlierExplainer,
    AnomalyExplainer,
    DataQualityReasoner,
    ModelMetricInterpreter,
    NarrativeBuilder,
    get_narrator,
)


# ── ColumnInterpreter ─────────────────────────────────────────────────────────

class TestColumnInterpreter:
    def setup_method(self):
        self.interp = ColumnInterpreter()

    def test_normal_distribution(self):
        result = self.interp.interpret("salary", {"mean": 50000, "std": 10000, "skewness": 0.1, "kurtosis": 0.2})
        assert result["shape"] == "Normal"
        assert "salary" in result["meaning"]
        assert len(result["meaning"]) > 30
        assert "recommendation" in result

    def test_right_skewed(self):
        result = self.interp.interpret("revenue", {"mean": 1200, "std": 5000, "skewness": 2.4, "kurtosis": 0.5})
        assert "right" in result["shape"].lower()
        assert "skew" in result["meaning"].lower() or "tail" in result["meaning"].lower()

    def test_left_skewed(self):
        result = self.interp.interpret("satisfaction", {"mean": 4.5, "std": 0.5, "skewness": -1.5, "kurtosis": 0.3})
        assert "left" in result["shape"].lower()

    def test_high_null_flag(self):
        result = self.interp.interpret("income", {"mean": 60000, "std": 20000, "skewness": 0.5}, null_pct=0.35)
        assert "flag" in result
        assert "35%" in result["flag"] or "0.35" in result["flag"] or "missing" in result["flag"].lower()

    def test_high_zero_flag(self):
        result = self.interp.interpret("purchases", {"mean": 0.3, "std": 1.2, "skewness": 1.8}, zero_pct=0.45)
        assert "flag" in result

    def test_returns_required_keys(self):
        result = self.interp.interpret("col_a", {"mean": 0, "std": 1, "skewness": 0})
        assert "shape" in result
        assert "meaning" in result
        assert "recommendation" in result


# ── CorrelationNarrator ───────────────────────────────────────────────────────

class TestCorrelationNarrator:
    def setup_method(self):
        self.narr = CorrelationNarrator()

    def test_strong_positive(self):
        text = self.narr.narrate("revenue", "loan_amount", 0.82)
        assert "strongly" in text.lower() or "strong" in text.lower()
        assert "multicollinear" in text.lower() or "collinear" in text.lower()

    def test_strong_negative(self):
        text = self.narr.narrate("default_rate", "credit_score", -0.78)
        assert "negative" in text.lower() or "decreases" in text.lower()

    def test_moderate(self):
        text = self.narr.narrate("age", "income", 0.55)
        assert "moderate" in text.lower()

    def test_weak(self):
        text = self.narr.narrate("shoe_size", "iq_score", 0.08)
        assert "weak" in text.lower()

    def test_multicollinearity_warning_fires(self):
        pairs = [{"a": "a", "b": "b", "r": 0.85}, {"a": "c", "b": "d", "r": 0.3}]
        warn = self.narr.multicollinearity_warning(pairs)
        assert warn is not None
        assert "multicollinear" in warn.lower() or "correlated" in warn.lower()

    def test_multicollinearity_no_warning_when_none_strong(self):
        pairs = [{"a": "a", "b": "b", "r": 0.4}]
        warn = self.narr.multicollinearity_warning(pairs)
        assert warn is None


# ── MissingValueAdvisor ───────────────────────────────────────────────────────

class TestMissingValueAdvisor:
    def setup_method(self):
        self.adv = MissingValueAdvisor()

    def test_very_low_null(self):
        text = self.adv.advise("col", 0.005)
        assert "mcar" in text.lower() or "random" in text.lower()

    def test_moderate_null_mar_guidance(self):
        text = self.adv.advise("credit_score", 0.10)
        assert "mar" in text.lower() or "at random" in text.lower() or "investigate" in text.lower()

    def test_high_null_mnar_warning(self):
        text = self.adv.advise("income", 0.35)
        assert "mnar" in text.lower() or "not at random" in text.lower() or "indicator" in text.lower()

    def test_very_high_null_drop_suggestion(self):
        text = self.adv.advise("old_column", 0.75)
        assert "drop" in text.lower() or "limited" in text.lower()


# ── OutlierExplainer ──────────────────────────────────────────────────────────

class TestOutlierExplainer:
    def setup_method(self):
        self.exp = OutlierExplainer()

    def test_low_outlier_rate(self):
        text = self.exp.explain("revenue", 5, 0.01, "IQR")
        assert "few" in text.lower() or "only" in text.lower() or "very" in text.lower()

    def test_moderate_outlier_rate(self):
        text = self.exp.explain("spend", 400, 0.04, "Z-Score")
        assert "notable" in text.lower() or "moderate" in text.lower() or "proportion" in text.lower()

    def test_high_outlier_rate(self):
        text = self.exp.explain("count", 1000, 0.12, "IQR")
        assert "high" in text.lower() or "structural" in text.lower()
        assert "not" in text.lower()  # "Do NOT simply drop"


# ── AnomalyExplainer ──────────────────────────────────────────────────────────

class TestAnomalyExplainer:
    def setup_method(self):
        self.exp = AnomalyExplainer()

    def test_very_low_anomaly(self):
        text = self.exp.explain(0.01, 10000)
        assert "very low" in text.lower() or "only" in text.lower()

    def test_moderate_anomaly(self):
        text = self.exp.explain(0.04, 5000)
        assert "moderate" in text.lower() or "review" in text.lower()

    def test_high_anomaly(self):
        text = self.exp.explain(0.15, 10000)
        assert "high" in text.lower() or "priority" in text.lower() or "systematic" in text.lower()


# ── DataQualityReasoner ───────────────────────────────────────────────────────

class TestDataQualityReasoner:
    def setup_method(self):
        self.reasoner = DataQualityReasoner()

    def test_known_flag(self):
        text = self.reasoner.explain_flag("HIGH_NULL")
        assert len(text) > 20
        assert "missing" in text.lower() or "null" in text.lower()

    def test_drift_flag(self):
        text = self.reasoner.explain_flag("DRIFT")
        assert "drift" in text.lower() or "distribution" in text.lower()

    def test_unknown_flag_fallback(self):
        text = self.reasoner.explain_flag("UNKNOWN_FLAG_XYZ")
        assert len(text) > 10  # Should have a fallback

    def test_remediation_critical(self):
        text = self.reasoner.remediation("HIGH_NULL", "CRITICAL")
        assert "🔴" in text or "immediate" in text.lower()

    def test_remediation_warning(self):
        text = self.reasoner.remediation("DRIFT", "WARNING")
        assert "🟡" in text or "recommend" in text.lower()


# ── ModelMetricInterpreter ────────────────────────────────────────────────────

class TestModelMetricInterpreter:
    def setup_method(self):
        self.interp = ModelMetricInterpreter()

    def test_excellent_roc_auc(self):
        text = self.interp.interpret_roc_auc(0.95)
        assert "exceptional" in text.lower() or "excellent" in text.lower()
        assert "0.95" in text

    def test_good_roc_auc(self):
        text = self.interp.interpret_roc_auc(0.85)
        assert "good" in text.lower()

    def test_poor_roc_auc(self):
        text = self.interp.interpret_roc_auc(0.60)
        assert "poor" in text.lower() or "limited" in text.lower()

    def test_high_f1(self):
        text = self.interp.interpret_f1(0.92)
        assert "excellent" in text.lower()

    def test_low_f1_imbalance_guidance(self):
        text = self.interp.interpret_f1(0.45)
        assert "imbalance" in text.lower() or "smote" in text.lower()

    def test_interpret_all_dict(self):
        results = self.interp.interpret_all({"roc_auc": 0.88, "f1": 0.75, "accuracy": 0.91})
        assert len(results) == 3
        assert all(isinstance(r, str) and len(r) > 5 for r in results)


# ── NarrativeBuilder ──────────────────────────────────────────────────────────

class TestNarrativeBuilder:
    def setup_method(self):
        self.builder = NarrativeBuilder()

    def _sample_eda(self):
        return {
            "summary": {"n_rows": 10000, "n_cols": 12, "overall_null_pct": 0.04, "anomaly_pct": 0.03},
            "numeric_stats": {
                "revenue":  {"mean": 1200, "std": 3000, "skewness": 2.1, "kurtosis": 0.5},
                "loan_amt": {"mean": 50000, "std": 20000, "skewness": 0.3, "kurtosis": -0.1},
            },
            "correlations": [{"col_a": "revenue", "col_b": "loan_amt", "correlation": 0.75}],
            "insights":     ["Revenue is right-skewed", "loan_amt is normally distributed"],
        }

    def test_build_returns_string(self):
        result = self.builder.build(eda_report=self._sample_eda(), confidence_score=0.88,
                                    row_count=10000, col_count=12)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_build_contains_sections(self):
        result = self.builder.build(eda_report=self._sample_eda(), confidence_score=0.88,
                                    row_count=10000, col_count=12)
        assert "Dataset Overview" in result
        assert "Recommendation" in result

    def test_build_approved_recommendation(self):
        result = self.builder.build(eda_report=self._sample_eda(), confidence_score=0.90,
                                    row_count=10000, col_count=12, gate1_decision="PASS",
                                    gate2_decision="PASS")
        assert "approved" in result.lower() or "✅" in result

    def test_build_rejected_recommendation(self):
        result = self.builder.build(confidence_score=0.30, row_count=1000, col_count=5,
                                    gate1_decision="FAIL", gate2_decision="FAIL")
        assert "not approved" in result.lower() or "🔴" in result

    def test_build_hf_prompt_structure(self):
        vr = {"gate_decision": "PASS", "confidence_score": 0.85, "row_count": 10000, "col_count": 12}
        prompt = self.builder.build_hf_prompt(verified_result=vr, eda_report=self._sample_eda())
        assert "Senior Data Scientist" in prompt
        assert "Executive Report" in prompt
        assert "RULES" in prompt
        # Should include pre-computed insights
        assert "n_rows" not in prompt.lower()  # Should be formatted, not raw keys

    def test_get_narrator_returns_builder(self):
        nb = get_narrator()
        assert isinstance(nb, NarrativeBuilder)


# ── LLMProvider integration ───────────────────────────────────────────────────

class TestLLMProviderNarrative:
    """Verify the rule-based fallback now returns a rich narrative."""

    def test_rule_based_narrative_structure(self):
        from reporting_service.llm_provider import LLMProvider
        provider = LLMProvider()
        result = provider.generate_summary({
            "gate_decision":    "PASS",
            "confidence_score": 0.87,
            "row_count":        10000,
            "col_count":        12,
            "eda_report": {
                "summary":   {"n_rows": 10000, "n_cols": 12, "overall_null_pct": 0.04, "anomaly_pct": 0.02},
                "numeric_stats": {"revenue": {"mean": 1200, "std": 3000, "skewness": 1.8}},
                "insights":  ["Revenue is right-skewed"],
                "correlations": [],
            },
        })
        assert isinstance(result, str)
        assert len(result.split()) > 30
        # Should no longer be the 4-line minimal fallback
        assert "Dataset Overview" in result or "Recommendation" in result

    def test_governance_block_still_works(self):
        from reporting_service.llm_provider import LLMProvider
        provider = LLMProvider()
        result = provider.generate_summary({"gate_decision": "FAIL", "confidence_score": 0.3})
        assert "[GOVERNANCE BLOCK]" in result
