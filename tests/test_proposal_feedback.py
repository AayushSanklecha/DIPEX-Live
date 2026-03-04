"""
tests/test_proposal_feedback.py
---------------------------------
Tests for:
  - ProposalRouter (Phase 5: Proposal Layer)
  - FeedbackController (Phase 14: Feedback & Retry Controller)
  - FinanceRules (Phase 3: Hard Gate 1 — finance domain)
"""
from __future__ import annotations

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# PROPOSAL ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class TestProposalRouter:

    @pytest.fixture
    def router(self):
        from proposal.proposal_router import ProposalRouter
        return ProposalRouter(domain="default", max_proposals=10)

    def test_returns_list(self, router):
        proposals = router.route()
        assert isinstance(proposals, list)

    def test_returns_required_keys(self, router):
        proposals = router.route(row_count=100)
        assert len(proposals) > 0
        for p in proposals:
            for key in ("operation", "tier", "rationale", "confidence", "priority", "estimated_ms"):
                assert key in p, f"Missing key '{key}' in proposal: {p}"

    def test_confidence_in_range(self, router):
        proposals = router.route(row_count=500)
        for p in proposals:
            assert 0.0 <= p["confidence"] <= 1.0

    def test_high_null_rate_triggers_data_cleaning(self, router):
        profile = {
            "columns": {
                "col_a": {"null_rate": 0.40, "skewness": 0.0, "cardinality_tier": "medium"},
                "col_b": {"null_rate": 0.60, "skewness": 0.0, "cardinality_tier": "medium"},
            },
            "analyst_flags": [],
        }
        proposals = router.route(profile=profile, row_count=200)
        ops = [p["operation"] for p in proposals]
        assert "data_cleaning" in ops

    def test_high_skewness_triggers_stats(self, router):
        profile = {
            "columns": {
                "income": {"null_rate": 0.0, "skewness": 3.5, "cardinality_tier": "high"},
            },
            "analyst_flags": [],
        }
        proposals = router.route(profile=profile, row_count=200)
        ops = [p["operation"] for p in proposals]
        assert "basic_stats" in ops

    def test_drift_triggers_experiment_design(self, router):
        drift_report = {"overall_psi": 0.35, "severity": "severe"}
        proposals = router.route(drift_report=drift_report, row_count=500)
        ops = [p["operation"] for p in proposals]
        assert "design_experiment" in ops

    def test_low_confidence_triggers_causal_inference(self, router):
        proposals = router.route(confidence_score=0.55, row_count=200)
        ops = [p["operation"] for p in proposals]
        assert "causal_inference" in ops

    def test_banking_domain_triggers_aml(self):
        from proposal.proposal_router import ProposalRouter
        router = ProposalRouter(domain="banking")
        proposals = router.route(row_count=500)
        ops = [p["operation"] for p in proposals]
        assert "aml_pattern_detection" in ops

    def test_finance_domain_triggers_time_series(self):
        from proposal.proposal_router import ProposalRouter
        router = ProposalRouter(domain="finance")
        proposals = router.route(row_count=500)
        ops = [p["operation"] for p in proposals]
        assert "time_series_analysis" in ops

    def test_max_proposals_cap(self, router):
        profile = {
            "columns": {
                f"col_{i}": {"null_rate": 0.2, "skewness": 3.0, "cardinality_tier": "unique"}
                for i in range(20)
            },
            "analyst_flags": [{"flag_type": "high_correlation"}] * 5,
        }
        proposals = router.route(profile=profile, row_count=1000, confidence_score=0.50)
        assert len(proposals) <= router.max_proposals

    def test_deduplication_no_duplicate_ops(self, router):
        proposals = router.route(row_count=1000, confidence_score=0.50)
        op_names = [p["operation"] for p in proposals]
        assert len(op_names) == len(set(op_names)), "Duplicate operations in proposals"

    def test_priority_sorted(self, router):
        proposals = router.route(row_count=200)
        priorities = [p["priority"] for p in proposals]
        assert priorities == sorted(priorities), "Proposals not sorted by priority"

    def test_filter_by_tier(self, router):
        proposals = router.route(row_count=500)
        junior = router.filter_by_tier(proposals, "junior")
        assert all(p["tier"] == "junior" for p in junior)

    def test_top_operation_returns_string(self, router):
        proposals = router.route(row_count=200)
        top = router.top_operation(proposals)
        assert top is None or isinstance(top, str)

    def test_empty_row_count_still_returns_proposals(self, router):
        proposals = router.route(row_count=0)
        # Should still have at least the report proposal
        assert len(proposals) >= 1

    def test_gate_reject_triggers_data_cleaning(self, router):
        class MockGateResult:
            decision = "REJECT"
            failures = [{"column": "revenue", "message": "negative values"}]
            warnings = []
        proposals = router.route(gate_result=MockGateResult(), row_count=100)
        ops = [p["operation"] for p in proposals]
        assert "data_cleaning" in ops


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackController:

    @pytest.fixture
    def controller(self, tmp_path):
        from ingestion.feedback_controller import FeedbackController
        return FeedbackController(
            max_retries=3,
            confidence_threshold=0.75,
            audit_dir=str(tmp_path),
        )

    @pytest.fixture
    def clean_df(self):
        return pd.DataFrame({
            "revenue":  [100.0, 200.0, 300.0] * 30,
            "category": ["A", "B", "C"] * 30,
        })

    @pytest.fixture
    def dirty_df(self):
        data = pd.DataFrame({
            "revenue":  [100.0, None, 300.0, None, None] * 20,
            "category": ["A", None, "C", None, "B"] * 20,
        })
        return data

    def test_returns_retry_result(self, controller, clean_df):
        from ingestion.feedback_controller import RetryResult
        result = controller.evaluate(clean_df, run_id="test-001", confidence_score=0.80)
        assert isinstance(result, RetryResult)

    def test_above_threshold_returns_pass_no_retry(self, controller, clean_df):
        result = controller.evaluate(clean_df, run_id="test-001", confidence_score=0.85)
        assert result.decision == "PASS"
        assert result.strategy == "NONE_NEEDED"

    def test_budget_exhausted_escalates(self, controller, dirty_df, tmp_path):
        result = controller.evaluate(
            dirty_df, run_id="escalate-001",
            confidence_score=0.10,
            attempt=4,   # exceeds max_retries=3
        )
        assert result.decision == "ESCALATED"

    def test_escalation_writes_to_jsonl(self, controller, dirty_df, tmp_path):
        controller.evaluate(
            dirty_df, run_id="esc-log-001",
            confidence_score=0.10,
            attempt=4,
        )
        log_file = tmp_path / "retry_escalations.jsonl"
        assert log_file.exists()
        line = json.loads(log_file.read_text().strip().splitlines()[0])
        assert line["run_id"] == "esc-log-001"
        assert "confidence" in line

    def test_strategy_selected_not_none(self, controller, dirty_df):
        result = controller.evaluate(dirty_df, run_id="strat-001", confidence_score=0.50)
        assert result.strategy in ["AGGRESSIVE_CLEAN", "IMPUTE_MEDIAN", "OUTLIER_CLIP",
                                   "DROP_LOW_QUALITY", "SCHEMA_RELAX", "FALLBACK_COPY"]

    def test_aggressive_clean_drops_nulls(self, controller, dirty_df):
        result = controller.evaluate(dirty_df, run_id="agg-001", confidence_score=0.40)
        if result.df is not None and result.strategy == "AGGRESSIVE_CLEAN":
            assert result.df.isnull().sum().sum() == 0

    def test_impute_median_fills_nulls(self, controller):
        from ingestion.feedback_controller import FeedbackController
        ctrl = FeedbackController(max_retries=3, audit_dir="/tmp")
        df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0] * 20})
        cleaned = ctrl._strategy_impute_median(df.copy())
        assert cleaned["x"].isnull().sum() == 0

    def test_outlier_clip_reduces_extremes(self, controller):
        from ingestion.feedback_controller import FeedbackController
        ctrl = FeedbackController(max_retries=3, audit_dir="/tmp")
        # 98 normal values + 2 extreme outliers: 99th pct = 1.0, not 1e6
        df = pd.DataFrame({"x": [1.0] * 98 + [1_000_000.0, -1_000_000.0]})
        clipped = ctrl._strategy_outlier_clip(df.copy())
        assert clipped["x"].max() < 1_000_000.0
        assert clipped["x"].min() > -1_000_000.0

    def test_drop_low_quality_removes_high_null_columns(self, controller):
        from ingestion.feedback_controller import FeedbackController
        ctrl = FeedbackController(max_retries=3, audit_dir="/tmp")
        df = pd.DataFrame({
            "good":  [1.0] * 100,
            "bad":   [None] * 80 + [1.0] * 20,  # 80% null
        })
        cleaned = ctrl._strategy_drop_low_quality(df.copy())
        assert "bad" not in cleaned.columns

    def test_retry_loop_runs_to_completion(self, controller, dirty_df):
        result = controller.run_retry_loop(
            dirty_df, run_id="loop-001", initial_confidence=0.40
        )
        assert result.decision in ("PASS", "REJECT", "ESCALATED")

    def test_reward_computed_correctly(self, controller):
        reward = controller._compute_reward(0.60, 0.75)
        assert abs(reward - 0.15) < 0.001

    def test_bandit_summary_has_required_keys(self, controller, dirty_df):
        controller.evaluate(dirty_df, run_id="bs-001", confidence_score=0.50)
        summary = controller.bandit_summary()
        assert "strategy_counts" in summary
        assert "strategy_rewards" in summary
        assert "best_strategy"   in summary

    def test_result_to_dict(self, controller, clean_df):
        result = controller.evaluate(clean_df, run_id="dict-001", confidence_score=0.80)
        d = result.to_dict()
        assert "run_id" in d
        assert "decision" in d
        assert "strategy" in d
        assert "timestamp" in d

    def test_retry_never_modifies_original_df(self, controller, dirty_df):
        """Safety invariant: original DataFrame is never mutated."""
        original_len = len(dirty_df)
        original_nulls = dirty_df.isnull().sum().sum()
        controller.run_retry_loop(dirty_df, run_id="immut-001", initial_confidence=0.40)
        assert len(dirty_df) == original_len
        assert dirty_df.isnull().sum().sum() == original_nulls


# ══════════════════════════════════════════════════════════════════════════════
# FINANCE REGULATORY RULES
# ══════════════════════════════════════════════════════════════════════════════

class TestFinanceRules:

    def test_revenue_recognition_negative_flags(self):
        from validation.regulatory.finance_rules import RevenueRecognitionRule
        rule = RevenueRecognitionRule(revenue_columns=["revenue"])
        df = pd.DataFrame({"revenue": [-100.0] * 30 + [200.0] * 70})
        violations = rule.evaluate(df)
        assert len(violations) > 0
        assert violations[0].severity == "ERROR"
        assert violations[0].offending_count == 30

    def test_revenue_recognition_credit_memo_exempt(self):
        from validation.regulatory.finance_rules import RevenueRecognitionRule
        rule = RevenueRecognitionRule(
            revenue_columns=["revenue"],
            credit_memo_column="is_credit",
        )
        df = pd.DataFrame({
            "revenue":   [-100.0] * 50 + [200.0] * 50,
            "is_credit": [True] * 50 + [False] * 50,
        })
        violations = rule.evaluate(df)
        # All negatives are credit memos — should be no violations
        assert len(violations) == 0

    def test_revenue_recognition_clean_passes(self):
        from validation.regulatory.finance_rules import RevenueRecognitionRule
        rule = RevenueRecognitionRule(revenue_columns=["revenue"])
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0] * 33})
        assert rule.evaluate(df) == []

    def test_capital_adequacy_breach(self):
        from validation.regulatory.finance_rules import CapitalAdequacyRule
        rule = CapitalAdequacyRule(tier1_col="t1", rwa_col="rwa", min_car=0.08)
        # 5% CAR < 8% minimum
        df = pd.DataFrame({"t1": [50.0] * 20, "rwa": [1000.0] * 20})
        violations = rule.evaluate(df)
        assert len(violations) > 0
        assert violations[0].severity == "CRITICAL"
        assert violations[0].offending_count == 20

    def test_capital_adequacy_passes(self):
        from validation.regulatory.finance_rules import CapitalAdequacyRule
        rule = CapitalAdequacyRule(tier1_col="t1", rwa_col="rwa", min_car=0.08)
        # 10% CAR > 8% minimum
        df = pd.DataFrame({"t1": [100.0] * 20, "rwa": [1000.0] * 20})
        assert rule.evaluate(df) == []

    def test_net_position_limit_long_breach(self):
        from validation.regulatory.finance_rules import NetPositionLimitRule
        rule = NetPositionLimitRule(
            position_column="pos", max_long=1000, max_short=500
        )
        df = pd.DataFrame({"pos": [1500.0] * 10 + [0.0] * 90})
        violations = rule.evaluate(df)
        assert any("long" in v.message.lower() for v in violations)

    def test_double_entry_balance_catches_imbalance(self):
        from validation.regulatory.finance_rules import DoubleEntryBalanceRule
        rule = DoubleEntryBalanceRule(
            amount_col="amount",
            transaction_id_col="tx_id",
            tolerance=0.01,
        )
        df = pd.DataFrame({
            "tx_id":  ["TX001", "TX001"],
            "amount": [100.0, -90.0],   # net 10.0 — imbalanced
        })
        violations = rule.evaluate(df)
        assert len(violations) > 0
        assert violations[0].offending_count == 1

    def test_double_entry_balance_passes_balanced(self):
        from validation.regulatory.finance_rules import DoubleEntryBalanceRule
        rule = DoubleEntryBalanceRule(
            amount_col="amount",
            transaction_id_col="tx_id",
        )
        df = pd.DataFrame({
            "tx_id":  ["TX001", "TX001"],
            "amount": [100.0, -100.0],  # net 0.0 — balanced
        })
        assert rule.evaluate(df) == []

    def test_fair_value_hierarchy_breach(self):
        from validation.regulatory.finance_rules import FairValueHierarchyRule
        rule = FairValueHierarchyRule(
            level3_col="l3",
            total_fair_value_col="total",
            max_level3_ratio=0.20,
        )
        df = pd.DataFrame({
            "l3":    [300.0] * 20,
            "total": [1000.0] * 20,  # 30% > 20% threshold
        })
        violations = rule.evaluate(df)
        assert len(violations) > 0
        assert violations[0].offending_count == 20

    def test_margin_call_threshold_flags(self):
        from validation.regulatory.finance_rules import MarginCallThresholdRule
        rule = MarginCallThresholdRule(
            margin_balance_col="balance",
            maintenance_margin_col="required",
        )
        df = pd.DataFrame({
            "balance":  [5000.0] * 10 + [25000.0] * 10,
            "required": [10000.0] * 20,  # 10 accounts below maintenance
        })
        violations = rule.evaluate(df)
        assert len(violations) > 0
        assert violations[0].offending_count == 10

    def test_sec_filing_bounds_lower(self):
        from validation.regulatory.finance_rules import SECFilingBoundsRule
        rule = SECFilingBoundsRule(column="assets", min_value=0.0)
        df = pd.DataFrame({"assets": [-500.0] * 5 + [1000.0] * 95})
        violations = rule.evaluate(df)
        assert len(violations) > 0

    def test_finance_rules_missing_columns_graceful(self):
        """All rules must be graceful when columns are missing."""
        from validation.regulatory.finance_rules import (
            RevenueRecognitionRule, CapitalAdequacyRule, DoubleEntryBalanceRule
        )
        empty_df = pd.DataFrame({"unrelated": [1, 2, 3]})
        assert RevenueRecognitionRule(["revenue"]).evaluate(empty_df) == []
        assert CapitalAdequacyRule("t1", "rwa").evaluate(empty_df) == []
        assert DoubleEntryBalanceRule("amount", "tx_id").evaluate(empty_df) == []
