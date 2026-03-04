"""
tests/test_hard_gate.py
-----------------------
Tests for Step 2 — Hard Gate 1: Deterministic Validation sub-system.
"""

import re
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from validation.null_validator import NullValidator
from validation.range_validator import (
    BoundRule,
    LogicalRule,
    PositivityRule,
    RangeValidator,
    RuleSeverity,
)
from validation.schema_validator import SchemaValidator
from validation.integrity_checker import IntegrityChecker
from validation.hard_gate import HardGate, GateResult
from validation.regulatory.banking_rules import (
    AMLThresholdRule,
    LoanRatioRule,
    PositiveAmountRule,
    RepaymentConsistencyRule,
)
from validation.regulatory.healthcare_rules import (
    AgeRangeRule,
    DiagnosisCodeFormatRule,
    PHIPresenceRule,
    VitalSignsRule,
)
from validation.regulatory.regulatory_engine import RegulatoryEngine


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

MINIMAL_CONFIG: dict = {
    "pipeline": {"qa_gate": {"null_threshold": 0.10}},
    "validation": {
        "null": {"global_threshold": 0.10, "warn_threshold": 0.05, "critical_fields": []},
        "range": {"bounds": [], "positivity": [], "logical": []},
        "integrity": {"check_duplicates": True, "id_columns": [], "referential": [], "cross_column_rules": []},
        "regulatory": {"domain": "generic"},
        "schema": {},
    },
}


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": list(range(1, 21)),
        "amount": rng.uniform(100, 5000, 20).round(2).tolist(),
        "age": rng.integers(18, 80, 20).tolist(),
        "status": ["ACTIVE"] * 15 + ["CLOSED"] * 5,
    })


# ─────────────────────────────────────────────────────────────
# NullValidator
# ─────────────────────────────────────────────────────────────

class TestNullValidator:
    def test_critical_field_with_nulls_returns_critical(self):
        df = pd.DataFrame({"id": [1, None, 3], "x": [1, 2, 3]})
        v = NullValidator(critical_fields=["id"])
        violations = v.validate(df)
        assert any(viol.severity == "CRITICAL" for viol in violations)

    def test_critical_field_without_nulls_passes(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        v = NullValidator(critical_fields=["id"])
        assert v.validate(df) == []

    def test_global_threshold_exceeded_returns_error(self):
        df = pd.DataFrame({"x": [None] * 8 + [1, 2]})  # 80% null
        v = NullValidator(global_threshold=0.10)
        violations = v.validate(df)
        assert any(viol.severity == "ERROR" for viol in violations)

    def test_per_column_threshold_override(self):
        df = pd.DataFrame({"notes": [None] * 3 + ["x"] * 7})  # 30% null
        v = NullValidator(global_threshold=0.10, column_thresholds={"notes": 0.40})
        violations = v.validate(df)
        # 30% < 40% → no ERROR (may get WARNING if above warn_threshold)
        assert not any(viol.severity == "ERROR" for viol in violations)

    def test_violations_sorted_critical_first(self):
        df = pd.DataFrame({"id": [None, 2], "x": [None] * 2})
        v = NullValidator(critical_fields=["id"], global_threshold=0.10)
        violations = v.validate(df)
        if len(violations) > 1:
            assert violations[0].severity == "CRITICAL"

    def test_empty_dataframe_returns_no_violations(self):
        v = NullValidator()
        assert v.validate(pd.DataFrame()) == []


# ─────────────────────────────────────────────────────────────
# RangeValidator
# ─────────────────────────────────────────────────────────────

class TestRangeValidator:
    def test_bound_violation_below_min(self):
        df = pd.DataFrame({"age": [-1, 25, 40]})
        v = RangeValidator(bound_rules=[BoundRule("age", min_value=0)])
        violations = v.validate(df)
        assert any("BOUND" in viol.rule_type for viol in violations)

    def test_bound_violation_above_max(self):
        df = pd.DataFrame({"age": [50, 200, 30]})
        v = RangeValidator(bound_rules=[BoundRule("age", max_value=130)])
        violations = v.validate(df)
        assert len(violations) == 1
        assert violations[0].offending_count == 1

    def test_positivity_strict(self):
        df = pd.DataFrame({"amount": [0.0, 100.0, -5.0]})
        v = RangeValidator(positivity_rules=[PositivityRule("amount", strict=True)])
        violations = v.validate(df)
        assert violations[0].offending_count == 2  # 0 and -5 both fail strict > 0

    def test_positivity_non_strict(self):
        df = pd.DataFrame({"amount": [0.0, 100.0, -5.0]})
        v = RangeValidator(positivity_rules=[PositivityRule("amount", strict=False)])
        violations = v.validate(df)
        assert violations[0].offending_count == 1  # only -5 fails >= 0

    def test_logical_rule_violation(self):
        df = pd.DataFrame({"loan": [500.0, 900.0], "limit": [400.0, 1000.0]})
        v = RangeValidator(logical_rules=[LogicalRule("loan", "<=", "limit")])
        violations = v.validate(df)
        assert violations[0].offending_count == 1  # 500 > 400

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        v = RangeValidator(bound_rules=[BoundRule("nonexistent", min_value=0)])
        assert v.validate(df) == []  # gracefully skipped

    def test_all_clean_returns_empty(self, clean_df):
        v = RangeValidator(
            bound_rules=[BoundRule("age", min_value=0, max_value=130)],
            positivity_rules=[PositivityRule("amount", strict=True)],
        )
        assert v.validate(clean_df) == []


# ─────────────────────────────────────────────────────────────
# SchemaValidator — new checks
# ─────────────────────────────────────────────────────────────

class TestSchemaValidatorExtended:
    CONFIG = MINIMAL_CONFIG

    def test_required_column_missing(self, clean_df):
        sv = SchemaValidator(self.CONFIG)
        errors = sv.validate(clean_df, {"required_columns": ["missing_col"]})
        assert any(e["type"] == "MISSING_REQUIRED_COLUMN" for e in errors)
        assert any(e["severity"] == "CRITICAL" for e in errors)

    def test_required_columns_present_passes(self, clean_df):
        sv = SchemaValidator(self.CONFIG)
        errors = sv.validate(clean_df, {"required_columns": ["id", "amount"]})
        assert not any(e["type"] == "MISSING_REQUIRED_COLUMN" for e in errors)

    def test_unique_key_violation(self):
        df = pd.DataFrame({"id": [1, 1, 2, 3]})
        sv = SchemaValidator(self.CONFIG)
        errors = sv.validate(df, {"unique_keys": ["id"]})
        assert any(e["type"] == "UNIQUE_KEY_VIOLATION" for e in errors)

    def test_compound_unique_key_violation(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "cust": ["A", "A"]})
        sv = SchemaValidator(self.CONFIG)
        errors = sv.validate(df, {"unique_keys": [["date", "cust"]]})
        assert any(e["type"] == "UNIQUE_KEY_VIOLATION" for e in errors)

    def test_future_timestamp_flagged(self):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)
        df = pd.DataFrame({"created_at": [pd.Timestamp.now(tz="UTC"), future]})
        sv = SchemaValidator(self.CONFIG)
        errors = sv.validate(df, {"timestamp_columns": ["created_at"]})
        assert any(e["type"] == "FUTURE_TIMESTAMP" for e in errors)

    def test_valid_timestamps_pass(self):
        df = pd.DataFrame({"ts": [pd.Timestamp("2023-01-01", tz="UTC")]})
        sv = SchemaValidator(self.CONFIG)
        errors = sv.validate(df, {"timestamp_columns": ["ts"]})
        assert not any(e["type"] in ("FUTURE_TIMESTAMP", "PRE_EPOCH_TIMESTAMP") for e in errors)


# ─────────────────────────────────────────────────────────────
# IntegrityChecker — extended
# ─────────────────────────────────────────────────────────────

class TestIntegrityCheckerExtended:
    def _cfg(self, **kwargs) -> dict:
        cfg = dict(MINIMAL_CONFIG)
        cfg["validation"]["integrity"] = {
            "check_duplicates": True,
            "id_columns": [],
            "referential": [],
            "cross_column_rules": [],
            **kwargs,
        }
        return cfg

    def test_duplicate_rows_flagged(self):
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
        ic = IntegrityChecker(self._cfg())
        errors = ic.check(df)
        assert any(e["type"] == "DUPLICATE_ROWS" for e in errors)

    def test_referential_integrity_violation(self, clean_df):
        cfg = self._cfg(referential=[
            {"column": "status", "allowed_values": ["ACTIVE"]}
        ])
        ic = IntegrityChecker(cfg)
        errors = ic.check(clean_df)  # clean_df has "CLOSED" which is not in ACTIVE
        assert any(e["type"] == "REFERENTIAL_INTEGRITY_VIOLATION" for e in errors)

    def test_cross_column_condition_not_null(self):
        df = pd.DataFrame({
            "status": ["CLOSED", "ACTIVE"],
            "close_date": [None, None],   # CLOSED row missing close_date
        })
        cfg = self._cfg(cross_column_rules=[{
            "if_col": "status", "if_value": "CLOSED",
            "then_col": "close_date", "then_condition": "not_null",
        }])
        ic = IntegrityChecker(cfg)
        errors = ic.check(df)
        assert any(e["type"] == "CROSS_COLUMN_CONSISTENCY_VIOLATION" for e in errors)


# ─────────────────────────────────────────────────────────────
# Banking Rules
# ─────────────────────────────────────────────────────────────

class TestBankingRules:
    def test_positive_amount_catches_negatives(self):
        df = pd.DataFrame({"amount": [100.0, -50.0, 200.0]})
        rule = PositiveAmountRule(amount_columns=["amount"])
        violations = rule.evaluate(df)
        assert len(violations) == 1
        assert violations[0].offending_count == 1

    def test_aml_threshold_flags_large_transactions(self):
        df = pd.DataFrame({"amount": [500.0, 15000.0, 200.0]})
        rule = AMLThresholdRule(amount_column="amount", threshold=10000.0)
        violations = rule.evaluate(df)
        assert len(violations) == 1
        assert violations[0].severity == "WARNING"

    def test_loan_ratio_violation(self):
        df = pd.DataFrame({"loan": [95.0], "value": [100.0]})  # LTV = 95%
        rule = LoanRatioRule(loan_col="loan", value_col="value", max_ltv=0.90)
        violations = rule.evaluate(df)
        assert len(violations) == 1

    def test_repayment_exceeds_balance(self):
        df = pd.DataFrame({"repayment": [600.0], "balance": [500.0]})
        rule = RepaymentConsistencyRule(repayment_col="repayment", balance_col="balance")
        violations = rule.evaluate(df)
        assert len(violations) == 1

    def test_all_clean_banking_data(self):
        df = pd.DataFrame({"amount": [100.0, 200.0], "loan": [80.0, 70.0], "value": [100.0, 100.0]})
        v1 = PositiveAmountRule(amount_columns=["amount"])
        v2 = LoanRatioRule(loan_col="loan", value_col="value", max_ltv=0.90)
        assert v1.evaluate(df) == []
        assert v2.evaluate(df) == []


# ─────────────────────────────────────────────────────────────
# Healthcare Rules
# ─────────────────────────────────────────────────────────────

class TestHealthcareRules:
    def test_age_out_of_bounds(self):
        df = pd.DataFrame({"age": [25, 150, -1]})
        rule = AgeRangeRule(age_column="age")
        violations = rule.evaluate(df)
        assert violations[0].offending_count == 2

    def test_age_in_bounds_passes(self):
        df = pd.DataFrame({"age": [0, 45, 130]})
        assert AgeRangeRule().evaluate(df) == []

    def test_vital_signs_out_of_range(self):
        df = pd.DataFrame({"heart_rate": [20, 350, 75]})  # 350 is too high
        rule = VitalSignsRule()
        violations = rule.evaluate(df)
        assert len(violations) == 1

    def test_icd10_format_valid(self):
        df = pd.DataFrame({"diagnosis_code": ["E11.9", "Z00", "M79.3"]})
        rule = DiagnosisCodeFormatRule(diagnosis_columns=["diagnosis_code"])
        assert rule.evaluate(df) == []

    def test_icd10_format_invalid(self):
        df = pd.DataFrame({"diagnosis_code": ["INVALID", "not-a-code"]})
        rule = DiagnosisCodeFormatRule(diagnosis_columns=["diagnosis_code"])
        violations = rule.evaluate(df)
        assert violations[0].offending_count == 2

    def test_phi_detection_ssn(self):
        df = pd.DataFrame({"notes": ["SSN is 123-45-6789", "nothing here"]})
        rule = PHIPresenceRule(text_columns=["notes"])
        violations = rule.evaluate(df)
        assert violations[0].severity == "CRITICAL"


# ─────────────────────────────────────────────────────────────
# HardGate (end-to-end)
# ─────────────────────────────────────────────────────────────

class TestHardGate:
    def test_clean_data_passes(self, clean_df):
        gate = HardGate.from_config(MINIMAL_CONFIG)
        result = gate.run(clean_df, run_id="test-PASS")
        assert result.decision == "PASS"
        assert result.suppress_learning is False

    def test_missing_critical_field_rejects(self, clean_df):
        config = dict(MINIMAL_CONFIG)
        config["validation"]["schema"] = {"required_columns": ["missing_required_col"]}
        gate = HardGate.from_config(config)
        result = gate.run(clean_df, run_id="test-REJECT-schema")
        assert result.decision == "REJECT"
        assert result.suppress_learning is True
        assert result.total_violations >= 1

    def test_null_critical_field_rejects(self):
        df = pd.DataFrame({"id": [None, 2, 3], "value": [1.0, 2.0, 3.0]})
        config = dict(MINIMAL_CONFIG)
        config["validation"]["null"] = {
            "global_threshold": 0.10,
            "warn_threshold": 0.05,
            "critical_fields": ["id"],
        }
        gate = HardGate.from_config(config)
        result = gate.run(df, run_id="test-REJECT-null")
        assert result.decision == "REJECT"
        assert result.suppress_learning is True

    def test_gate_result_has_required_fields(self, clean_df):
        gate = HardGate.from_config(MINIMAL_CONFIG)
        result = gate.run(clean_df, run_id="test-fields")
        d = result.to_dict()
        for key in ("run_id", "decision", "reason", "suppress_learning", "failures", "warnings"):
            assert key in d

    def test_regulatory_phi_causes_rejection(self):
        df = pd.DataFrame({
            "notes": ["Patient SSN: 123-45-6789"],
            "age": [45],
            "diagnosis_code": ["E11.9"],
        })
        config = dict(MINIMAL_CONFIG)
        config["validation"]["regulatory"] = {
            "domain": "healthcare",
            "healthcare": {
                "age_column": "age",
                "diagnosis_columns": ["diagnosis_code"],
                "text_columns_for_phi_scan": ["notes"],
                "allowed_phi_columns": [],
            },
        }
        gate = HardGate.from_config(config)
        result = gate.run(df, run_id="test-PHI-reject")
        assert result.decision == "REJECT"
        assert result.suppress_learning is True
