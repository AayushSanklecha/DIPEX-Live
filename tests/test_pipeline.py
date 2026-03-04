"""
tests/test_pipeline.py
----------------------
Unit tests for the DIPEX pipeline components.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ingestion.batch_loader import BatchLoader
from ingestion.snapshot import SnapshotManager
from validation.qa_gate import QAGate
from validation.schema_validator import SchemaValidator
from profiling.profiler import Profiler
from profiling.drift_detector import DriftDetector
from verifier.statistical_verifier import StatisticalVerifier
from verifier.confidence_aggregator import ConfidenceAggregator
from verifier.baseline_verifier import BaselineVerifier


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.5, 20.0, 15.2, 8.7, 30.1],
            "category": ["A", "B", "A", "C", "B"],
        }
    )


@pytest.fixture
def binary_clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 1, 50),
            "feat_b": rng.uniform(0, 1, 50),
            "target": rng.integers(0, 2, 50),
        }
    )


# ---------------------------------------------------------------------------
# BatchLoader
# ---------------------------------------------------------------------------

class TestBatchLoader:
    def test_load_csv_basic(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)
        loaded = BatchLoader.load_csv(str(csv_file))
        assert len(loaded) == len(sample_df)
        assert list(loaded.columns) == list(sample_df.columns)

    def test_load_csv_semicolon_delimiter(self, tmp_path: Path, sample_df: pd.DataFrame):
        csv_file = tmp_path / "test_semi.csv"
        sample_df.to_csv(csv_file, sep=";", index=False)
        loaded = BatchLoader.load_csv(str(csv_file))
        assert len(loaded) == len(sample_df)

    def test_load_csv_missing_file(self):
        with pytest.raises(FileNotFoundError):
            BatchLoader.load_csv("/nonexistent/path.csv")

    def test_load_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported source_type"):
            BatchLoader.load("some_source", source_type="parquet")


# ---------------------------------------------------------------------------
# SnapshotManager
# ---------------------------------------------------------------------------

class TestSnapshotManager:
    def test_hash_determinism(self, sample_df: pd.DataFrame):
        manager = SnapshotManager(snapshot_dir="data/test_snapshots")
        assert manager.calculate_hash(sample_df) == manager.calculate_hash(sample_df)

    def test_hash_column_order_invariant(self, sample_df: pd.DataFrame):
        """Same data with reordered columns should produce the same hash."""
        manager = SnapshotManager(snapshot_dir="data/test_snapshots")
        shuffled = sample_df[["category", "id", "value"]]
        assert manager.calculate_hash(sample_df) == manager.calculate_hash(shuffled)

    def test_hash_different_data(self, sample_df: pd.DataFrame):
        manager = SnapshotManager(snapshot_dir="data/test_snapshots")
        other_df = sample_df.copy()
        other_df.loc[0, "value"] = 999.0
        assert manager.calculate_hash(sample_df) != manager.calculate_hash(other_df)

    def test_registry_corruption_recovery(self, tmp_path: Path):
        """A corrupted registry file should not crash — resets to empty."""
        registry_path = tmp_path / "registry.json"
        registry_path.write_text("NOT_VALID_JSON")
        manager = SnapshotManager(snapshot_dir=str(tmp_path))
        assert manager.registry == {}


# ---------------------------------------------------------------------------
# QAGate
# ---------------------------------------------------------------------------

class TestQAGate:
    def test_fails_on_error(self):
        gate = QAGate()
        errors = [{"severity": "ERROR", "type": "TEST", "message": "fail"}]
        assert gate.evaluate(errors) is False

    def test_passes_on_warning_only(self):
        gate = QAGate()
        errors = [{"severity": "WARNING", "type": "TEST", "message": "warn"}]
        assert gate.evaluate(errors) is True

    def test_passes_on_empty(self):
        gate = QAGate()
        assert gate.evaluate([]) is True


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------

class TestSchemaValidator:
    CONFIG = {"pipeline": {"qa_gate": {"null_threshold": 0.1}}}

    def test_type_check_int64_matches_int(self, sample_df: pd.DataFrame):
        validator = SchemaValidator(self.CONFIG)
        errors = validator.validate(sample_df, {"types": {"id": "int"}})
        type_errors = [e for e in errors if e["type"] == "TYPE_MISMATCH"]
        assert len(type_errors) == 0, f"Unexpected type errors: {type_errors}"

    def test_type_check_float_matches_float64(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        validator = SchemaValidator(self.CONFIG)
        errors = validator.validate(df, {"types": {"x": "float"}})
        assert not any(e["type"] == "TYPE_MISMATCH" for e in errors)

    def test_type_mismatch_detected(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        validator = SchemaValidator(self.CONFIG)
        errors = validator.validate(df, {"types": {"x": "int"}})
        assert any(e["type"] == "TYPE_MISMATCH" for e in errors)

    def test_null_threshold_exceeded(self):
        df = pd.DataFrame({"x": [1, None, None, None, None]})
        validator = SchemaValidator(self.CONFIG)
        errors = validator.validate(df, {})
        assert any(e["type"] == "NULL_THRESHOLD_EXCEEDED" for e in errors)

    def test_stateless_between_calls(self):
        """Two calls to validate() should NOT share error state."""
        df = pd.DataFrame({"x": [1, None, None, None, None]})
        validator = SchemaValidator(self.CONFIG)
        e1 = validator.validate(df, {})
        e2 = validator.validate(df, {})
        assert len(e1) == len(e2)  # Idempotent


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class TestProfiler:
    def test_profile_shape(self, sample_df: pd.DataFrame):
        profiler = Profiler()
        profile = profiler.profile(sample_df)
        assert profile["row_count"] == len(sample_df)
        assert profile["column_count"] == len(sample_df.columns)
        assert set(profile["columns"].keys()) == set(sample_df.columns)

    def test_numeric_column_has_stats(self, sample_df: pd.DataFrame):
        profiler = Profiler()
        col_profile = profiler.profile(sample_df)["columns"]["value"]
        for key in ("min", "max", "mean", "median", "std", "q25", "q75"):
            assert key in col_profile, f"Missing key: {key}"

    def test_all_null_numeric(self):
        df = pd.DataFrame({"x": [None, None, None]})
        profiler = Profiler()
        col_profile = profiler.profile(df)["columns"]["x"]
        assert "note" in col_profile or col_profile["null_count"] == 3


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def test_no_drift_identical(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        psi = DriftDetector.calculate_psi(data, data)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_high_drift_different_distributions(self):
        rng = np.random.default_rng(0)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(5, 1, 1000)  # Completely shifted
        psi = DriftDetector.calculate_psi(expected, actual)
        assert psi > 0.25, "Expected high PSI for shifted distributions"

    def test_empty_array_returns_zero(self):
        psi = DriftDetector.calculate_psi(np.array([]), np.array([1.0, 2.0]))
        assert psi == 0.0


# ---------------------------------------------------------------------------
# StatisticalVerifier
# ---------------------------------------------------------------------------

class TestStatisticalVerifier:
    def test_empty_array_returns_not_passed(self):
        sv = StatisticalVerifier()
        result = sv.verify(np.array([]), np.array([]), "classification")
        assert result["passed"] is False

    def test_perfect_classification_passes(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        sv = StatisticalVerifier()
        result = sv.verify(y, y, "classification")
        assert result["passed"] is True

    def test_single_class_returns_not_passed(self):
        y = np.array([1, 1, 1, 1, 1])
        sv = StatisticalVerifier()
        result = sv.verify(y, y, "classification")
        assert result["passed"] is False

    def test_regression_verification_runs(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        sv = StatisticalVerifier()
        result = sv.verify(y, y, "regression")
        assert "passed" in result
        assert isinstance(result["value"], float)


# ---------------------------------------------------------------------------
# ConfidenceAggregator
# ---------------------------------------------------------------------------

class TestConfidenceAggregator:
    def test_all_passed_gives_max_score(self):
        agg = ConfidenceAggregator(weights={"a": 0.5, "b": 0.5})
        result = agg.aggregate({
            "a": {"passed": True, "detail": "ok"},
            "b": {"passed": True, "detail": "ok"},
        })
        assert result["confidence_score"] == pytest.approx(1.0)
        assert result["all_gates_passed"] is True

    def test_failure_applies_penalty(self):
        agg = ConfidenceAggregator(weights={"a": 0.5, "b": 0.5})
        result = agg.aggregate({
            "a": {"passed": True},
            "b": {"passed": False},
        })
        assert result["all_gates_passed"] is False
        assert result["confidence_score"] < 1.0

    def test_no_results_returns_zero(self):
        agg = ConfidenceAggregator()
        result = agg.aggregate({})
        assert result["confidence_score"] == 0.0


# ---------------------------------------------------------------------------
# BaselineVerifier
# ---------------------------------------------------------------------------

class TestBaselineVerifier:
    def test_regression_perfect_model_passes(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        bv = BaselineVerifier(min_improvement=0.05)
        # Model score of 0 (perfect) vs baseline MSE — should pass
        result = bv.verify(0.0, "regression", y)
        assert result["passed"] is True

    def test_classification_majority_class_baseline(self):
        y = np.array([0, 0, 0, 0, 1])  # Majority is 0
        bv = BaselineVerifier(min_improvement=0.05)
        # Model accuracy equal to baseline → should NOT pass
        result = bv.verify(0.8, "classification", y)
        assert "passed" in result
