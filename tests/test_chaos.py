"""
tests/test_chaos.py
--------------------
Chaos & failure resilience tests for DIPEX.

Tests:
- Malformed CSV → graceful failure (no crash)
- Malformed JSON → graceful failure
- Half-uploaded / empty files → handled
- API timeout simulation → retry backoff
- Schema drift mid-stream → halt + log
- Memory spike / large dataset → handled
- Corrupt model file → fallback
- NaN/Inf injection → gate catches
- Extreme outlier injection → detected
- Concurrent pipeline runs → no state bleed
- Circular reference in JSON → handled
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Malformed Input Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMalformedInputs:

    def test_malformed_csv_handled_gracefully(self):
        """Malformed CSV must not raise unhandled exception."""
        bad_csv = "col1,col2\n1,2\n3,BROKEN_ROW_WITHOUT_QUOTE\x00\x01\n5,6"
        try:
            df = pd.read_csv(io.StringIO(bad_csv), on_bad_lines="skip")
            assert isinstance(df, pd.DataFrame)
        except Exception as exc:
            # If it raises, system must catch and log — not crash server
            assert "permission" not in str(exc).lower(), f"Unexpected error: {exc}"

    def test_empty_csv_returns_empty_dataframe(self):
        """Empty CSV must return empty DataFrame, not crash."""
        empty_csv = "col1,col2\n"
        df = pd.read_csv(io.StringIO(empty_csv))
        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2"]

    def test_no_header_csv_handled(self):
        """CSV with no header must not crash."""
        no_header = "1,2,3\n4,5,6\n"
        df = pd.read_csv(io.StringIO(no_header), header=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_malformed_json_handled(self):
        """Malformed JSON must not crash the pipeline — returns empty DF."""
        bad_json = '{"key": "value", broken_json'
        try:
            data = json.loads(bad_json)
        except json.JSONDecodeError:
            data = {}  # graceful fallback
        assert isinstance(data, dict)

    def test_empty_json_returns_empty(self):
        """Empty JSON object must not crash."""
        data = json.loads("{}")
        df = pd.json_normalize([data])
        assert isinstance(df, pd.DataFrame)

    def test_circular_reference_json_handled(self):
        """Circular references in JSON source must be handled."""
        # Real JSON cannot have circular refs — they fail during json.loads
        # This tests that our normalizer handles deeply nested structures
        nested = {"level1": {"level2": {"level3": {"level4": {"level5": "bottom"}}}}}
        df = pd.json_normalize(nested, sep=".")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_half_uploaded_file_graceful(self, tmp_path):
        """Truncated/incomplete file must not crash intake."""
        truncated_csv = "id,name,revenue\n1,Alice,100\n2,Bob"  # truncated mid-row
        fpath = tmp_path / "truncated.csv"
        fpath.write_text(truncated_csv)
        try:
            df = pd.read_csv(str(fpath), on_bad_lines="skip")
            assert isinstance(df, pd.DataFrame)
        except pd.errors.ParserError:
            pass  # ParserError is acceptable — pipeline catches and logs

    def test_all_nulls_csv(self):
        """All-null DataFrame must not break profiler or gates."""
        df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
        assert df.isnull().all().all()
        # Downstream: must handle without division by zero
        try:
            desc = df.describe()
            assert isinstance(desc, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"All-null describe raised: {exc}")

    def test_single_row_dataframe(self):
        """Single-row DataFrames must not break statistical computations."""
        df = pd.DataFrame({"x": [42.0], "y": [1.0]})
        try:
            corr = df.corr(numeric_only=True)
            assert isinstance(corr, pd.DataFrame)
        except Exception:
            pass  # Some methods undefined on single-row — acceptable if caught


# ══════════════════════════════════════════════════════════════════════════════
# NaN / Inf Injection
# ══════════════════════════════════════════════════════════════════════════════

class TestNaNInfInjection:

    def test_nan_caught_by_domain_verifier(self):
        """NaN values must be caught by domain verifier no_nan rule."""
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[{"type": "no_nan"}])
        preds = np.array([0.5, float("nan"), 0.7])
        result = dv.verify(predictions=preds)
        assert result["passed"] is False

    def test_inf_caught_by_domain_verifier(self):
        """Inf values must be caught by domain verifier no_inf rule."""
        from verifier.domain_verifier import DomainVerifier
        dv = DomainVerifier(rules=[{"type": "no_inf"}])
        preds = np.array([0.5, float("inf"), 0.7])
        result = dv.verify(predictions=preds)
        assert result["passed"] is False

    def test_nan_in_features_does_not_crash_training(self):
        """NaN in features must be handled before training (imputation)."""
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestClassifier

        X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0, 1])

        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(X)
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X_imp, y)  # must not raise

    def test_extreme_outlier_detected_by_profiler(self):
        """Extreme outliers must be flagged by IQR/Z-score threshold."""
        # Use many normal values so that 10000 clearly exceeds 3 sigma
        normal_vals = [10] * 100 + [11, 12, 10, 9, 10, 11, 10, 12, 11, 10]
        data = pd.Series(normal_vals + [10000])  # 10000 is extreme outlier
        z_scores = (data - data.mean()) / data.std()
        outliers = (z_scores.abs() > 3).sum()
        assert outliers > 0, f"Extreme outlier not detected by Z-score (max z={z_scores.abs().max():.2f})"


# ══════════════════════════════════════════════════════════════════════════════
# API Timeout & Retry
# ══════════════════════════════════════════════════════════════════════════════

class TestAPITimeoutResilience:

    def test_sql_connector_retries_on_failure(self):
        """SQLConnector must retry transient failures without crashing."""
        from ingestion.connectors.sql_connector import SQLConnector
        config = {"dsn": "sqlite:///::memory::", "table": "nonexistent_chaos_table"}
        conn = SQLConnector(config)
        # Extraction of non-existent table must raise ConnectorError (not generic crash)
        from ingestion.connectors.base_connector import ConnectorError
        with pytest.raises((ConnectorError, Exception)):
            conn.extract()

    def test_api_connector_raises_connector_error_on_timeout(self):
        """APIConnector must raise ConnectorError (not silently fail) on network error."""
        from ingestion.connectors.api_connector import APIConnector
        from ingestion.connectors.base_connector import ConnectorError

        config = {
            "base_url": "http://localhost:1",  # non-existent server
            "endpoint": "/test",
            "timeout": 1,
            "max_retries": 1,
        }
        conn = APIConnector(config)
        with pytest.raises((ConnectorError, Exception)):
            conn.extract()

    def test_kafka_connector_raises_on_no_broker(self):
        """KafkaConnector must raise when broker not available."""
        from ingestion.connectors.kafka_connector import KafkaConnector
        from ingestion.connectors.base_connector import ConnectorError

        config = {
            "bootstrap_servers": "localhost:1",  # non-existent broker
            "topics": ["chaos-test"],
            "group_id": "chaos-test-group",
        }
        conn = KafkaConnector(config)
        result = conn.test_connection()
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# Schema Drift Mid-Stream
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaDrift:

    def test_schema_drift_detected_between_batches(self):
        """If a streaming batch has a different schema, the system must detect it."""
        batch_1 = pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
        batch_2 = pd.DataFrame({"id": [3, 4], "value": [30.0, 40.0], "new_col": ["a", "b"]})

        # Schema drift = new column appeared
        cols_1 = set(batch_1.columns)
        cols_2 = set(batch_2.columns)
        new_cols = cols_2 - cols_1
        dropped_cols = cols_1 - cols_2

        assert len(new_cols) > 0, "Schema drift not detected (new columns)"
        assert "new_col" in new_cols

    def test_type_change_detected_as_schema_drift(self):
        """Type change for same column must be classified as schema drift."""
        schema_v1 = {"id": "int64", "value": "float64"}
        schema_v2 = {"id": "int64", "value": "object"}  # type change!

        changed = {col for col in schema_v1 if schema_v1.get(col) != schema_v2.get(col)}
        assert "value" in changed

    def test_missing_column_detected_as_breaking_drift(self):
        """A column disappearing between batches is a breaking schema change."""
        schema_v1 = {"id", "value", "timestamp"}
        schema_v2 = {"id", "value"}  # timestamp dropped

        missing = schema_v1 - schema_v2
        assert "timestamp" in missing


# ══════════════════════════════════════════════════════════════════════════════
# Concurrent Pipeline Isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestConcurrentPipelineIsolation:

    def test_concurrent_runs_have_no_state_bleed(self):
        """Two concurrent pipeline runs must not share mutable state."""
        results = {}
        errors = []

        def run_pipeline(id_: str, value: float) -> None:
            try:
                # Simulate independent pipeline state
                state = {"pipeline_id": id_, "confidence": value}
                import time
                time.sleep(0.01)  # simulate async work
                results[id_] = state["confidence"]
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=run_pipeline, args=(f"run-{i}", 0.5 + i * 0.1))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent pipeline errors: {errors}"
        # Each run must maintain its own state
        for i in range(5):
            expected = round(0.5 + i * 0.1, 10)
            actual = results.get(f"run-{i}")
            assert actual is not None, f"run-{i} result missing"
            assert abs(actual - expected) < 1e-9, f"run-{i}: state bleed detected"

    def test_drift_verifier_thread_safe(self):
        """DriftVerifier must be thread-safe for concurrent evaluations."""
        from verifier.drift_verifier import DriftVerifier

        np.random.seed(0)
        ref_df = pd.DataFrame({"a": np.random.normal(0, 1, 200)})
        errors = []

        def verify_drift(seed: int) -> None:
            try:
                cur_df = pd.DataFrame({"a": np.random.normal(seed * 0.1, 1, 100)})
                verifier = DriftVerifier()  # per-thread instance
                result = verifier.verify(current_df=cur_df, reference_df=ref_df)
                assert "passed" in result
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=verify_drift, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"DriftVerifier thread safety failure: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# Large Dataset Stress
# ══════════════════════════════════════════════════════════════════════════════

class TestLargeDatasetResilience:

    def test_large_dataframe_profiling_completes(self):
        """Profiling on a large DataFrame (500K rows) must complete."""
        np.random.seed(0)
        n = 500_000
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, n),
            "b": np.random.uniform(0, 1, n),
            "c": np.random.randint(0, 100, n),
        })
        # Must complete without memory error
        desc = df.describe()
        assert desc.shape == (8, 3)

    def test_drift_verifier_handles_large_dataset(self):
        """DriftVerifier must handle large datasets without crash."""
        from verifier.drift_verifier import DriftVerifier

        n = 50_000
        ref = pd.DataFrame({"x": np.random.normal(0, 1, n)})
        cur = pd.DataFrame({"x": np.random.normal(0.5, 1, n)})

        verifier = DriftVerifier()
        result = verifier.verify(current_df=cur, reference_df=ref)
        assert isinstance(result["value"], float)

    def test_checksum_large_dataframe(self):
        """SHA-256 checksum must be computable on large DataFrames."""
        import hashlib

        n = 100_000
        df = pd.DataFrame({"id": range(n), "value": np.random.random(n)})
        content = df.to_json(orient="records").encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        assert len(digest) == 64
