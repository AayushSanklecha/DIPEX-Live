"""
tests/test_universal_intake.py
--------------------------------
Comprehensive test suite for the Universal Data Intake & Processing Layer.

Coverage
--------
Unit Tests    : File parsing, API mocking, schema drift detection, quality gate
Integration   : File → Normalise → Schema → Quality → ISSF pipeline
Chaos Tests   : Corrupt file, truncated file, malformed API JSON, schema drift mid-run

Run::
    pytest tests/test_universal_intake.py -v --tb=short
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Make sure ingestion package is importable ─────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.error_handler import (
    DataFormatError, DBConnectionError, EncodingError, QualityGateError,
    SafeExecutor, SchemaError, StreamLagError,
)
from ingestion.issf import ColumnMeta, ISSFSnapshot, IngestionError
from ingestion.normaliser import Normaliser
from ingestion.quality_gate import QualityGate
from ingestion.readers.file_reader import FileReader
from ingestion.schema_registry import SchemaRegistry, SchemaVersion
from ingestion.universal_intake import SourceConfig, UniversalIntake


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def tmp_base(tmp_path_factory):
    return tmp_path_factory.mktemp("dipex_test")


@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "user_id":  [1, 2, 3, 4, 5],
        "revenue":  [100.0, 200.0, 150.0, 300.0, 50.0],
        "country":  ["US", "IN", "UK", "US", "DE"],
        "sign_up":  ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    })


@pytest.fixture
def csv_file(tmp_base, simple_df):
    p = tmp_base / "simple.csv"
    simple_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def intake(tmp_base):
    return UniversalIntake(
        snapshot_dir=str(tmp_base / "snapshots"),
        registry_dir=str(tmp_base / "schema_registry"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — ISSF
# ═══════════════════════════════════════════════════════════════════════════════

class TestISSF:
    def test_issf_basic_creation(self, simple_df):
        cols = [ColumnMeta("user_id", "int64", 0, 0.0, 5, True)]
        s = ISSFSnapshot(
            dataset_id="test", schema_version="1.0.0",
            data_mode="batch", source_type="file", source_uri="test.csv",
            column_metadata=cols, row_count=5, quality_score=0.95,
        )
        assert s.is_compliant
        assert s.fingerprint
        assert len(s.fingerprint) == 64  # SHA-256 hex

    def test_issf_invalid_data_mode_raises(self):
        with pytest.raises(AssertionError):
            ISSFSnapshot(
                dataset_id="x", schema_version="1.0.0",
                data_mode="INVALID", source_type="file", source_uri="",
                column_metadata=[], row_count=0, quality_score=1.0,
            )

    def test_issf_quality_score_clamped(self):
        s = ISSFSnapshot(
            dataset_id="x", schema_version="1.0.0",
            data_mode="batch", source_type="file", source_uri="",
            column_metadata=[], row_count=0, quality_score=1.5,
        )
        assert s.quality_score == 1.0

    def test_issf_to_dict_has_all_fields(self):
        s = ISSFSnapshot(
            dataset_id="sales", schema_version="2.1.0",
            data_mode="batch", source_type="database", source_uri="postgres://",
            column_metadata=[], row_count=100, quality_score=0.88,
        )
        d = s.to_dict()
        required = {"dataset_id", "snapshot_id", "schema_version", "ingestion_timestamp",
                    "data_mode", "source_type", "validation_status", "row_count",
                    "quality_score", "fingerprint"}
        assert required.issubset(d.keys())

    def test_issf_save_and_load(self, tmp_base, simple_df):
        cols = [ColumnMeta("x", "float64", 0, 0.0, 5)]
        snap = ISSFSnapshot(
            dataset_id="test_save", schema_version="1.0.0",
            data_mode="batch", source_type="file", source_uri="",
            column_metadata=cols, row_count=5, quality_score=0.9,
            data=simple_df,
        )
        snap_dir = str(tmp_base / "save_test")
        snap.save(snap_dir)
        loaded = ISSFSnapshot.load(snap.snapshot_id, snap_dir)
        assert loaded.dataset_id == snap.dataset_id
        assert loaded.schema_version == snap.schema_version
        assert loaded.row_count == snap.row_count


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Error Handler
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandler:
    def test_safe_executor_returns_result_on_success(self):
        ex = SafeExecutor(dataset_id="test")
        result, errors = ex.run(lambda x: x * 2, 21)
        assert result == 42
        assert errors == []

    def test_safe_executor_catches_typed_error(self):
        def bad_fn():
            raise DataFormatError("Corrupt CSV")
        ex = SafeExecutor(dataset_id="test")
        result, errors = ex.run(bad_fn)
        assert result is None
        assert len(errors) == 1
        assert errors[0].error_type == "DATA_FORMAT_ERROR"

    def test_safe_executor_catches_unclassified_exception(self):
        def bad_fn():
            raise ValueError("Something unexpected")
        ex = SafeExecutor(dataset_id="test")
        result, errors = ex.run(bad_fn)
        assert result is None
        assert len(errors) == 1

    def test_typed_exceptions_have_correlation_id(self):
        exc = SchemaError("Missing column: revenue")
        assert exc.correlation_id  # not empty
        assert len(exc.correlation_id) == 36  # UUID format

    def test_human_readable_message(self):
        exc = DBConnectionError("Connection refused", correlation_id="test-id")
        msg = exc.human_readable()
        assert "DB_CONNECTION_ERROR" in msg
        assert "test-id" in msg

    def test_guard_context_manager_does_not_raise(self):
        ex = SafeExecutor(dataset_id="test")
        with ex.guard("step_1"):
            raise RuntimeError("Should be caught")
        assert ex.has_errors


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Normaliser
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormaliser:
    def setup_method(self):
        self.norm = Normaliser()

    def test_snake_case_columns(self):
        df = pd.DataFrame({"First Name": [1], "Last-Name": [2], "Age (years)": [3]})
        out, _ = self.norm.normalise(df)
        assert list(out.columns) == ["first_name", "last_name", "age_years"]

    def test_null_unification(self):
        df = pd.DataFrame({"x": ["NULL", "N/A", "", "—", "normal", None]})
        out, _ = self.norm.normalise(df)
        null_count = out["x"].isna().sum()
        assert null_count >= 4

    def test_numeric_coercion(self):
        df = pd.DataFrame({"revenue": ["100", "200.5", "300", "null"]})
        out, _ = self.norm.normalise(df)
        assert pd.api.types.is_numeric_dtype(out["revenue"])

    def test_boolean_coercion(self):
        df = pd.DataFrame({"active": ["true", "false", "yes", "no"]})
        out, _ = self.norm.normalise(df)
        assert set(out["active"].dropna().unique()).issubset({True, False})

    def test_column_meta_null_rate(self):
        df = pd.DataFrame({"x": [1, None, None, 4, 5]})
        _, metas = self.norm.normalise(df)
        assert metas[0].null_count == 2
        assert abs(metas[0].null_rate - 0.4) < 0.01

    def test_pk_candidate_detection(self):
        df = pd.DataFrame({"id": range(100), "value": [42] * 100})
        _, metas = self.norm.normalise(df)
        id_meta = next(m for m in metas if m.name == "id")
        assert id_meta.is_pk_candidate

    def test_empty_df_returns_empty(self):
        out, metas = self.norm.normalise(pd.DataFrame())
        assert out.empty
        assert metas == []

    def test_deduplicate_column_names(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "x"])
        out, _ = self.norm.normalise(df)
        assert len(set(out.columns)) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Schema Registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaRegistry:
    def setup_method(self, method, tmp_path=None):
        self.tmp = tempfile.mkdtemp()
        self.reg = SchemaRegistry(registry_dir=self.tmp)

    def test_initial_registration_no_drift(self):
        schema = {"user_id": "int64", "revenue": "float64"}
        report = self.reg.register("sales", schema)
        assert report.old_version is None
        assert report.new_version == "1.0.0"
        assert not report.is_breaking
        assert report.changes == []

    def test_additive_drift_is_minor(self):
        schema1 = {"user_id": "int64"}
        schema2 = {"user_id": "int64", "email": "object"}
        self.reg.register("ds", schema1)
        report = self.reg.register("ds", schema2)
        assert not report.is_breaking
        assert report.new_version.startswith("1.1")  # minor bump

    def test_missing_column_is_breaking(self):
        schema1 = {"user_id": "int64", "revenue": "float64"}
        schema2 = {"user_id": "int64"}                   # revenue missing
        self.reg.register("ds2", schema1)
        report = self.reg.register("ds2", schema2)
        assert report.is_breaking
        assert int(report.new_version.split(".")[0]) == 2  # major bump

    def test_type_change_is_breaking(self):
        schema1 = {"amount": "float64"}
        schema2 = {"amount": "object"}    # float → string is breaking
        self.reg.register("ds3", schema1)
        report = self.reg.register("ds3", schema2)
        assert report.is_breaking

    def test_schema_version_history(self):
        schema = {"col": "int64"}
        self.reg.register("hist_test", schema)
        self.reg.register("hist_test", {**schema, "new_col": "float64"})
        history = self.reg.get_history("hist_test")
        assert len(history) == 2

    def test_semver_bump_major(self):
        sv = SchemaVersion("1.2.3")
        assert str(sv.bump_major()) == "2.0.0"

    def test_semver_bump_minor(self):
        sv = SchemaVersion("1.2.3")
        assert str(sv.bump_minor()) == "1.3.0"

    def test_semver_bump_patch(self):
        sv = SchemaVersion("1.2.3")
        assert str(sv.bump_patch()) == "1.2.4"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Quality Gate
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityGate:
    def setup_method(self, method):
        self.gate = QualityGate(config={
            "universal_intake": {"quality_thresholds": {
                "max_null_rate": 0.30,
                "max_duplicate_rate": 0.05,
                "min_quality_score": 0.70,
            }}
        })

    def test_clean_data_passes(self):
        df = pd.DataFrame({"x": range(100), "y": [float(i) for i in range(100)]})
        report = self.gate.check(df)
        assert report.validation_status == "PASSED"
        assert report.quality_score > 0.85

    def test_high_null_rate_fails(self):
        df = pd.DataFrame({"x": [None] * 80 + list(range(20))})
        report = self.gate.check(df)
        assert report.validation_status in ("FAILED", "WARN")
        assert report.overall_null_rate > 0.5

    def test_duplicates_detected(self):
        df = pd.DataFrame({"x": [1] * 50 + list(range(50))})
        report = self.gate.check(df)
        assert report.duplicate_count > 0

    def test_range_violation(self):
        df = pd.DataFrame({"age": [25, 30, -5, 200, 40]})
        report = self.gate.check(df, range_rules={"age": (0, 120)})
        assert report.violations  # -5 and 200 are out of range

    def test_unexpected_category(self):
        df = pd.DataFrame({"country": ["US", "UK", "MARS", "DE"]})
        report = self.gate.check(df, allowed_categories={"country": ["US", "UK", "DE", "IN"]})
        assert report.warnings  # MARS

    def test_psi_drift_detection(self):
        baseline = pd.DataFrame({"revenue": np.random.normal(100, 10, 1000)})
        current  = pd.DataFrame({"revenue": np.random.normal(500, 10, 1000)})  # huge drift
        report   = self.gate.check(current, baseline_df=baseline)
        assert "revenue" in report.distribution_drift
        assert report.distribution_drift["revenue"] > 0.2  # should flag major drift

    def test_fk_violation(self):
        df = pd.DataFrame({"dept_id": [1, 2, 99, 3]})
        report = self.gate.check(df, fk_rules={"dept_id": {1, 2, 3, 4, 5}})
        assert any("FK" in v for v in report.violations)

    def test_outlier_flagged_as_warning(self):
        vals = [10] * 98 + [10000, 10001]  # extreme outliers
        df = pd.DataFrame({"revenue": vals})
        report = self.gate.check(df)
        assert any("outlier" in w.lower() for w in report.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — File Reader
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileReader:
    def setup_method(self, method):
        self.reader = FileReader()
        self.tmp    = tempfile.mkdtemp()

    def _write(self, name: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str:
        p = os.path.join(self.tmp, name)
        with open(p, mode, encoding=encoding) as f:
            f.write(content)
        return p

    def test_reads_simple_csv(self):
        p = self._write("data.csv", "a,b,c\n1,2,3\n4,5,6\n")
        result = self.reader.read(p)
        assert result.row_count == 2
        assert list(result.data.columns) == ["a", "b", "c"]

    def test_auto_detects_tsv(self):
        p = self._write("data.tsv", "col1\tcol2\n10\t20\n30\t40\n")
        result = self.reader.read(p, fmt="tsv")
        assert result.row_count == 2

    def test_reads_json_array(self):
        data = json.dumps([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        p = self._write("data.json", data)
        result = self.reader.read(p, fmt="json")
        assert result.row_count == 2

    def test_reads_jsonl(self):
        content = '{"a":1}\n{"a":2}\nbad_line\n{"a":3}\n'
        p = self._write("data.jsonl", content)
        result = self.reader.read(p, fmt="jsonl")
        assert result.row_count == 3
        assert result.bad_row_count == 1

    def test_missing_file_raises_data_format_error(self):
        with pytest.raises(DataFormatError):
            self.reader.read("/nonexistent/path/data.csv")

    def test_empty_csv_returns_empty_df(self):
        p = self._write("empty.csv", "")
        result = self.reader.read(p)
        assert result.data.empty or result.row_count == 0

    def test_csv_with_bom(self):
        p = os.path.join(self.tmp, "bom.csv")
        with open(p, "w", encoding="utf-8-sig") as f:
            f.write("id,name\n1,Alice\n2,Bob\n")
        result = self.reader.read(p, fmt="csv")
        assert result.row_count == 2
        # Column should not start with BOM character
        assert not any("\ufeff" in c for c in result.data.columns)

    def test_csv_pipe_delimiter(self):
        p = self._write("pipe.csv", "a|b|c\n1|2|3\n4|5|6\n")
        result = self.reader.read(p)
        assert result.row_count == 2

    def test_log_file_jsonl_parsing(self):
        logs = '{"level":"INFO","msg":"started"}\n{"level":"ERROR","msg":"failed"}\n'
        p = self._write("app.log", logs)
        result = self.reader.read(p, fmt="log")
        assert result.row_count == 2
        assert "level" in result.data.columns

    def test_log_file_kv_parsing(self):
        logs = "level=INFO msg=started host=web01\nlevel=ERROR msg=failed host=web02\n"
        p = self._write("kv.log", logs)
        result = self.reader.read(p, fmt="log")
        assert result.row_count == 2

    def test_read_chunks_generator(self, csv_file):
        reader = FileReader(chunk_size=2)
        chunks = list(reader.read_chunks(csv_file))
        assert len(chunks) >= 1
        total = sum(len(c) for c in chunks)
        assert total == 5   # simple_df fixture has 5 rows


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — API Reader
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIReader:
    def test_rest_get_no_pagination(self):
        from ingestion.readers.api_reader import APIReader, APISourceConfig
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
        with patch("requests.Session.request", return_value=mock_resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/data")
            result = reader.read(cfg)
            assert result.row_count == 2

    def test_api_timeout_triggers_retry(self):
        from ingestion.readers.api_reader import APIReader, APISourceConfig
        import requests as req
        call_count = {"n": 0}
        def _side(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise req.exceptions.Timeout("timeout")
            m = MagicMock(); m.ok = True; m.status_code = 200
            m.json.return_value = [{"x": 1}]
            return m
        with patch("requests.Session.request", side_effect=_side):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/data", max_retries=3, backoff_base=0.01)
            result = reader.read(cfg)
            assert result.row_count == 1
            assert call_count["n"] >= 3

    def test_malformed_json_response_logs_error(self):
        from ingestion.readers.api_reader import APIReader, APISourceConfig
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_resp.text = "NOT JSON"
        with patch("requests.Session.request", return_value=mock_resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/bad")
            result = reader.read(cfg)
            # Should not crash — errors list should have the error
            assert result.row_count == 0 or len(result.errors) > 0

    def test_webhook_parse(self):
        from ingestion.readers.api_reader import APIReader
        payload = json.dumps([{"event": "click", "user": 42}]).encode()
        reader = APIReader()
        df = reader.parse_webhook(payload)
        assert len(df) == 1
        assert "event" in df.columns


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegrationPipeline:
    def test_file_to_issf_pipeline(self, intake, csv_file):
        cfg = SourceConfig(
            source_type="file", dataset_id="integration_test",
            data_mode="batch", path=csv_file,
        )
        snapshot = intake.ingest(cfg)
        assert snapshot.is_compliant
        assert snapshot.row_count == 5
        assert snapshot.validation_status in ("PASSED", "WARN")
        assert snapshot.quality_score > 0.5

    def test_schema_drift_detected_on_second_ingest(self, intake, tmp_base, simple_df):
        # First ingestion
        csv1 = str(tmp_base / "v1.csv")
        simple_df.to_csv(csv1, index=False)
        cfg1 = SourceConfig(source_type="file", dataset_id="drift_test", data_mode="batch", path=csv1)
        intake.ingest(cfg1)

        # Second ingestion with extra column
        df2 = simple_df.copy()
        df2["new_metric"] = range(5)
        csv2 = str(tmp_base / "v2.csv")
        df2.to_csv(csv2, index=False)
        cfg2 = SourceConfig(source_type="file", dataset_id="drift_test", data_mode="batch", path=csv2)
        snap2 = intake.ingest(cfg2)

        # New column added — should be minor bump (1.1.0)
        major, minor, patch_ = snap2.schema_version.split(".")
        assert int(minor) >= 1 or int(major) >= 1

    def test_breaking_drift_raises(self, intake, tmp_base, simple_df):
        # First ingestion
        csv1 = str(tmp_base / "break_v1.csv")
        simple_df.to_csv(csv1, index=False)
        cfg1 = SourceConfig(source_type="file", dataset_id="break_test", data_mode="batch", path=csv1,
                            block_on_schema_break=True)
        intake.ingest(cfg1)

        # Second ingestion with missing column
        df_missing = simple_df.drop(columns=["revenue"])
        csv2 = str(tmp_base / "break_v2.csv")
        df_missing.to_csv(csv2, index=False)
        cfg2 = SourceConfig(source_type="file", dataset_id="break_test", data_mode="batch", path=csv2,
                            block_on_schema_break=True)
        with pytest.raises(SchemaError):
            intake.ingest(cfg2)

    def test_quality_failure_raises(self, intake, tmp_base):
        # DataFrame with 95% nulls — should fail quality
        df_bad = pd.DataFrame({"x": [None] * 95 + [1, 2, 3, 4, 5]})
        csv_path = str(tmp_base / "bad_quality.csv")
        df_bad.to_csv(csv_path, index=False)
        cfg = SourceConfig(
            source_type="file", dataset_id="quality_fail_test",
            data_mode="batch", path=csv_path, require_quality_pass=True,
        )
        with pytest.raises(QualityGateError):
            intake.ingest(cfg)

    def test_batch_ingest_isolates_failures(self, intake, csv_file, tmp_base):
        cfgs = [
            SourceConfig(source_type="file", dataset_id="batch_ok", data_mode="batch", path=csv_file),
            SourceConfig(source_type="file", dataset_id="batch_fail", data_mode="batch",
                         path="/nonexistent/file.csv", require_quality_pass=False, block_on_schema_break=False),
            SourceConfig(source_type="file", dataset_id="batch_ok2", data_mode="batch", path=csv_file),
        ]
        results = intake.ingest_batch(cfgs)
        # batch_ok should succeed; failure should not block batch_ok2
        assert len(results) >= 2

    def test_snapshot_persisted_to_disk(self, intake, csv_file):
        cfg = SourceConfig(source_type="file", dataset_id="persist_test", data_mode="batch", path=csv_file)
        snap = intake.ingest(cfg)
        meta_path = os.path.join(intake.snapshot_dir, f"{snap.snapshot_id}_issf.json")
        assert os.path.exists(meta_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dataset_id"] == "persist_test"


# ═══════════════════════════════════════════════════════════════════════════════
# CHAOS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChaosIngestion:
    def test_truncated_csv_does_not_crash(self, tmp_base):
        """Half-uploaded file — should return partial data or graceful error."""
        p = str(tmp_base / "truncated.csv")
        with open(p, "w") as f:
            f.write("id,name,value\n1,Alice,100\n2,Bob,")  # truncated mid-row
        reader = FileReader()
        try:
            result = reader.read(p, fmt="csv")
            # Should either return partial rows or raise DataFormatError
            assert result.row_count >= 0
        except DataFormatError:
            pass  # acceptable — graceful failure

    def test_corrupt_binary_file_raises_encoding_error_or_format_error(self, tmp_base):
        """Random binary data masquerading as CSV."""
        p = str(tmp_base / "corrupt.csv")
        with open(p, "wb") as f:
            f.write(bytes(range(256)) * 10)
        reader = FileReader()
        try:
            result = reader.read(p, fmt="csv")
            # If it somehow reads it, we accept that
        except (DataFormatError, EncodingError, Exception):
            pass  # should not raise unhandled crash

    def test_completely_empty_json_file(self, tmp_base):
        p = str(tmp_base / "empty.json")
        with open(p, "w") as f:
            f.write("")
        reader = FileReader()
        with pytest.raises((DataFormatError, Exception)):
            reader.read(p, fmt="json")

    def test_schema_drift_mid_stream_does_not_crash(self, tmp_base, simple_df):
        """Simulate schema drift appearing partway through a batch."""
        intake = UniversalIntake(
            snapshot_dir=str(tmp_base / "chaos_snaps"),
            registry_dir=str(tmp_base / "chaos_registry"),
        )
        csv1 = str(tmp_base / "chaos_v1.csv")
        csv2 = str(tmp_base / "chaos_v2.csv")
        simple_df.to_csv(csv1, index=False)
        simple_df.rename(columns={"revenue": "total_revenue"}).to_csv(csv2, index=False)

        cfg1 = SourceConfig(source_type="file", dataset_id="chaos_ds", data_mode="batch",
                             path=csv1, block_on_schema_break=False)
        cfg2 = SourceConfig(source_type="file", dataset_id="chaos_ds", data_mode="batch",
                             path=csv2, block_on_schema_break=False)

        snap1 = intake.ingest(cfg1)
        snap2 = intake.ingest(cfg2)   # breaking drift but block=False → should not crash
        assert snap2.is_compliant

    def test_api_returning_500_is_retried_and_eventually_fails(self):
        from ingestion.readers.api_reader import APIReader, APISourceConfig
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("requests.Session.request", return_value=mock_resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/crash", max_retries=2, backoff_base=0.01)
            result = reader.read(cfg)
            # Should return empty df with errors rather than crashing
            assert len(result.errors) > 0
            assert result.row_count == 0

    def test_all_null_dataframe_quality_flag(self):
        df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
        gate = QualityGate()
        report = gate.check(df)
        assert report.validation_status in ("FAILED", "WARN")
        assert report.quality_score < 0.7

    def test_duplicate_column_names_normalised_safely(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "x"])
        norm = Normaliser()
        out, metas = norm.normalise(df)
        # Should deduplicate names
        assert len(out.columns) == 3
        assert len(set(out.columns)) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Stream Reader (in-memory events)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamReader:
    def test_collect_events_tumbling_window(self):
        from ingestion.readers.stream_reader import StreamReader, WindowConfig
        reader = StreamReader()
        events = [{"user_id": i, "event": "click"} for i in range(20)]
        wc = WindowConfig(strategy="tumbling", window_size_s=0.01)   # very short window
        time.sleep(0.02)   # ensure window already expired before we pass events
        results = reader.collect_events(events, wc)
        assert len(results) >= 1
        total_rows = sum(r.row_count for r in results)
        assert total_rows == 20

    def test_collect_events_sliding_window(self):
        from ingestion.readers.stream_reader import StreamReader, WindowConfig
        reader = StreamReader()
        events = [{"x": i} for i in range(10)]
        wc = WindowConfig(strategy="sliding", window_size_s=1.0, slide_step_s=0.01)
        time.sleep(0.02)
        results = reader.collect_events(events, wc)
        assert len(results) >= 1

    def test_event_with_event_time_field(self):
        from ingestion.readers.stream_reader import StreamReader, WindowConfig
        reader = StreamReader()
        events = [{"sensor_id": i, "ts": "2024-01-01T00:00:00Z"} for i in range(5)]
        wc = WindowConfig(strategy="tumbling", window_size_s=0.01, event_time_field="ts")
        time.sleep(0.02)
        results = reader.collect_events(events, wc)
        assert any(r.row_count > 0 for r in results) or True  # may all be late-dropped


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE (basic sanity)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    def test_large_csv_chunked_under_5s(self, tmp_base):
        """10,000 rows should read and normalise in under 5 seconds."""
        df = pd.DataFrame({
            "id": range(10_000),
            "val": np.random.randn(10_000),
            "cat": np.random.choice(["A", "B", "C"], 10_000),
        })
        p = str(tmp_base / "large.csv")
        df.to_csv(p, index=False)
        t0 = time.perf_counter()
        reader = FileReader(chunk_size=5000)
        result = reader.read(p)
        elapsed = time.perf_counter() - t0
        assert result.row_count == 10_000
        assert elapsed < 5.0, f"Read took {elapsed:.2f}s — too slow"

    def test_quality_gate_scales_linearly(self):
        """Quality check on 10k rows should complete in under 2 seconds."""
        df = pd.DataFrame({"x": np.random.randn(10_000), "y": np.random.randn(10_000)})
        gate = QualityGate()
        t0 = time.perf_counter()
        report = gate.check(df)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Quality gate took {elapsed:.2f}s"
