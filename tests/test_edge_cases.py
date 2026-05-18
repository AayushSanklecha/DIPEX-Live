"""
tests/test_edge_cases.py
--------------------------
Edge case and adversarial input tests.

Tests the system's resilience against:
  - Empty, oversized, and malformed datasets
  - SQL injection-like strings in query params
  - Unicode edge cases
  - Very large numeric values
  - Missing required fields in API payloads
  - Concurrent / performance smoke tests
"""
from __future__ import annotations

import io
import os
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# client fixture is provided by tests/conftest.py (auth-aware)

# ── API Edge Cases ─────────────────────────────────────────────────────────────

class TestAPIEdgeCases:

    def test_get_with_huge_limit(self, client):
        """limit=99999 should not crash the server."""
        r = client.get("/api/results?limit=99999")
        assert r.status_code not in (500, 503)

    def test_get_with_zero_limit(self, client):
        r = client.get("/api/results?limit=0")
        assert r.status_code not in (500, 503)

    def test_run_id_with_special_chars(self, client):
        """Path segment with special chars should not 500."""
        r = client.get("/api/results/" + "'; DROP TABLE pipeline_runs; --")
        assert r.status_code != 500

    def test_run_id_with_unicode(self, client):
        r = client.get("/api/results/résultat-éàü-测试-001")
        assert r.status_code != 500

    def test_run_id_extremely_long(self, client):
        r = client.get("/api/results/" + "a" * 512)
        assert r.status_code != 500

    def test_pipeline_run_with_null_values(self, client):
        r = client.post("/api/pipeline/simple-run", json={
            "dataset_id": None,
            "source_kind": None,
        })
        assert r.status_code in (400, 422), f"Expected validation error, got {r.status_code}"

    def test_pipeline_run_extra_unknown_fields(self, client):
        """Extra unknown fields in the request body should not cause 500."""
        r = client.post("/api/pipeline/simple-run", json={
            "dataset_id": "test",
            "source_kind": "file",
            "unknown_field_xyz": "should_be_ignored",
            "another_unknown": 12345,
        })
        assert r.status_code != 500

    def test_ingest_with_single_row(self, client):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        csv_bytes = df.to_csv(index=False).encode()
        r = client.post(
            "/ingest/file",
            files={"file": ("single.csv", csv_bytes, "text/csv")}
        )
        assert r.status_code != 500

    def test_ingest_very_wide_csv(self, client):
        """CSV with 500 columns should either succeed or fail gracefully."""
        df = pd.DataFrame({f"col_{i}": [1, 2, 3] for i in range(500)})
        csv_bytes = df.to_csv(index=False).encode()
        r = client.post(
            "/ingest/file",
            files={"file": ("wide.csv", csv_bytes, "text/csv")}
        )
        assert r.status_code != 500


# ── DataFrame Edge Cases (unit level) ──────────────────────────────────────────

class TestDataFrameEdgeCases:

    def test_empty_dataframe_describe(self):
        df = pd.DataFrame({"a": []})
        result = df.describe()
        assert result.at["count", "a"] == 0

    def test_all_null_column(self):
        df = pd.DataFrame({"a": [None, None, None]})
        mean = df["a"].mean()
        # Should return NaN, not throw
        assert np.isnan(mean)

    def test_constant_column_std(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
        std = df["a"].std()
        assert std == 0.0

    def test_single_value_column(self):
        df = pd.DataFrame({"a": [42]})
        assert df["a"].mean() == 42
        assert df["a"].min() == 42
        assert df["a"].max() == 42

    def test_very_large_numeric_values(self):
        df = pd.DataFrame({"a": [1e300, 2e300, 3e300]})
        # Should not overflow Python
        assert df["a"].sum() > 0

    def test_negative_values(self):
        df = pd.DataFrame({"a": [-100, -50, 0, 50, 100]})
        assert df["a"].mean() == 0
        assert df["a"].min() == -100
        assert df["a"].max() == 100

    def test_mixed_types_coercion(self):
        df = pd.DataFrame({"a": [1, "2", 3.0]})
        coerced = pd.to_numeric(df["a"], errors="coerce")
        assert list(coerced) == [1.0, 2.0, 3.0]

    def test_unicode_column_names(self):
        df = pd.DataFrame({"年齢": [25, 30], "収入": [50000, 60000]})
        assert "年齢" in df.columns
        assert "収入" in df.columns

    def test_column_name_with_spaces(self):
        df = pd.DataFrame({"column name": [1, 2, 3]})
        assert "column name" in df.columns
        assert df["column name"].sum() == 6

    def test_duplicate_column_names_handled(self):
        """DataFrames with duplicate columns should not crash basic operations."""
        data = io.StringIO("a,a,b\n1,2,3\n4,5,6")
        df = pd.read_csv(data)
        # Just verify it can be read
        assert len(df.columns) == 3


# ── Security-related Edge Cases ───────────────────────────────────────────────

class TestSecurityEdgeCases:

    def test_xss_in_dataset_id(self, client):
        """XSS strings in path params should not return HTML with scripts."""
        r = client.get("/api/results/<script>alert(1)</script>")
        assert r.status_code != 500
        if r.headers.get("content-type", "").startswith("application/json"):
            # JSON response is fine
            pass
        else:
            assert "<script>" not in r.text

    def test_path_traversal_in_dataset_id(self, client):
        """../../etc/passwd in path should be rejected or handled."""
        r = client.get("/api/results/../../etc/passwd")
        assert r.status_code in (404, 400, 422), (
            f"Path traversal not blocked! Status: {r.status_code}"
        )

    def test_very_large_payload(self, client):
        """Huge JSON body should not OOM the server."""
        payload = {"dataset_id": "x" * 10_000, "source_kind": "file"}
        r = client.post("/api/pipeline/simple-run", json=payload)
        assert r.status_code != 500

    def test_content_type_validation(self, client):
        """Sending non-JSON content type to a JSON endpoint should be handled."""
        r = client.post(
            "/api/pipeline/simple-run",
            content=b"not json",
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code not in (500, 503)
