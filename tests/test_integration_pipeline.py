"""
tests/test_integration_pipeline.py
------------------------------------
Integration tests for the core pipeline API routes:
  - POST /api/pipeline/simple-run (main entry point from frontend)
  - POST /ingest/file (file upload)
  - GET /api/results  (result listing)
  - GET /api/results/latest (last run)
  - GET /api/audit/  (audit trail)
  - GET /api/export/* (export endpoints)

Tests use FastAPI's TestClient (no live Docker required).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import uuid

import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# client fixture is provided by tests/conftest.py (auth-aware)


@pytest.fixture(scope="module")
def sample_csv_bytes():
    """Generate a minimal CSV for testing file uploads."""
    df = pd.DataFrame({
        "age":    [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 190000, 210000],
        "label":  [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── File Upload (Ingest) ───────────────────────────────────────────────────────

class TestIngestFile:

    def test_ingest_file_valid_csv(self, client, sample_csv_bytes):
        r = client.post(
            "/ingest/file",
            files={"file": ("test_data.csv", sample_csv_bytes, "text/csv")}
        )
        assert r.status_code in (200, 201), f"Expected 200/201, got {r.status_code}: {r.text}"

    def test_ingest_file_no_file_422(self, client):
        r = client.post("/ingest/file")
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"

    def test_ingest_file_wrong_type(self, client):
        r = client.post(
            "/ingest/file",
            files={"file": ("test.exe", b"binary garbage", "application/octet-stream")}
        )
        # Should reject or handle gracefully — not 500
        assert r.status_code != 500, "Server error on invalid file type"

    def test_ingest_file_empty_csv(self, client):
        r = client.post(
            "/ingest/file",
            files={"file": ("empty.csv", b"", "text/csv")}
        )
        assert r.status_code != 500, "Server error on empty file"

    def test_ingest_file_returns_dataset_id(self, client, sample_csv_bytes):
        r = client.post(
            "/ingest/file",
            files={"file": ("data.csv", sample_csv_bytes, "text/csv")}
        )
        if r.status_code in (200, 201):
            data = r.json()
            # Should have some kind of dataset identifier
            assert any(k in data for k in ("dataset_id", "id", "file_id", "filename")), (
                f"No dataset identifier in response: {data}"
            )


# ── Pipeline Simple Run ────────────────────────────────────────────────────────

class TestPipelineSimpleRun:

    def test_pipeline_no_body_422(self, client):
        r = client.post("/api/pipeline/simple-run", json={})
        assert r.status_code in (400, 422), f"Unexpected: {r.status_code}"

    def test_pipeline_nonexistent_dataset_graceful(self, client):
        r = client.post("/api/pipeline/simple-run", json={
            "dataset_id": "does_not_exist_" + str(uuid.uuid4()),
            "source_kind": "file",
        })
        # Should fail gracefully — not 500
        assert r.status_code != 500, f"Server crashed: {r.text}"

    def test_pipeline_invalid_source_kind(self, client):
        r = client.post("/api/pipeline/simple-run", json={
            "dataset_id": "some_id",
            "source_kind": "invalid_kind_xyz",
        })
        assert r.status_code != 500, "Server crashed on invalid source_kind"

    def test_pipeline_missing_dataset_id(self, client):
        r = client.post("/api/pipeline/simple-run", json={"source_kind": "file"})
        assert r.status_code in (400, 422), "Expected validation error for missing dataset_id"


# ── Results API ───────────────────────────────────────────────────────────────

class TestResultsAPI:

    def test_list_results_200(self, client):
        r = client.get("/api/results")
        assert r.status_code == 200

    def test_list_results_schema(self, client):
        data = client.get("/api/results").json()
        assert "runs" in data
        assert "total" in data
        assert isinstance(data["runs"], list)
        assert isinstance(data["total"], int)

    def test_list_results_limit_param(self, client):
        r = client.get("/api/results?limit=3")
        assert r.status_code == 200
        assert len(r.json()["runs"]) <= 3

    def test_list_results_negative_limit(self, client):
        r = client.get("/api/results?limit=-1")
        # Should not 500
        assert r.status_code != 500

    def test_results_latest_404_when_empty(self, client):
        """When no runs exist, /latest should return 404 not 500."""
        r = client.get("/api/results/latest")
        assert r.status_code in (200, 404), f"Unexpected: {r.status_code}"

    def test_results_by_id_nonexistent(self, client):
        r = client.get("/api/results/run_nonexistent_001")
        assert r.status_code in (200, 404), f"Unexpected: {r.status_code}"

    def test_results_by_id_schema_when_404(self, client):
        """404 response should still be JSON, not HTML."""
        r = client.get("/api/results/nonexistent_xyz_123")
        if r.status_code == 404:
            data = r.json()
            assert "detail" in data or "run_id" in data


# ── Audit API ─────────────────────────────────────────────────────────────────

class TestAuditAPI:

    def test_audit_returns_200_or_404(self, client):
        r = client.get("/api/audit/")
        assert r.status_code in (200, 404)

    def test_audit_not_500(self, client):
        r = client.get("/api/audit/")
        assert r.status_code != 500


# ── Stats/EDA API ─────────────────────────────────────────────────────────────

class TestStatsAPI:

    def test_stats_describe_no_body(self, client):
        r = client.post("/stats/describe", json={})
        assert r.status_code in (200, 400, 404, 422)

    def test_stats_describe_not_500(self, client):
        r = client.post("/stats/describe", json={})
        assert r.status_code != 500

    def test_stats_with_fake_dataset(self, client):
        r = client.post("/stats/describe", json={"dataset_id": "fake_id"})
        assert r.status_code != 500

    def test_stats_profiling_no_body(self, client):
        r = client.post("/stats/profile", json={})
        # Should either work or return structured error
        assert r.status_code not in (500, 503)


# ── Preprocess API ────────────────────────────────────────────────────────────

class TestPreprocessAPI:

    def test_preprocess_no_body(self, client):
        r = client.post("/preprocess/run", json={})
        assert r.status_code in (200, 400, 404, 422)

    def test_preprocess_not_500(self, client):
        r = client.post("/preprocess/run", json={})
        assert r.status_code != 500


# ── Analyst API ──────────────────────────────────────────────────────────────

class TestAnalystAPI:

    def test_analyst_run_not_500(self, client):
        r = client.post("/analyst/run", json={"dataset_id": "fake_test_id"})
        assert r.status_code != 500
