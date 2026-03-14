"""
tests/test_health_and_system.py
--------------------------------
Unit + Integration tests for all system-level endpoints and startup checks.

Gap coverage:
  - /health must return the rich schema (status, version, uptime, db_ok, model_registry_ok, timestamp)
  - /metrics must return numeric types, not strings
  - / root must list architecture layers
  - /docs and /openapi.json must respond 200
  - All mounted route prefixes must appear in OpenAPI schema
"""
from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_status_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200, r.text

    def test_health_schema_keys(self, client):
        data = client.get("/health").json()
        required = {"status", "version", "uptime", "db_ok", "model_registry_ok", "timestamp"}
        missing = required - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_health_status_valid_values(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_uptime_non_negative(self, client):
        data = client.get("/health").json()
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] >= 0

    def test_health_db_ok_is_bool(self, client):
        data = client.get("/health").json()
        assert isinstance(data["db_ok"], bool)
        assert isinstance(data["model_registry_ok"], bool)

    def test_health_version_format(self, client):
        data = client.get("/health").json()
        assert isinstance(data["version"], str)
        parts = data["version"].split(".")
        assert len(parts) == 3, f"Expected semver, got {data['version']}"

    def test_health_timestamp_is_iso8601(self, client):
        from datetime import datetime
        data = client.get("/health").json()
        # Should parse without exception
        dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert dt is not None

    def test_health_repeated_calls_uptime_increases(self, client):
        """Uptime should increase between successive calls."""
        up1 = client.get("/health").json()["uptime"]
        time.sleep(0.1)
        up2 = client.get("/health").json()["uptime"]
        assert up2 >= up1


# ── / root ─────────────────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_200(self, client):
        assert client.get("/").status_code == 200

    def test_root_schema(self, client):
        data = client.get("/").json()
        for key in ("name", "version", "status", "docs"):
            assert key in data, f"Missing key '{key}'"

    def test_root_status_operational(self, client):
        assert client.get("/").json()["status"] == "operational"

    def test_root_architecture_is_list(self, client):
        data = client.get("/").json()
        assert "architecture" in data
        layers = data["architecture"]
        assert isinstance(layers, list)
        assert len(layers) >= 3, "Expected at least 3 architecture layers"


# ── /metrics ───────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_schema(self, client):
        data = client.get("/metrics").json()
        for key in ("total_pipeline_runs", "passed_runs", "pass_rate", "uptime_seconds"):
            assert key in data, f"Missing key '{key}'"

    def test_metrics_pass_rate_in_0_1(self, client):
        data = client.get("/metrics").json()
        pr = data["pass_rate"]
        assert 0.0 <= pr <= 1.0, f"pass_rate out of range: {pr}"

    def test_metrics_types(self, client):
        data = client.get("/metrics").json()
        assert isinstance(data["total_pipeline_runs"], int)
        assert isinstance(data["passed_runs"], int)
        assert isinstance(data["uptime_seconds"], (int, float))


# ── /docs and /openapi.json ───────────────────────────────────────────────────

class TestDocs:

    def test_docs_200(self, client):
        assert client.get("/docs").status_code == 200

    def test_openapi_200(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_openapi_schema_structure(self, client):
        data = client.get("/openapi.json").json()
        for key in ("openapi", "info", "paths"):
            assert key in data

    def test_route_coverage(self, client):
        """All core route prefixes must appear in the OpenAPI schema."""
        expected = ["/api/results", "/ingest", "/stats", "/preprocess"]
        paths = list(client.get("/openapi.json").json()["paths"].keys())
        missing = [p for p in expected if not any(rp.startswith(p) for rp in paths)]
        assert not missing, f"Route prefixes missing from OpenAPI schema: {missing}"
