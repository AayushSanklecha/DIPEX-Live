"""
tests/test_api_ingestion.py
-------------------------------
HTTP API ingestion tests — no live API required.
All HTTP requests are mocked via requests.Session.request.

Coverage:
  - Simple GET → list response → DataFrame
  - data_path extraction from nested JSON
  - Page-based pagination
  - Cursor-based pagination
  - 429 rate-limit retry
  - 500 error handling (returns errors, not crash)
  - Connection error (raises, not crash)
  - Empty response → empty DataFrame
  - Auth headers sent correctly (api_key, bearer)
  - Webhook payload parsing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.readers.api_reader import (
    APIReader, APISourceConfig, APIReadResult,
    AuthConfig, PaginationConfig, AuthProvider,
)
from ingestion.error_handler import APITimeoutError, APIResponseError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(data=None, status_code=200, ok=True, headers=None, text=""):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text or json.dumps(data or [])
    if data is not None:
        resp.json.return_value = data
    else:
        resp.json.side_effect = json.JSONDecodeError("empty", "", 0)
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# APIReader.read() Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIReader:

    def test_simple_list_response(self):
        """GET returning a JSON list → DataFrame."""
        data = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
        resp = _mock_response(data)

        with patch("requests.Session.request", return_value=resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/data")
            result = reader.read(cfg)

        assert isinstance(result, APIReadResult)
        assert result.row_count == 2
        assert list(result.data.columns) == ["id", "val"]

    def test_data_path_extraction(self):
        """data_path extracts nested records from response JSON."""
        data = {"results": {"items": [{"x": 1}, {"x": 2}, {"x": 3}]}}
        resp = _mock_response(data)

        with patch("requests.Session.request", return_value=resp):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/nested",
                data_path="results.items",
            )
            result = reader.read(cfg)

        assert result.row_count == 3
        assert "x" in result.data.columns

    def test_page_based_pagination(self):
        """Page-based pagination fetches multiple pages."""
        page1 = _mock_response([{"id": 1}, {"id": 2}])
        page2 = _mock_response([{"id": 3}])
        page3 = _mock_response([])  # empty page → stop

        responses = [page1, page2, page3]
        call_idx = {"i": 0}

        def _side(*a, **kw):
            r = responses[min(call_idx["i"], len(responses) - 1)]
            call_idx["i"] += 1
            return r

        with patch("requests.Session.request", side_effect=_side):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/paginated",
                pagination=PaginationConfig(
                    strategy="page",
                    page_size=2,
                    max_pages=5,
                ),
            )
            result = reader.read(cfg)

        assert result.row_count == 3
        assert result.pages_fetched >= 2

    def test_cursor_pagination(self):
        """Cursor-based pagination follows next_cursor."""
        page1_data = {"data": [{"id": 1}], "next_cursor": "abc123"}
        page2_data = {"data": [{"id": 2}], "next_cursor": None}

        page1 = _mock_response(page1_data)
        page2 = _mock_response(page2_data)
        responses = [page1, page2]
        call_idx = {"i": 0}

        def _side(*a, **kw):
            r = responses[min(call_idx["i"], len(responses) - 1)]
            call_idx["i"] += 1
            return r

        with patch("requests.Session.request", side_effect=_side):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/cursor",
                data_path="data",
                pagination=PaginationConfig(
                    strategy="cursor",
                    cursor_field="next_cursor",
                    max_pages=5,
                ),
            )
            result = reader.read(cfg)

        assert result.row_count == 2
        assert result.pages_fetched >= 2

    def test_rate_limit_429_retried(self):
        """429 Too Many Requests triggers retry with backoff."""
        rate_limited = _mock_response(
            None, status_code=429, ok=False,
            headers={"Retry-After": "0"},
            text="Rate limited",
        )
        success = _mock_response([{"id": 1}])

        responses = [rate_limited, success]
        call_idx = {"i": 0}

        def _side(*a, **kw):
            r = responses[min(call_idx["i"], len(responses) - 1)]
            call_idx["i"] += 1
            return r

        with patch("requests.Session.request", side_effect=_side):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/rate_limited",
                max_retries=3,
                backoff_base=0.01,
            )
            result = reader.read(cfg)

        assert result.row_count == 1
        # At least one request was rate-limited and retried
        assert call_idx["i"] >= 2

    def test_500_error_returns_errors_not_crash(self):
        """HTTP 500 error is captured in errors, not raised."""
        resp = _mock_response(
            None, status_code=500, ok=False, text="Internal Server Error"
        )

        with patch("requests.Session.request", return_value=resp):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/crash",
                max_retries=2,
                backoff_base=0.01,
            )
            result = reader.read(cfg)

        assert result.row_count == 0
        assert len(result.errors) > 0

    def test_connection_error_returns_errors(self):
        """ConnectionError is caught, not raised."""
        import requests as req

        with patch("requests.Session.request", side_effect=req.ConnectionError("DNS failed")):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/down",
                max_retries=1,
                backoff_base=0.01,
            )
            result = reader.read(cfg)

        assert result.row_count == 0
        assert len(result.errors) > 0

    def test_timeout_error_returns_errors(self):
        """Timeout is caught after retries, errors list populated."""
        import requests as req

        with patch("requests.Session.request", side_effect=req.Timeout("timed out")):
            reader = APIReader()
            cfg = APISourceConfig(
                url="https://api.example.com/slow",
                max_retries=2,
                backoff_base=0.01,
            )
            result = reader.read(cfg)

        assert result.row_count == 0
        assert len(result.errors) > 0

    def test_empty_response_returns_empty_dataframe(self):
        """Empty JSON array → empty DataFrame, not error."""
        resp = _mock_response([])

        with patch("requests.Session.request", return_value=resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/empty")
            result = reader.read(cfg)

        assert result.row_count == 0
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.empty

    def test_dict_response_single_object(self):
        """API returning a single dict (not list) is handled."""
        resp = _mock_response({"id": 99, "status": "ok"})

        with patch("requests.Session.request", return_value=resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/single")
            result = reader.read(cfg)

        # Should wrap single dict in list → 1-row DataFrame
        assert result.row_count == 1

    def test_malformed_json_handled(self):
        """Non-JSON response is caught, not raised."""
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        resp.text = "NOT JSON"
        resp.headers = {}

        with patch("requests.Session.request", return_value=resp):
            reader = APIReader()
            cfg = APISourceConfig(url="https://api.example.com/bad")
            result = reader.read(cfg)

        assert result.row_count == 0 or len(result.errors) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Auth Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthProvider:
    def test_api_key_header(self):
        """api_key strategy adds X-API-Key header."""
        auth = AuthConfig(strategy="api_key", api_key="my-secret-key")
        provider = AuthProvider(auth)
        headers = provider.get_headers()
        assert headers["X-API-Key"] == "my-secret-key"

    def test_custom_api_key_header(self):
        """api_key strategy with custom header name."""
        auth = AuthConfig(strategy="api_key", api_key="k", api_key_header="Authorization")
        provider = AuthProvider(auth)
        headers = provider.get_headers()
        assert headers["Authorization"] == "k"

    def test_bearer_token_header(self):
        """bearer strategy adds Authorization: Bearer header."""
        auth = AuthConfig(strategy="bearer", bearer_token="tok123")
        provider = AuthProvider(auth)
        headers = provider.get_headers()
        assert headers["Authorization"] == "Bearer tok123"

    def test_no_auth_returns_empty(self):
        """'none' strategy returns empty headers dict."""
        auth = AuthConfig(strategy="none")
        provider = AuthProvider(auth)
        headers = provider.get_headers()
        assert headers == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Webhook Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebhookParsing:
    def test_parse_webhook_list_payload(self):
        """parse_webhook() handles list payload."""
        reader = APIReader()
        payload = json.dumps([{"event": "click", "user": 42}]).encode()
        df = reader.parse_webhook(payload)
        assert len(df) == 1
        assert "event" in df.columns

    def test_parse_webhook_single_dict(self):
        """parse_webhook() handles single dict payload."""
        reader = APIReader()
        payload = json.dumps({"event": "signup", "user": 7}).encode()
        df = reader.parse_webhook(payload)
        assert len(df) == 1
        assert "event" in df.columns

    def test_parse_webhook_invalid_json(self):
        """parse_webhook() with invalid JSON raises DataFormatError."""
        from ingestion.error_handler import DataFormatError
        reader = APIReader()
        with pytest.raises(DataFormatError):
            reader.parse_webhook(b"NOT JSON!!!")
