# tests/conftest.py
"""
Shared pytest fixtures for DIPEX test suite.
Provides auth-aware TestClient that passes X-API-Key header automatically.

Sprint 4 — Issue 1: fixes 18 test regressions caused by Sprint 3's
APIKeyMiddleware being added without updating existing test fixtures.
"""
import os
import pytest
from fastapi.testclient import TestClient

# Test API key — matches what test_api_auth.py uses
TEST_API_KEY = "dipex-test-key-sprint4"

@pytest.fixture(autouse=True, scope="function")
def _clear_data_uploads():
    """
    Ensure data/uploads/ is cleared before and after each test
    to prevent test-state pollution.
    """
    import shutil
    import glob
    for f in glob.glob("data/uploads/*.csv"):
        try: os.remove(f)
        except OSError: pass
    yield
    for f in glob.glob("data/uploads/*.csv"):
        try: os.remove(f)
        except OSError: pass


@pytest.fixture(autouse=True)
def _set_test_api_key(monkeypatch):
    """
    Ensure DIPEX_API_KEY is set to a known test value for ALL tests.
    This prevents auth middleware from blocking requests during testing.
    """
    monkeypatch.setenv("DIPEX_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("DIPEX_ENV", "development")


@pytest.fixture(scope="function")
def client():
    """
    Auth-aware FastAPI TestClient.
    Automatically includes X-API-Key header on every request.
    Replaces the per-file TestClient fixtures that lacked auth headers.
    """
    # Import inside fixture to ensure env vars are set first
    import importlib
    import api.middleware.auth as auth_mod
    importlib.reload(auth_mod)
    import api.app as app_mod
    importlib.reload(app_mod)

    class AuthTestClient(TestClient):
        """TestClient that automatically injects the API key header."""
        def request(self, method, url, **kwargs):
            headers = kwargs.pop("headers", {}) or {}
            headers.setdefault("X-API-Key", TEST_API_KEY)
            return super().request(method, url, headers=headers, **kwargs)

    with AuthTestClient(app_mod.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="function")
def client_no_auth():
    """
    TestClient WITHOUT auth headers.
    Use only when explicitly testing that unauthenticated requests fail.
    """
    import importlib
    import api.middleware.auth as auth_mod
    importlib.reload(auth_mod)
    import api.app as app_mod
    importlib.reload(app_mod)
    with TestClient(app_mod.app, raise_server_exceptions=False) as c:
        yield c
