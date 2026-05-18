# tests/test_api_auth.py
"""
API key authentication middleware tests — Sprint 3, Issue C4.
Tests: health bypass, wrong key, correct key, dev mode, missing header, production abort.
"""
import os
import importlib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_no_key(monkeypatch):
    """Client when no API key is configured (dev mode — auth disabled)."""
    monkeypatch.delenv("DIPEX_API_KEY", raising=False)
    monkeypatch.setenv("DIPEX_ENV", "development")
    # Force re-import to pick up env changes
    import api.middleware.auth as auth_mod
    importlib.reload(auth_mod)
    import api.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


@pytest.fixture
def client_with_key(monkeypatch):
    """Client when API key IS configured."""
    monkeypatch.setenv("DIPEX_API_KEY", "test-sprint3-key")
    monkeypatch.setenv("DIPEX_ENV", "development")
    import api.middleware.auth as auth_mod
    importlib.reload(auth_mod)
    import api.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_health_exempt_no_auth_needed(client_with_key):
    """Health endpoint must be accessible without any auth key."""
    response = client_with_key.get("/health")
    assert response.status_code == 200


def test_wrong_key_returns_401(client_with_key):
    """Wrong API key must return 401."""
    response = client_with_key.get("/metrics", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["error"]


def test_correct_key_returns_200(client_with_key):
    """Correct API key must allow through."""
    response = client_with_key.get("/metrics", headers={"X-API-Key": "test-sprint3-key"})
    assert response.status_code == 200


def test_dev_mode_no_key_allows_all(client_no_key):
    """In dev mode with no key set, all requests pass through."""
    response = client_no_key.get("/metrics")
    assert response.status_code == 200


def test_missing_key_header_returns_401(client_with_key):
    """Request with no X-API-Key header must return 401 when key is configured."""
    response = client_with_key.get("/metrics")
    assert response.status_code == 401


def test_production_aborts_without_key(monkeypatch):
    """LAW 10: production startup must abort when key is missing."""
    monkeypatch.delenv("DIPEX_API_KEY", raising=False)
    monkeypatch.setenv("DIPEX_ENV", "production")
    import api.middleware.auth as auth_mod
    importlib.reload(auth_mod)
    with pytest.raises(EnvironmentError, match="DIPEX_API_KEY"):
        class FakeApp:
            pass
        auth_mod.APIKeyMiddleware(FakeApp())
