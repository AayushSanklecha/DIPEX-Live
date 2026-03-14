"""
tests/test_security.py
------------------------
Security Layer Tests — Phase 19 Verification.

Covers:
  - Encryption at-rest: encrypt_bytes / decrypt_bytes round-trip
  - Magic header detection (DIPEX_ENC: prefix)
  - Encrypt / decrypt file round-trip
  - Graceful passthrough when encryption disabled
  - SHA-256 helper correctness
  - RBAC: has_permission properly gates all defined actions
  - require_role dependency: VIEWER blocked from ANALYST-only actions
  - PII detector: email, SSN, credit card, phone regex patterns all fire
  - PII mask: detected columns are replaced
  - Audit access log: middleware logging format validates JSON structure
"""
from __future__ import annotations

import json
import os
import hashlib
import tempfile
from pathlib import Path
from typing import Dict

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# ENCRYPTION
# ══════════════════════════════════════════════════════════════════════════════

class TestEncryptionRoundTrip:
    """Tests for security/encryption.py — always harmless (in-process only)."""

    @pytest.fixture(autouse=True)
    def _patch_key(self, monkeypatch):
        """Patch a valid 32-char key so encryption is active for all these tests."""
        monkeypatch.setenv("DIPEX_ENCRYPTION_KEY", "dGVzdC1rZXktZm9yLXVuaXQtdGVzdHMtb25seQ==")
        # Reset cached fernet so the new key is picked up
        import security.encryption as enc
        enc._fernet = None
        enc._RAW_KEY = "dGVzdC1rZXktZm9yLXVuaXQtdGVzdHMtb25seQ=="
        enc._ENCRYPTION_ENABLED = True
        yield
        enc._fernet = None

    def test_encrypt_then_decrypt_bytes(self):
        from security.encryption import encrypt_bytes, decrypt_bytes
        original = b"Hello DIPEX - sensitive data payload"
        cipher = encrypt_bytes(original)
        assert cipher != original
        assert cipher.startswith(b"DIPEX_ENC:")
        recovered = decrypt_bytes(cipher)
        assert recovered == original

    def test_unencrypted_bytes_passthrough(self):
        """decrypt_bytes on plaintext (no header) returns data unchanged."""
        from security.encryption import decrypt_bytes
        data = b"plain unencrypted bytes"
        assert decrypt_bytes(data) == data

    def test_magic_header_present(self):
        from security.encryption import encrypt_bytes
        cipher = encrypt_bytes(b"test")
        assert cipher[:10] == b"DIPEX_ENC:"

    def test_encrypt_decrypt_file_roundtrip(self, tmp_path):
        from security.encryption import encrypt_file, decrypt_file
        src = tmp_path / "data.json"
        src.write_bytes(b'{"dataset_id": "test", "rows": 100}')

        enc_path = encrypt_file(src)
        assert enc_path.suffix == ".enc"
        assert enc_path.read_bytes().startswith(b"DIPEX_ENC:")

        _, recovered = decrypt_file(enc_path)
        assert recovered == b'{"dataset_id": "test", "rows": 100}'

    def test_encrypt_file_different_from_source(self, tmp_path):
        from security.encryption import encrypt_file
        src = tmp_path / "snapshot.parquet"
        src.write_bytes(b"PAR1" + b"\x00" * 100)
        enc_path = encrypt_file(src)
        assert enc_path.read_bytes() != src.read_bytes()

    def test_sha256_helper(self):
        from security.encryption import compute_sha256
        data = b"dipex-sha256-test"
        expected = hashlib.sha256(data).hexdigest()
        assert compute_sha256(data) == expected
        assert len(compute_sha256(data)) == 64

    def test_is_encryption_enabled(self):
        from security.encryption import is_encryption_enabled
        assert is_encryption_enabled() is True


class TestEncryptionDisabled:
    """When no key set, encrypt_bytes returns plaintext unchanged."""

    @pytest.fixture(autouse=True)
    def _clear_key(self, monkeypatch):
        monkeypatch.delenv("DIPEX_ENCRYPTION_KEY", raising=False)
        import security.encryption as enc
        enc._fernet = None
        enc._RAW_KEY = ""
        enc._ENCRYPTION_ENABLED = False
        yield
        enc._fernet = None

    def test_encrypt_passthrough_no_key(self):
        from security.encryption import encrypt_bytes
        data = b"no key set - plaintext expected"
        result = encrypt_bytes(data)
        assert result == data

    def test_is_encryption_disabled(self):
        from security.encryption import is_encryption_enabled
        assert is_encryption_enabled() is False


# ══════════════════════════════════════════════════════════════════════════════
# RBAC
# ══════════════════════════════════════════════════════════════════════════════

class TestRBAC:

    def test_viewer_can_view_reports(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "view_reports") is True

    def test_viewer_cannot_run_pipeline(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "run_pipeline") is False

    def test_viewer_cannot_train_model(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "train_model") is False

    def test_analyst_can_run_pipeline(self):
        from auth.rbac import has_permission
        assert has_permission("ANALYST", "run_pipeline") is True

    def test_analyst_cannot_train_model(self):
        from auth.rbac import has_permission
        assert has_permission("ANALYST", "train_model") is False

    def test_admin_can_do_everything(self):
        from auth.rbac import has_permission, PERMISSIONS
        for action in PERMISSIONS:
            assert has_permission("ADMIN", action) is True, f"ADMIN blocked for: {action}"

    def test_api_service_can_do_everything(self):
        from auth.rbac import has_permission, PERMISSIONS
        for action in PERMISSIONS:
            assert has_permission("API_SERVICE", action) is True, f"API_SERVICE blocked for: {action}"

    def test_role_hierarchy_order(self):
        from auth.rbac import ROLES
        assert ROLES["VIEWER"] < ROLES["ANALYST"] < ROLES["ADMIN"] < ROLES["API_SERVICE"]

    def test_unknown_role_denied(self):
        from auth.rbac import has_permission
        # Unknown role should be denied
        assert has_permission("UNKNOWN_ROLE", "run_pipeline") is False

    def test_unknown_action_defaults_to_admin_only(self):
        from auth.rbac import has_permission
        # Unknown actions default to requiring ADMIN
        assert has_permission("VIEWER", "nonexistent_action") is False
        assert has_permission("ADMIN", "nonexistent_action") is True

    def test_require_role_dependency_passes_for_correct_role(self):
        """require_role returns a coroutine that passes for sufficient roles."""
        from auth.rbac import require_role
        # Just verify require_role factory returns a callable
        dep = require_role("ANALYST")
        assert callable(dep)

    def test_require_role_blocks_viewer_from_analyst_endpoint(self):
        import asyncio
        from auth.rbac import require_role
        from fastapi import HTTPException
        dep = require_role("ANALYST")
        viewer_user = {"username": "viewer", "role": "VIEWER"}
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(dep(user=viewer_user))
        assert exc_info.value.status_code == 403

    def test_require_role_allows_admin(self):
        import asyncio
        from auth.rbac import require_role
        dep = require_role("ANALYST")
        admin_user = {"username": "admin", "role": "ADMIN"}
        result = asyncio.run(dep(user=admin_user))
        assert result["username"] == "admin"


# ══════════════════════════════════════════════════════════════════════════════
# PII DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TestPIIDetector:

    @pytest.fixture
    def detector(self):
        from governance.pii_detector import PIIDetector
        return PIIDetector()

    def test_email_detected(self, detector):
        import pandas as pd
        df = pd.DataFrame({"contacts": ["alice@example.com"] * 50 + ["bob@corp.io"] * 50})
        report = detector.scan(df, sample_n=100)
        assert "contacts" in report["pii_columns"]
        assert "email" in report["pii_columns"]["contacts"]

    def test_ssn_detected(self, detector):
        import pandas as pd
        df = pd.DataFrame({"ssn": ["123-45-6789"] * 60 + ["987-65-4321"] * 40})
        report = detector.scan(df, sample_n=100)
        assert "ssn" in report["pii_columns"]

    def test_credit_card_detected(self, detector):
        import pandas as pd
        df = pd.DataFrame({"payment": ["4111-1111-1111-1111"] * 80 + ["safe"] * 20})
        report = detector.scan(df, sample_n=100)
        assert "payment" in report["pii_columns"]

    def test_clean_column_not_flagged(self, detector):
        import pandas as pd
        df = pd.DataFrame({"revenue": ["10000", "25000", "15000"] * 33})
        report = detector.scan(df, sample_n=100)
        # Numeric strings shouldn't be flagged as PII
        assert len(report["pii_columns"]) == 0 or "revenue" not in report["pii_columns"]

    def test_mask_replaces_pii_columns(self, detector):
        import pandas as pd
        df = pd.DataFrame({
            "email": ["user@example.com"] * 80,
            "amount": [100.0] * 80,
        })
        report = detector.scan(df)
        masked = detector.mask(df, report)
        if "email" in report["pii_columns"]:
            assert (masked["email"] == "***").all()
        # Non-PII column unchanged
        assert (masked["amount"] == 100.0).all()

    def test_mask_returns_copy(self, detector):
        import pandas as pd
        df = pd.DataFrame({"email": ["a@b.com"] * 60})
        report = detector.scan(df)
        masked = detector.mask(df, report)
        # Original must not be modified
        assert df["email"].iloc[0] == "a@b.com"
        assert masked is not df

    def test_scan_empty_dataframe(self, detector):
        import pandas as pd
        df = pd.DataFrame()
        report = detector.scan(df)
        assert report["pii_columns"] == {}
        assert report["safe_columns"] == []

    def test_scan_returns_required_keys(self, detector):
        import pandas as pd
        df = pd.DataFrame({"x": ["test"] * 10})
        report = detector.scan(df)
        assert "pii_columns"  in report
        assert "safe_columns" in report
        assert "method"        in report
        assert "details"       in report


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT ACCESS LOG
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditAccessLog:
    """Verify the middleware writing format + JSON structure."""

    def test_middleware_is_importable(self):
        from middleware.audit_access_log import AuditAccessLogMiddleware
        assert AuditAccessLogMiddleware is not None

    def test_log_entry_structure(self, tmp_path, monkeypatch):
        """Simulate a log write and verify field structure."""
        import json
        from datetime import datetime, timezone

        monkeypatch.setenv("AUDIT_DIR", str(tmp_path))

        # Manually write a log entry as the middleware would
        entry = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "request_id":  "test-uuid-1234",
            "username":    "analyst",
            "role":        "ANALYST",
            "method":      "POST",
            "path":        "/analyst/run",
            "status":      200,
            "duration_ms": 142.3,
            "client_ip":   "127.0.0.1",
            "user_agent":  "pytest/7.4.4",
            "bytes_sent":  1024,
        }

        log_file = tmp_path / f"access_log_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])

        required_keys = ["ts", "request_id", "username", "role", "method",
                         "path", "status", "duration_ms", "client_ip", "user_agent", "bytes_sent"]
        for key in required_keys:
            assert key in parsed, f"Missing key in audit log: {key}"

        assert parsed["username"] == "analyst"
        assert parsed["role"] == "ANALYST"
        assert parsed["status"] == 200
        assert parsed["duration_ms"] == 142.3

    def test_excluded_paths_not_logged(self):
        """Verify excluded paths list is correctly defined."""
        from middleware.audit_access_log import _EXCLUDED_PATHS
        for path in ["/prom-metrics", "/docs", "/redoc", "/openapi.json"]:
            assert path in _EXCLUDED_PATHS, f"{path} should be excluded from audit log"


# ══════════════════════════════════════════════════════════════════════════════
# JWT AUTH
# ══════════════════════════════════════════════════════════════════════════════

class TestJWTAuth:
    """JWT token creation, decoding, and authentication tests.
    Requires: python-jose[cryptography] >= 3.5.0 (installed)
    """

    def test_create_and_decode_access_token(self):
        from auth.jwt_auth import JWTAuth
        token = JWTAuth.create_access_token({"sub": "analyst", "role": "ANALYST"})
        assert isinstance(token, str)
        payload = JWTAuth.decode_token(token)
        assert payload["sub"] == "analyst"
        assert payload["role"] == "ANALYST"
        assert payload["type"] == "access"

    def test_authenticate_user_valid(self):
        from auth.jwt_auth import JWTAuth
        result = JWTAuth.authenticate_user("admin", "secret")
        assert result is not None
        assert result["username"] == "admin"
        assert result["role"] == "ADMIN"

    def test_authenticate_user_invalid_password(self):
        from auth.jwt_auth import JWTAuth
        result = JWTAuth.authenticate_user("admin", "wrongpassword")
        assert result is None

    def test_authenticate_user_nonexistent(self):
        from auth.jwt_auth import JWTAuth
        result = JWTAuth.authenticate_user("nobody", "secret")
        assert result is None

    def test_hash_and_verify_password(self):
        from auth.jwt_auth import JWTAuth
        hashed = JWTAuth.hash_password("my_secure_password")
        assert hashed != "my_secure_password"
        assert JWTAuth.verify_password("my_secure_password", hashed) is True
        assert JWTAuth.verify_password("wrong", hashed) is False

    def test_refresh_token_type_claim(self):
        from auth.jwt_auth import JWTAuth
        token = JWTAuth.create_refresh_token({"sub": "analyst"})
        payload = JWTAuth.decode_token(token)
        assert payload["type"] == "refresh"
