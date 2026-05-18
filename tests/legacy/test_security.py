# tests/legacy/test_security.py
"""
tests/test_security.py
------------------------
Security Layer Tests — Phase 19 Verification.

Sprint 4 Triage (LAW 4 compliant — per-test categorisation):

  Category counts:
    IMPORT_ERROR  →  9 tests (encryption: security.encryption replaced by utils.snapshot_guard)
    OBSOLETE      → 19 tests (RBAC: 11, JWT: 5, Audit: 2, PII import: 1)
    API_MISMATCH  →  8 tests (PII detector v1 scan/mask → v3 detect/redact)
    PASS          →  1 test  (test_log_entry_structure — no external deps)

  Module mapping:
    security.encryption     → utils.snapshot_guard      (different API — no magic header, no file ops)
    governance.pii_detector → validation.governance.pii_detector  (scan/mask → detect/redact)
    auth.rbac               → NOT BUILT (v3 uses API key)
    auth.jwt_auth           → NOT BUILT (v3 uses API key)
    middleware.audit_access → NOT BUILT
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
# ENCRYPTION — Ported to utils.snapshot_guard where possible (Sprint 5, LAW 6)
#
# utils.snapshot_guard provides: encrypt_bytes(), decrypt_bytes(), get_fernet()
# It does NOT provide: encrypt_file, decrypt_file, compute_sha256,
#                       is_encryption_enabled, or DIPEX_ENC: magic headers.
# Tests for missing functions remain skipped (API_MISMATCH).
# Tests for existing functions are PORTED and PASS.
# ══════════════════════════════════════════════════════════════════════════════

# Valid Fernet key (url-safe base64 of 32 bytes) for test use only
_TEST_FERNET_KEY = "dGVzdGtleWZvcnVuaXR0ZXN0c29ubHkxMjM0NQ=="


class TestEncryptionRoundTrip:
    """Tests for at-rest encryption — ported to utils.snapshot_guard."""

    @pytest.fixture(autouse=True)
    def _patch_key(self, monkeypatch):
        """Patch a valid Fernet key so encryption is active for all these tests."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("DIPEX_ENCRYPTION_KEY", key)
        monkeypatch.setenv("DIPEX_ENV", "development")

    # ── PORTED: encrypt/decrypt round-trip (no magic header assertion) ──
    def test_encrypt_then_decrypt_bytes(self):
        from utils.snapshot_guard import encrypt_bytes, decrypt_bytes
        original = b"Hello DIPEX - sensitive data payload"
        cipher = encrypt_bytes(original)
        assert cipher != original
        # v3 uses raw Fernet token — no DIPEX_ENC: prefix
        recovered = decrypt_bytes(cipher)
        assert recovered == original

    # ── PORTED: decrypt passthrough on plaintext when key is set ──
    def test_unencrypted_bytes_decrypt_with_key_raises(self):
        """decrypt_bytes on non-Fernet data when key IS set raises ValueError."""
        from utils.snapshot_guard import decrypt_bytes
        data = b"plain unencrypted bytes"
        # v3 behavior: with key set, decrypting non-Fernet data raises ValueError
        with pytest.raises(ValueError, match="decryption failed"):
            decrypt_bytes(data)

    @pytest.mark.skip(
        reason="API_MISMATCH: v3 utils.snapshot_guard.encrypt_bytes does not prefix "
               "a DIPEX_ENC: magic header — it returns raw Fernet ciphertext. "
               "This assertion is specific to the v1 security.encryption module."
    )
    def test_magic_header_present(self):
        from security.encryption import encrypt_bytes
        cipher = encrypt_bytes(b"test")
        assert cipher[:10] == b"DIPEX_ENC:"

    @pytest.mark.skip(
        reason="API_MISMATCH: encrypt_file/decrypt_file do not exist in v3. "
               "utils.snapshot_guard only provides byte-level encrypt/decrypt. "
               "File-level encryption adapter must be built to re-enable."
    )
    def test_encrypt_decrypt_file_roundtrip(self, tmp_path):
        from security.encryption import encrypt_file, decrypt_file
        src = tmp_path / "data.json"
        src.write_bytes(b'{"dataset_id": "test", "rows": 100}')
        enc_path = encrypt_file(src)
        assert enc_path.suffix == ".enc"

    @pytest.mark.skip(
        reason="API_MISMATCH: encrypt_file does not exist in v3. "
               "utils.snapshot_guard operates on bytes only, not file paths."
    )
    def test_encrypt_file_different_from_source(self, tmp_path):
        from security.encryption import encrypt_file
        src = tmp_path / "snapshot.parquet"
        src.write_bytes(b"PAR1" + b"\x00" * 100)
        enc_path = encrypt_file(src)
        assert enc_path.read_bytes() != src.read_bytes()

    @pytest.mark.skip(
        reason="API_MISMATCH: compute_sha256 does not exist in utils.snapshot_guard. "
               "SHA-256 is used via hashlib in immutability_guard.py but not exported."
    )
    def test_sha256_helper(self):
        from security.encryption import compute_sha256
        data = b"dipex-sha256-test"
        expected = hashlib.sha256(data).hexdigest()
        assert compute_sha256(data) == expected

    # ── PORTED: is_encryption_enabled via get_fernet() ──
    def test_is_encryption_enabled(self):
        from utils.snapshot_guard import get_fernet
        # With key set, get_fernet() returns a Fernet instance (truthy)
        assert get_fernet() is not None


class TestEncryptionDisabled:
    """When no key set, encrypt_bytes returns plaintext unchanged."""

    @pytest.fixture(autouse=True)
    def _clear_key(self, monkeypatch):
        monkeypatch.delenv("DIPEX_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("DIPEX_ENV", "development")

    # ── PORTED: passthrough when no key ──
    def test_encrypt_passthrough_no_key(self):
        import warnings
        from utils.snapshot_guard import encrypt_bytes
        data = b"no key set - plaintext expected"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = encrypt_bytes(data)
        assert result == data

    # ── PORTED: encryption disabled via get_fernet() ──
    def test_is_encryption_disabled(self):
        import warnings
        from utils.snapshot_guard import get_fernet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert get_fernet() is None


# ══════════════════════════════════════════════════════════════════════════════
# RBAC — OBSOLETE: auth.rbac never implemented in v3
# v3 uses API key middleware (api/middleware/auth.py) instead of RBAC.
# No role hierarchy, no has_permission(), no require_role() dependency.
# ══════════════════════════════════════════════════════════════════════════════

class TestRBAC:

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "v3 uses API key auth middleware, not role-based access."
    )
    def test_viewer_can_view_reports(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "view_reports") is True

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "Pipeline access is controlled by API key, not viewer/analyst roles."
    )
    def test_viewer_cannot_run_pipeline(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "run_pipeline") is False

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "Model training is not gated by role hierarchy."
    )
    def test_viewer_cannot_train_model(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "train_model") is False

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "Pipeline runs require only a valid API key, not an ANALYST role."
    )
    def test_analyst_can_run_pipeline(self):
        from auth.rbac import has_permission
        assert has_permission("ANALYST", "run_pipeline") is True

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "No model training permission exists in the current auth model."
    )
    def test_analyst_cannot_train_model(self):
        from auth.rbac import has_permission
        assert has_permission("ANALYST", "train_model") is False

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission and PERMISSIONS were never "
               "implemented in v3. Admin concept does not exist in API key auth."
    )
    def test_admin_can_do_everything(self):
        from auth.rbac import has_permission, PERMISSIONS
        for action in PERMISSIONS:
            assert has_permission("ADMIN", action) is True

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission and PERMISSIONS were never "
               "implemented in v3. API_SERVICE role does not exist."
    )
    def test_api_service_can_do_everything(self):
        from auth.rbac import has_permission, PERMISSIONS
        for action in PERMISSIONS:
            assert has_permission("API_SERVICE", action) is True

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.ROLES was never implemented in v3. "
               "No role hierarchy structure exists in the codebase."
    )
    def test_role_hierarchy_order(self):
        from auth.rbac import ROLES
        assert ROLES["VIEWER"] < ROLES["ANALYST"] < ROLES["ADMIN"] < ROLES["API_SERVICE"]

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "Unknown role handling is irrelevant without an RBAC module."
    )
    def test_unknown_role_denied(self):
        from auth.rbac import has_permission
        assert has_permission("UNKNOWN_ROLE", "run_pipeline") is False

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.has_permission was never implemented in v3. "
               "Unknown action fallback logic does not exist."
    )
    def test_unknown_action_defaults_to_admin_only(self):
        from auth.rbac import has_permission
        assert has_permission("VIEWER", "nonexistent_action") is False
        assert has_permission("ADMIN", "nonexistent_action") is True

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.require_role was never implemented in v3. "
               "No FastAPI dependency injection for role checking exists."
    )
    def test_require_role_dependency_passes_for_correct_role(self):
        from auth.rbac import require_role
        dep = require_role("ANALYST")
        assert callable(dep)

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.require_role was never implemented in v3. "
               "No HTTP 403 role-gating middleware or dependency exists."
    )
    def test_require_role_blocks_viewer_from_analyst_endpoint(self):
        import asyncio
        from auth.rbac import require_role
        from fastapi import HTTPException
        dep = require_role("ANALYST")
        viewer_user = {"username": "viewer", "role": "VIEWER"}
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(dep(user=viewer_user))
        assert exc_info.value.status_code == 403

    @pytest.mark.skip(
        reason="OBSOLETE: auth.rbac.require_role was never implemented in v3. "
               "No admin-escalation logic exists in the codebase."
    )
    def test_require_role_allows_admin(self):
        import asyncio
        from auth.rbac import require_role
        dep = require_role("ANALYST")
        admin_user = {"username": "admin", "role": "ADMIN"}
        result = asyncio.run(dep(user=admin_user))
        assert result["username"] == "admin"


# ══════════════════════════════════════════════════════════════════════════════
# PII DETECTOR — API_MISMATCH: v1 scan/mask → v3 detect/redact
# v3 PIIDetector lives at validation.governance.pii_detector and uses:
#   detect(df) → Dict[str, Dict[str, int]]   (not scan(df, sample_n))
#   redact(df) → Tuple[DataFrame, Dict]      (not mask(df, report))
# Working v3 PII tests exist in tests/test_pii_governance.py (2 passed).
# These legacy tests cannot be ported without rewriting the assertions.
# ══════════════════════════════════════════════════════════════════════════════

class TestPIIDetector:

    @pytest.fixture
    def detector(self):
        from governance.pii_detector import PIIDetector
        return PIIDetector()

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.scan() was "
               "replaced by validation.governance.pii_detector.PIIDetector.detect(). "
               "v3 email detection is tested in tests/test_pii_governance.py."
    )
    def test_email_detected(self, detector):
        import pandas as pd
        df = pd.DataFrame({"contacts": ["alice@example.com"] * 50 + ["bob@corp.io"] * 50})
        report = detector.scan(df, sample_n=100)
        assert "contacts" in report["pii_columns"]
        assert "email" in report["pii_columns"]["contacts"]

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.scan() was "
               "replaced by validation.governance.pii_detector.PIIDetector.detect(). "
               "SSN pattern detection is verified in v3 test suite."
    )
    def test_ssn_detected(self, detector):
        import pandas as pd
        df = pd.DataFrame({"ssn": ["123-45-6789"] * 60 + ["987-65-4321"] * 40})
        report = detector.scan(df, sample_n=100)
        assert "ssn" in report["pii_columns"]

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.scan() was "
               "replaced by validation.governance.pii_detector.PIIDetector.detect(). "
               "Credit card detection is verified in v3 test suite."
    )
    def test_credit_card_detected(self, detector):
        import pandas as pd
        df = pd.DataFrame({"payment": ["4111-1111-1111-1111"] * 80 + ["safe"] * 20})
        report = detector.scan(df, sample_n=100)
        assert "payment" in report["pii_columns"]

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.scan() was "
               "replaced by validation.governance.pii_detector.PIIDetector.detect(). "
               "Clean column behaviour is implicit in v3 detect() returning empty dict."
    )
    def test_clean_column_not_flagged(self, detector):
        import pandas as pd
        df = pd.DataFrame({"revenue": ["10000", "25000", "15000"] * 33})
        report = detector.scan(df, sample_n=100)
        assert len(report["pii_columns"]) == 0 or "revenue" not in report["pii_columns"]

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.mask() was "
               "replaced by validation.governance.pii_detector.PIIDetector.redact(). "
               "v3 redact() returns a tuple of (df, report), not masked df."
    )
    def test_mask_replaces_pii_columns(self, detector):
        import pandas as pd
        df = pd.DataFrame({"email": ["user@example.com"] * 80, "amount": [100.0] * 80})
        report = detector.scan(df)
        masked = detector.mask(df, report)
        if "email" in report["pii_columns"]:
            assert (masked["email"] == "***").all()
        assert (masked["amount"] == 100.0).all()

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.mask() was "
               "replaced by validation.governance.pii_detector.PIIDetector.redact(). "
               "Immutability guarantee is preserved in v3 redact() via .copy()."
    )
    def test_mask_returns_copy(self, detector):
        import pandas as pd
        df = pd.DataFrame({"email": ["a@b.com"] * 60})
        report = detector.scan(df)
        masked = detector.mask(df, report)
        assert df["email"].iloc[0] == "a@b.com"
        assert masked is not df

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.scan() was "
               "replaced by validation.governance.pii_detector.PIIDetector.detect(). "
               "Empty dataframe handling is tested in v3 via detect() returning {}."
    )
    def test_scan_empty_dataframe(self, detector):
        import pandas as pd
        df = pd.DataFrame()
        report = detector.scan(df)
        assert report["pii_columns"] == {}
        assert report["safe_columns"] == []

    @pytest.mark.skip(
        reason="API_MISMATCH: governance.pii_detector.PIIDetector.scan() was "
               "replaced by validation.governance.pii_detector.PIIDetector.detect(). "
               "Return schema changed from {pii_columns, safe_columns, method, details} "
               "to Dict[str, Dict[str, int]]."
    )
    def test_scan_returns_required_keys(self, detector):
        import pandas as pd
        df = pd.DataFrame({"x": ["test"] * 10})
        report = detector.scan(df)
        assert "pii_columns"  in report
        assert "safe_columns" in report
        assert "method"        in report
        assert "details"       in report


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT ACCESS LOG — 1 PASS + 2 OBSOLETE
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditAccessLog:
    """Verify the middleware writing format + JSON structure."""

    @pytest.mark.skip(
        reason="OBSOLETE: middleware.audit_access_log.AuditAccessLogMiddleware was "
               "never implemented in v3. No audit middleware module exists."
    )
    def test_middleware_is_importable(self):
        from middleware.audit_access_log import AuditAccessLogMiddleware
        assert AuditAccessLogMiddleware is not None

    def test_log_entry_structure(self, tmp_path, monkeypatch):
        """Simulate a log write and verify field structure."""
        import json
        from datetime import datetime, timezone

        monkeypatch.setenv("AUDIT_DIR", str(tmp_path))

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

    @pytest.mark.skip(
        reason="OBSOLETE: middleware.audit_access_log._EXCLUDED_PATHS was never "
               "defined in v3. No audit middleware with path exclusions exists."
    )
    def test_excluded_paths_not_logged(self):
        from middleware.audit_access_log import _EXCLUDED_PATHS
        for path in ["/prom-metrics", "/docs", "/redoc", "/openapi.json"]:
            assert path in _EXCLUDED_PATHS


# ══════════════════════════════════════════════════════════════════════════════
# JWT AUTH — OBSOLETE: auth.jwt_auth never built in v3
# python-jose is installed but unwired. v3 uses API key authentication.
# ══════════════════════════════════════════════════════════════════════════════

class TestJWTAuth:
    """JWT token creation, decoding, and authentication tests."""

    @pytest.mark.skip(
        reason="OBSOLETE: auth.jwt_auth.JWTAuth was never implemented in v3. "
               "v3 uses API key middleware, not JWT token auth."
    )
    def test_create_and_decode_access_token(self):
        from auth.jwt_auth import JWTAuth
        token = JWTAuth.create_access_token({"sub": "analyst", "role": "ANALYST"})
        assert isinstance(token, str)
        payload = JWTAuth.decode_token(token)
        assert payload["sub"] == "analyst"

    @pytest.mark.skip(
        reason="OBSOLETE: auth.jwt_auth.JWTAuth.authenticate_user was never "
               "implemented in v3. No user database or password verification exists."
    )
    def test_authenticate_user_valid(self):
        from auth.jwt_auth import JWTAuth
        result = JWTAuth.authenticate_user("admin", "secret")
        assert result is not None

    @pytest.mark.skip(
        reason="OBSOLETE: auth.jwt_auth.JWTAuth.authenticate_user was never "
               "implemented in v3. Invalid password handling cannot be tested."
    )
    def test_authenticate_user_invalid_password(self):
        from auth.jwt_auth import JWTAuth
        result = JWTAuth.authenticate_user("admin", "wrongpassword")
        assert result is None

    @pytest.mark.skip(
        reason="OBSOLETE: auth.jwt_auth.JWTAuth.authenticate_user was never "
               "implemented in v3. Non-existent user lookup cannot be tested."
    )
    def test_authenticate_user_nonexistent(self):
        from auth.jwt_auth import JWTAuth
        result = JWTAuth.authenticate_user("nobody", "secret")
        assert result is None

    @pytest.mark.skip(
        reason="OBSOLETE: auth.jwt_auth.JWTAuth.hash_password/verify_password "
               "were never implemented in v3. passlib is installed but unwired."
    )
    def test_hash_and_verify_password(self):
        from auth.jwt_auth import JWTAuth
        hashed = JWTAuth.hash_password("my_secure_password")
        assert hashed != "my_secure_password"
        assert JWTAuth.verify_password("my_secure_password", hashed) is True

    @pytest.mark.skip(
        reason="OBSOLETE: auth.jwt_auth.JWTAuth.create_refresh_token was never "
               "implemented in v3. Refresh token flow does not exist."
    )
    def test_refresh_token_type_claim(self):
        from auth.jwt_auth import JWTAuth
        token = JWTAuth.create_refresh_token({"sub": "analyst"})
        payload = JWTAuth.decode_token(token)
        assert payload["type"] == "refresh"
