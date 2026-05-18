# utils/snapshot_guard.py
"""
Snapshot encryption guard.
Prevents unencrypted PII/PHI data from being written to disk in production.

Issue 07: Fernet encryption support for at-rest snapshot security.
Encryption key: DIPEX_ENCRYPTION_KEY env var (Fernet format).
"""

import os
import warnings
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def get_fernet() -> Fernet | None:
    """
    Returns a Fernet instance if DIPEX_ENCRYPTION_KEY is set.
    Returns None in development mode (emits warning).
    Raises EnvironmentError in production mode if key is missing.
    """
    key = os.environ.get("DIPEX_ENCRYPTION_KEY")
    env = os.environ.get("DIPEX_ENV", "development")

    if not key:
        if env == "production":
            raise EnvironmentError(
                "DIPEX_ENCRYPTION_KEY is not set. "
                "All snapshots contain potentially sensitive data and MUST be "
                "encrypted in production. Set DIPEX_ENCRYPTION_KEY before starting."
            )
        warnings.warn(
            "DIPEX_ENCRYPTION_KEY not set — snapshots are UNENCRYPTED. "
            "This is acceptable for development only.",
            stacklevel=2,
        )
        return None

    return Fernet(key.encode())


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt data bytes. In dev mode without key, returns data unchanged."""
    f = get_fernet()
    return f.encrypt(data) if f else data


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt data bytes. In dev mode without key, returns data unchanged."""
    f = get_fernet()
    if f is None:
        return data
    try:
        return f.decrypt(data)
    except InvalidToken as exc:
        raise ValueError(
            "Snapshot decryption failed. The DIPEX_ENCRYPTION_KEY may be "
            "wrong or the snapshot may be corrupted."
        ) from exc
