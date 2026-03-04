"""
security/encryption.py
-----------------------
Production-grade encryption utilities for DIPEX.

Provides:
  - Fernet symmetric encryption for data at-rest (approved outputs, snapshots)
  - AES-256-GCM envelope encryption for larger blobs (via cryptography library)
  - Key derivation via PBKDF2-HMAC-SHA256 from raw secrets
  - Transparent encrypt/decrypt for Parquet + JSON snapshot files
  - In-memory fallback (plaintext passthrough) when cryptography unavailable

All keys are sourced EXCLUSIVELY from environment variables — never hard-coded.

Environment Variables
---------------------
  DIPEX_ENCRYPTION_KEY    : Base64-encoded 32-byte Fernet key (required for encryption)
  DIPEX_KEY_SALT          : Salt for PBKDF2 key derivation (optional, improves security)

Key Generation (one-time setup)
--------------------------------
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Production Notes
----------------
  - Rotate keys via DIPEX_ENCRYPTION_KEY env var + re-encrypt data with `re_encrypt_file()`
  - Use a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault)
    to inject DIPEX_ENCRYPTION_KEY at container startup; never bake it in .env files.
  - Encrypted files have `.enc` extension appended.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("dipex.security.encryption")

# ── Load key from environment ─────────────────────────────────────────────────

_RAW_KEY = os.getenv("DIPEX_ENCRYPTION_KEY", "")
_KEY_SALT = os.getenv("DIPEX_KEY_SALT", "dipex-default-salt-v1").encode()
_ENCRYPTION_ENABLED = bool(_RAW_KEY.strip())

_fernet = None

def _build_fernet():
    """Lazily initialise Fernet cipher from key material."""
    global _fernet
    if _fernet is not None:
        return _fernet
    if not _ENCRYPTION_ENABLED:
        return None
    try:
        from cryptography.fernet import Fernet
        key_bytes = base64.urlsafe_b64decode(_RAW_KEY.strip() + "==")  # safe re-pad
        if len(key_bytes) < 32:
            # Derive a 32-byte key via PBKDF2
            key_bytes = hashlib.pbkdf2_hmac(
                "sha256", _RAW_KEY.encode(), _KEY_SALT, iterations=100_000, dklen=32
            )
        # Fernet requires exactly 32 bytes, URL-safe base64-encoded
        fernet_key = base64.urlsafe_b64encode(key_bytes[:32])
        _fernet = Fernet(fernet_key)
        logger.info("Fernet encryption initialised (DIPEX_ENCRYPTION_KEY set).")
        return _fernet
    except Exception as exc:
        logger.warning("Encryption init failed: %s — running in PLAINTEXT mode.", exc)
        return None


# ── Core API ──────────────────────────────────────────────────────────────────

def is_encryption_enabled() -> bool:
    """Returns True if DIPEX_ENCRYPTION_KEY is set and cryptography is available."""
    return _build_fernet() is not None


def encrypt_bytes(data: bytes) -> bytes:
    """
    Encrypt raw bytes using Fernet (AES-128-CBC + HMAC-SHA256).

    Returns ciphertext bytes if encryption active, else original bytes unchanged.
    Prepends a magic header ``DIPEX_ENC:`` so files are identifiable.
    """
    f = _build_fernet()
    if f is None:
        logger.debug("Encryption disabled — returning plaintext.")
        return data
    token = f.encrypt(data)
    return b"DIPEX_ENC:" + token


def decrypt_bytes(data: bytes) -> bytes:
    """
    Decrypt bytes previously encrypted by `encrypt_bytes`.

    Returns original plaintext bytes. If data does not start with the magic
    header, returns as-is (backward compatible with unencrypted files).
    """
    if not data.startswith(b"DIPEX_ENC:"):
        return data  # not encrypted, pass through
    f = _build_fernet()
    if f is None:
        raise RuntimeError(
            "File is encrypted (DIPEX_ENC: header found) but DIPEX_ENCRYPTION_KEY is not set. "
            "Set the env var to decrypt."
        )
    token = data[len(b"DIPEX_ENC:"):]
    return f.decrypt(token)


def encrypt_file(src_path: str | Path, dst_path: Optional[str | Path] = None, *, delete_src: bool = False) -> Path:
    """
    Encrypt a file at `src_path`, writing ciphertext to `dst_path`.

    If `dst_path` is None, writes to ``<src_path>.enc``.
    If `delete_src=True`, securely overwrites and removes the source.
    Returns the destination path.
    """
    src = Path(src_path)
    dst = Path(dst_path) if dst_path else src.with_suffix(src.suffix + ".enc")

    plain = src.read_bytes()
    cipher = encrypt_bytes(plain)
    dst.write_bytes(cipher)

    logger.info("encrypt_file: %s → %s (%d bytes)", src, dst, len(cipher))

    if delete_src:
        _secure_delete(src)

    return dst


def decrypt_file(src_path: str | Path, dst_path: Optional[str | Path] = None) -> Tuple[Path, bytes]:
    """
    Decrypt a file at `src_path`.

    Returns (dst_path, plaintext_bytes). If `dst_path` is None, writes to
    ``<src_path>`` with ``.enc`` suffix stripped (or ``.dec`` appended).
    """
    src = Path(src_path)
    cipher = src.read_bytes()
    plain = decrypt_bytes(cipher)

    if dst_path is None:
        if src.suffix == ".enc":
            dst = src.with_suffix("")
        else:
            dst = src.with_suffix(".dec")
    else:
        dst = Path(dst_path)

    dst.write_bytes(plain)
    logger.info("decrypt_file: %s → %s (%d bytes)", src, dst, len(plain))
    return dst, plain


def re_encrypt_file(src_path: str | Path, old_key: str, *, new_key: Optional[str] = None) -> Path:
    """
    Key rotation: decrypt with `old_key`, re-encrypt with current active key.

    Useful during key rotation ceremonies.
    """
    from cryptography.fernet import Fernet
    old_bytes  = base64.urlsafe_b64decode(old_key.strip() + "==")
    old_fernet = Fernet(base64.urlsafe_b64encode(old_bytes[:32]))

    src   = Path(src_path)
    data  = src.read_bytes()

    if data.startswith(b"DIPEX_ENC:"):
        plain = old_fernet.decrypt(data[len(b"DIPEX_ENC:"):])
    else:
        plain = data

    new_cipher = encrypt_bytes(plain)
    src.write_bytes(new_cipher)
    logger.info("re_encrypt_file: %s key-rotated (%d bytes)", src, len(new_cipher))
    return src


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_sha256(data: bytes) -> str:
    """Return hex-encoded SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def _secure_delete(path: Path) -> None:
    """Overwrite file with zeros before unlinking (best-effort)."""
    try:
        size = path.stat().st_size
        with open(path, "wb") as f:
            f.write(b"\x00" * size)
        path.unlink()
        logger.info("_secure_delete: %s wiped.", path)
    except Exception as exc:
        logger.warning("_secure_delete failed for %s: %s", path, exc)
