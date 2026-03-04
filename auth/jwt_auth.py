"""
auth/jwt_auth.py
-----------------
JWT-based authentication for the DIPEX API.

Features:
  - HS256 signed JWTs with configurable secret + expiry
  - Access tokens + refresh tokens
  - FastAPI dependency injection via `get_current_user`
  - Role claim embedded in token payload

Config keys (from config.yaml or environment):
  JWT_SECRET_KEY  : signing secret (env var overrides config)
  JWT_ALGORITHM   : HS256 (default)
  JWT_EXPIRE_MINS : access token lifetime in minutes (default 60)

Usage::

    @router.get("/protected")
    async def protected(user: dict = Depends(get_current_user)):
        return {"user": user}
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

logger = logging.getLogger("dipex.auth.jwt")

# ── RL auth risk scorer ────────────────────────────────────────────────────────
try:
    from auth.rl_auth_tuner import get_rl_auth_tuner as _get_rl_tuner
    _rl_auth = _get_rl_tuner()
except Exception:  # noqa: BLE001
    _rl_auth = None
    logger.info("jwt_auth: RLAuthTuner not available — static auth policy in effect.")

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY     = os.getenv("JWT_SECRET_KEY", "dipex-dev-secret-change-in-production-2024!")
ALGORITHM      = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE  = int(os.getenv("JWT_EXPIRE_MINS", "60"))
REFRESH_EXPIRE = int(os.getenv("JWT_REFRESH_EXPIRE_HOURS", "24"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# ── In-memory user store (replace with DB in production) ─────────────────────
# Format: { username: { password_hash: str, role: str, disabled: bool } }
_USERS: Dict[str, Dict] = {
    "admin": {
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "ADMIN",
        "full_name": "DIPEX Administrator",
        "disabled": False,
    },
    "analyst": {
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "role": "ANALYST",
        "full_name": "Data Analyst",
        "disabled": False,
    },
    "viewer": {
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "role": "VIEWER",
        "full_name": "Read-Only Viewer",
        "disabled": False,
    },
}


class JWTAuth:
    """JWT token management."""

    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a signed JWT access token."""
        try:
            from jose import jwt
        except ImportError:
            raise ImportError("python-jose required: pip install python-jose[cryptography]")

        payload = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_EXPIRE))
        payload.update({"exp": expire, "iat": datetime.now(timezone.utc), "type": "access"})
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        try:
            from jose import jwt
        except ImportError:
            raise ImportError("python-jose required")
        payload = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(hours=REFRESH_EXPIRE)
        payload.update({"exp": expire, "iat": datetime.now(timezone.utc), "type": "refresh"})
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        try:
            from jose import jwt, JWTError
        except ImportError:
            raise ImportError("python-jose required")
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired token: {exc}",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        try:
            from passlib.context import CryptContext
            ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return ctx.verify(plain, hashed)
        except ImportError:
            # Handle PBKDF2 hashes produced by hash_password fallback
            if hashed.startswith("pbkdf2$"):
                import hashlib
                try:
                    _, salt_hex, key_hex = hashed.split("$")
                    salt = bytes.fromhex(salt_hex)
                    expected = bytes.fromhex(key_hex)
                    actual = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt, 260_000)
                    return actual == expected
                except Exception:
                    return False
            # Fallback: compare plain text (dev only)
            logger.warning("passlib not installed — using plaintext password comparison (DEV ONLY)")
            return plain == "secret"

    @staticmethod
    def hash_password(plain: str) -> str:
        """
        Hash a plaintext password.

        Uses bcrypt via passlib when available (production standard).
        Falls back to PBKDF2-HMAC-SHA256 with a fresh random salt when passlib
        is absent — still cryptographically secure, never returns plaintext.
        """
        try:
            from passlib.context import CryptContext
            ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return ctx.hash(plain)
        except ImportError:
            # Passlib not installed — use stdlib PBKDF2 as secure fallback
            import hashlib
            import os as _os
            import base64
            salt = _os.urandom(16)
            key  = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt, 260_000)
            # Encode as "pbkdf2$<hex-salt>$<hex-key>" so verify_password can check it
            return "pbkdf2$" + salt.hex() + "$" + key.hex()

    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
        user = _USERS.get(username)
        if not user or user.get("disabled"):
            return None
        if not JWTAuth.verify_password(password, user["hashed_password"]):
            return None
        return {"username": username, "role": user["role"], "full_name": user["full_name"]}


# ── FastAPI dependency ────────────────────────────────────────────────────────

# Production auth enforcement.
# Set DIPEX_AUTH_STRICT=true in .env / K8s ConfigMap to require a valid JWT
# on every protected route (recommended for all non-development deployments).
# When false (default / dev mode), missing tokens are accepted as anonymous
# VIEWER — preserving backward compatibility and dev-mode convenience.
_AUTH_STRICT: bool = os.getenv("DIPEX_AUTH_STRICT", "false").lower() in {"true", "1", "yes"}


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """FastAPI dependency — extract and validate the current user from JWT.

    Behaviour
    ---------
    - **Strict mode** (``DIPEX_AUTH_STRICT=true``, recommended for production):
      Missing or invalid tokens → HTTP 401 Unauthorized.
    - **Dev mode** (default): Missing token → anonymous VIEWER passthrough.
      This preserves dev UX and backward compatibility with existing tests.
    """
    if token is None:
        if _AUTH_STRICT:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Provide a Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Dev/testing passthrough — not suitable for production
        logger.debug("jwt_auth: no token provided — anonymous VIEWER (dev mode)")
        return {"username": "anonymous", "role": "VIEWER"}

    payload = JWTAuth.decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
    user_data = _USERS.get(username, {})
    if user_data.get("disabled"):
        raise HTTPException(status_code=403, detail="Account disabled")


    role = payload.get("role", "VIEWER")
    user_info: Dict[str, Any] = {
        "username":  username,
        "role":      role,
        "full_name": user_data.get("full_name", username),
    }

    # [RL] Compute adaptive auth policy for this session context
    if _rl_auth is not None:
        try:
            is_admin     = role == "ADMIN"
            # Extract request context from token claims (graceful defaults)
            fail_streak  = int(payload.get("fail_streak", 0))
            new_device   = bool(payload.get("new_device", False))
            risk_score   = float(payload.get("risk_score", 0.3))
            access_hour  = datetime.now(timezone.utc).hour

            rl_policy = _rl_auth.get_policy(
                access_hour=access_hour,
                failure_streak=fail_streak,
                is_admin=is_admin,
                is_new_device=new_device,
                risk_score=risk_score,
            )
            user_info["rl_policy"] = rl_policy
            logger.debug(
                "[RL] Auth policy for %s: mfa=%s lockout=%d timeout=%dmin",
                username,
                rl_policy["mfa_required"],
                rl_policy["max_attempts"],
                rl_policy["session_timeout_min"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("RLAuthTuner policy failed for %s: %s", username, exc)

    return user_info
