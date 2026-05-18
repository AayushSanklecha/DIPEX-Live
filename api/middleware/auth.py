# api/middleware/auth.py
"""
Minimal API key authentication middleware for DIPEX.

All requests to non-exempt paths must include:
  Header: X-API-Key: <value of DIPEX_API_KEY env var>

Exempt paths (no auth required):
  GET  /          (root)
  GET  /health    (liveness probe — needed by Docker healthcheck)
  GET  /docs      (Swagger UI)
  GET  /redoc     (ReDoc)
  GET  /openapi.json

Configuration:
  Set DIPEX_API_KEY in your .env file.
  If not set in production (DIPEX_ENV=production), startup aborts.
  If not set in development, auth is DISABLED with a clear warning.

Usage:
  Add to api/app.py:
    from api.middleware.auth import APIKeyMiddleware
    app.add_middleware(APIKeyMiddleware)
"""

import os
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._key = os.environ.get("DIPEX_API_KEY", "")
        self._env = os.environ.get("DIPEX_ENV", "development")

        if not self._key:
            if self._env == "production":
                raise EnvironmentError(
                    "DIPEX_API_KEY is not set. "
                    "Authentication is mandatory in production. "
                    "Set DIPEX_API_KEY in your .env file before starting."
                )
            logger.warning(
                "⚠ DIPEX_API_KEY not set — authentication DISABLED. "
                "All endpoints are publicly accessible. "
                "This is acceptable for local development ONLY."
            )

    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Auth disabled in dev mode (no key set)
        if not self._key:
            return await call_next(request)

        # Validate key
        provided = request.headers.get("X-API-Key", "")
        if provided != self._key:
            logger.warning(
                "Unauthorized request to %s from %s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Valid X-API-Key header required.",
                    "docs": "/docs",
                },
            )

        return await call_next(request)
