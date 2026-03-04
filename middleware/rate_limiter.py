"""
middleware/rate_limiter.py
---------------------------
Token-bucket rate limiting middleware for DIPEX FastAPI.

Features:
  - Per-IP rate limiting with configurable window and max requests
  - Per-user (JWT sub) rate limiting for authenticated endpoints
  - Returns HTTP 429 with Retry-After header
  - Whitelisted paths (health, docs) exempt from limiting
  - In-process memory store (replace with Redis in production)

Config (config.yaml → rate_limiting):
  enabled        : true
  requests_per_minute : 60
  burst          : 10     # extra burst capacity above the window
  whitelist_paths: [/health, /docs, /openapi.json]

Usage (in api/app.py)::

    from middleware.rate_limiter import RateLimiterMiddleware
    app.add_middleware(RateLimiterMiddleware, requests_per_minute=60)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("dipex.middleware.rate_limiter")

DEFAULT_RPM     = 60
DEFAULT_BURST   = 15
WHITELIST_PATHS: Set[str] = {
    "/health", "/", "/docs", "/openapi.json",
    "/redoc", "/favicon.ico",
}


class _TokenBucket:
    """Sliding-window token bucket per client."""

    def __init__(self, rpm: int, burst: int) -> None:
        self.rpm   = rpm
        self.burst = burst
        self.window_secs = 60.0
        self._timestamps: Deque[float] = deque()

    def is_allowed(self) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - self.window_secs
        # Evict old timestamps
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        capacity = self.rpm + self.burst
        if len(self._timestamps) < capacity:
            self._timestamps.append(now)
            return True, 0.0
        # Compute when oldest request falls out of the window
        oldest = self._timestamps[0]
        retry_after = (oldest + self.window_secs) - now
        return False, max(retry_after, 1.0)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Per-IP token-bucket rate limiting middleware.

    Parameters
    ----------
    requests_per_minute : int  default 60
    burst               : int  extra burst above RPM (default 15)
    whitelist_paths     : set  paths exempt from limiting
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = DEFAULT_RPM,
        burst: int = DEFAULT_BURST,
        whitelist_paths: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(app)
        self.rpm = requests_per_minute
        self.burst = burst
        self.whitelist = whitelist_paths or WHITELIST_PATHS
        self._buckets: Dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(self.rpm, self.burst)
        )

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path

        # Whitelist check
        if path in self.whitelist or path.startswith("/static"):
            return await call_next(request)

        # Client key: prefer user ID from JWT, fall back to IP
        client_key = self._get_client_key(request)
        bucket = self._buckets[client_key]
        allowed, retry_after = bucket.is_allowed()

        if not allowed:
            logger.warning("Rate limit exceeded: client=%s path=%s retry_after=%.1fs",
                           client_key, path, retry_after)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.",
                    "retry_after_seconds": round(retry_after, 1),
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        response = await call_next(request)
        # Add rate limit headers
        remaining = max(0, self.rpm + self.burst - len(bucket._timestamps))
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        return response

    @staticmethod
    def _get_client_key(request: Request) -> str:
        """Extract a stable client identifier."""
        # Try JWT sub from Authorization header
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                from auth.jwt_auth import JWTAuth
                payload = JWTAuth.decode_token(auth.split(" ", 1)[1])
                sub = payload.get("sub")
                if sub:
                    return f"user:{sub}"
            except Exception:  # noqa: BLE001
                pass
        # Fall back to IP
        fwd = request.headers.get("X-Forwarded-For")
        ip = fwd.split(",")[0].strip() if fwd else request.client.host if request.client else "unknown"
        return f"ip:{ip}"
