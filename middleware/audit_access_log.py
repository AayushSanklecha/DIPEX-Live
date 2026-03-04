"""
middleware/audit_access_log.py
-------------------------------
DIPEX Audit Access Log Middleware.

Who accessed what, when, from where — for every authenticated API request.

Log format (one JSON line per request in ``audit/access_log.jsonl``):
{
  "ts":         "2026-02-28T16:00:00.123456+00:00",  // ISO-8601 UTC
  "request_id": "a3f2...",                            // UUID4 per request
  "username":   "analyst",                            // from JWT or "anonymous"
  "role":       "ANALYST",
  "method":     "POST",
  "path":       "/analyst/run",
  "status":     200,
  "duration_ms": 142.3,
  "client_ip":  "192.168.1.10",
  "user_agent": "httpx/0.26.0",
  "bytes_sent": 1024
}

Design Decisions
----------------
- Async-safe: appends atomically via single write() call
- Non-blocking: log write happens after response is sent
- Scrubs Authorization header — no tokens written to disk
- Excluded paths: /prom-metrics, /docs, /redoc, /openapi.json (noisy, low value)
- File rotates daily (one file per UTC date): access_log_YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dipex.middleware.audit")

_AUDIT_DIR = Path(os.getenv("AUDIT_DIR", "audit"))
_EXCLUDED_PATHS = frozenset({
    "/prom-metrics", "/docs", "/redoc", "/openapi.json",
    "/favicon.ico", "/robots.txt",
})


class AuditAccessLogMiddleware:
    """
    ASGI middleware that writes one JSON log line per authenticated API request.

    Integrate in app.py::

        from middleware.audit_access_log import AuditAccessLogMiddleware
        app.add_middleware(AuditAccessLogMiddleware)
    """

    def __init__(self, app) -> None:
        self.app = app
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _EXCLUDED_PATHS or path.startswith("/dashboard"):
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Capture status code from response
        status_code = 0
        bytes_sent  = 0
        response_started = False

        async def send_wrapper(message):
            nonlocal status_code, bytes_sent, response_started
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                response_started = True
            elif message["type"] == "http.response.body":
                bytes_sent += len(message.get("body", b""))
            await send(message)

        await self.app(scope, receive, send_wrapper)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        # Extract request metadata
        headers = dict(scope.get("headers", []))
        method  = scope.get("method", "UNKNOWN")
        client  = scope.get("client", ("unknown", 0))
        ip      = (headers.get(b"x-forwarded-for", client[0] if client else b"").decode("latin1")
                   .split(",")[0].strip())
        ua      = headers.get(b"user-agent", b"").decode("latin1", errors="replace")

        # Username / role from custom scope state set by JWT dependency (best-effort)
        username = "anonymous"
        role     = "VIEWER"
        try:
            state = scope.get("state", {})
            if hasattr(state, "user"):
                username = state.user.get("username", "anonymous")
                role     = state.user.get("role", "VIEWER")
        except Exception:
            pass

        entry = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "request_id":  request_id,
            "username":    username,
            "role":        role,
            "method":      method,
            "path":        path,
            "status":      status_code,
            "duration_ms": duration_ms,
            "client_ip":   ip,
            "user_agent":  ua[:200],
            "bytes_sent":  bytes_sent,
        }

        # Rotate daily
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = _AUDIT_DIR / f"access_log_{today}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("AuditAccessLog write failed: %s", exc)
