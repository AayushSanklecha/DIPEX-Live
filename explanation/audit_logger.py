"""
explanation/audit_logger.py
----------------------------
Maintains an append-only, structured (JSONL) audit trail of pipeline events.
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Maintains an immutable, structured JSONL audit trail.

    Events are appended in ISO-8601 UTC and are never overwritten,
    guaranteeing an immutable audit chain.
    """

    def __init__(self, log_path: str = "audit/audit.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Appends a timestamped event to the JSONL audit log."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        logger.debug("Audit event logged: %s", event_type)

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Returns the last `limit` events from the audit log.

        Uses a bounded deque (O(limit) memory) so it does not read the
        entire file into RAM regardless of log file size.
        """
        if not self.log_path.exists():
            return []

        buf: deque = deque(maxlen=limit)
        with open(self.log_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    buf.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed audit log line: %s", exc)

        return list(buf)
