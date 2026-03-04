"""
api/routes/audit.py
-------------------
Simplified audit log endpoint.
"""

import json
import os
from fastapi import APIRouter

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/")
async def get_audit_logs(limit: int = 50):
    """Returns the most recent audit log entries."""
    audit_log_path = "audit/audit.jsonl"
    logs = []
    
    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    pass
    
    return {"entries": logs[-limit:] if logs else [], "total": len(logs)}
