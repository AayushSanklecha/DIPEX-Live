from fastapi import APIRouter
from explanation.audit_logger import AuditLogger

router = APIRouter(prefix="/api/audit", tags=["audit"])
audit_logger = AuditLogger()

@router.get("/")
async def get_audit_logs(limit: int = 50):
    """Returns the most recent audit log entries."""
    return audit_logger.get_logs(limit=limit)
