"""
auth/rbac.py
-------------
Role-Based Access Control (RBAC) for DIPEX.

Roles (ascending privilege):
  VIEWER      — read-only: reports, audit, results
  ANALYST     — + run pipeline, preprocess, stats, SQL, cohort
  ADMIN       — + train models, governance changes, catalog writes
  API_SERVICE — programmatic full-access (service accounts)

Permissions matrix is enforced via the `require_role` FastAPI dependency.

Usage::

    from auth.rbac import require_role

    @router.post("/governance/catalog/register")
    async def register(
        req: CatalogRegisterRequest,
        user: dict = Depends(require_role("ANALYST")),
    ):
        ...
"""

from __future__ import annotations

from typing import List, Set

from fastapi import Depends, HTTPException, status

from auth.jwt_auth import get_current_user

# ── Role hierarchy ────────────────────────────────────────────────────────────
ROLES = {
    "VIEWER":      0,
    "ANALYST":     1,
    "ADMIN":       2,
    "API_SERVICE": 3,
}

# ── Permission matrix ─────────────────────────────────────────────────────────
PERMISSIONS: dict[str, Set[str]] = {
    # action                         : minimum role required
    "run_pipeline":                    {"ANALYST", "ADMIN", "API_SERVICE"},
    "preprocess":                      {"ANALYST", "ADMIN", "API_SERVICE"},
    "execute_sql":                     {"ANALYST", "ADMIN", "API_SERVICE"},
    "run_stats":                       {"ANALYST", "ADMIN", "API_SERVICE"},
    "run_cohort":                      {"ANALYST", "ADMIN", "API_SERVICE"},
    "run_drift":                       {"ANALYST", "ADMIN", "API_SERVICE"},
    "view_reports":                    {"VIEWER",  "ANALYST", "ADMIN", "API_SERVICE"},
    "generate_report":                 {"ANALYST", "ADMIN", "API_SERVICE"},
    "view_audit":                      {"VIEWER",  "ANALYST", "ADMIN", "API_SERVICE"},
    "view_governance":                 {"VIEWER",  "ANALYST", "ADMIN", "API_SERVICE"},
    "modify_governance_catalog":       {"ADMIN",   "API_SERVICE"},
    "train_model":                     {"ADMIN",   "API_SERVICE"},
    "view_model_registry":             {"ANALYST", "ADMIN", "API_SERVICE"},
    "admin_only":                      {"ADMIN",   "API_SERVICE"},
}


def has_permission(role: str, action: str) -> bool:
    """Check if a role is authorized for an action."""
    allowed = PERMISSIONS.get(action, {"ADMIN"})
    user_level = ROLES.get(role, -1)
    return any(ROLES.get(r, -1) <= user_level for r in allowed if ROLES.get(r, -1) <= user_level)


def require_role(minimum_role: str):
    """
    FastAPI dependency factory.

    Usage::

        @router.post("/model/train")
        async def train(user: dict = Depends(require_role("ADMIN"))):
            ...
    """
    min_level = ROLES.get(minimum_role, 999)

    async def _check_role(user: dict = Depends(get_current_user)) -> dict:
        role = user.get("role", "VIEWER")
        user_level = ROLES.get(role, -1)
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Insufficient privileges. Required: {minimum_role} "
                    f"(you are: {role}). Contact your DIPEX administrator."
                ),
            )
        return user

    return _check_role


def require_permission(action: str):
    """Dependency that checks a named permission."""
    async def _check(user: dict = Depends(get_current_user)) -> dict:
        role = user.get("role", "VIEWER")
        if not has_permission(role, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' is not authorized for action '{action}'.",
            )
        return user
    return _check


class RBACMiddleware:
    """Lightweight middleware to inject user info for logging (does not enforce — use dependencies)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        # Pass through — enforcement is done at dependency level
        await self.app(scope, receive, send)
