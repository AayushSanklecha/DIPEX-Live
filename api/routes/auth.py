"""
api/routes/auth.py
-------------------
Authentication endpoints: token issue, refresh, user info.

POST /auth/token   — issue JWT access + refresh tokens (OAuth2 password flow)
POST /auth/refresh — exchange refresh token for a new access token
GET  /auth/me      — return current user profile
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from auth.jwt_auth import JWTAuth, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger("dipex.api.auth")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue JWT access + refresh tokens (OAuth2 password flow)."""
    user = JWTAuth.authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning("Failed login attempt for username: %s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = {"sub": user["username"], "role": user["role"]}
    access_token  = JWTAuth.create_access_token(token_data)
    refresh_token = JWTAuth.create_refresh_token(token_data)

    logger.info("Issued token for user '%s' (role=%s)", user["username"], user["role"])
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        role=user["role"],
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    """Exchange a refresh token for a new access token."""
    try:
        payload = JWTAuth.decode_token(req.refresh_token)
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Provided token is not a refresh token")

    token_data = {"sub": payload["sub"], "role": payload.get("role", "VIEWER")}
    access_token  = JWTAuth.create_access_token(token_data)
    refresh_token = JWTAuth.create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        role=token_data["role"],
    )


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    """Return current authenticated user profile."""
    return {
        "username": user["username"],
        "role": user["role"],
        "full_name": user.get("full_name", user["username"]),
        "permissions": {
            "can_run_pipeline": user["role"] in ("ANALYST", "ADMIN", "API_SERVICE"),
            "can_train_models": user["role"] in ("ADMIN", "API_SERVICE"),
            "can_modify_catalog": user["role"] in ("ADMIN", "API_SERVICE"),
            "can_view_reports": True,
        },
    }
