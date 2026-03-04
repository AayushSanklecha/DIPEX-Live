"""
api/routes/cohort.py
----------------------
Cohort analysis API endpoints.

POST /cohort/retention   — compute retention matrix
POST /cohort/ltv         — compute LTV cohort curves
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/cohort", tags=["Cohort Analysis"])
logger = logging.getLogger("dipex.api.cohort")


class RetentionRequest(BaseModel):
    run_id: Optional[str] = None
    cohort_col: str          # e.g. "signup_month"
    entity_col: str          # e.g. "user_id"
    activity_col: str        # e.g. "activity_month"
    max_periods: int = 12


class LTVRequest(BaseModel):
    run_id: Optional[str] = None
    cohort_col: str
    entity_col: str
    activity_col: str
    value_col: str           # e.g. "revenue"
    max_periods: int = 12


@router.post("/retention")
async def cohort_retention(req: RetentionRequest):
    """Compute cohort retention matrix."""
    from query_engine.cohort_analysis import CohortAnalyzer

    df = _load_df(req.run_id)
    ca = CohortAnalyzer()
    result = ca.retention_matrix(
        df,
        cohort_col=req.cohort_col,
        entity_col=req.entity_col,
        activity_col=req.activity_col,
        max_periods=req.max_periods,
    )
    if "error" in result:
        raise HTTPException(400, detail=result["error"])

    summary = ca.summary_stats(result)
    return {**result, "summary": summary}


@router.post("/ltv")
async def cohort_ltv(req: LTVRequest):
    """Compute cumulative LTV cohort curves."""
    from query_engine.cohort_analysis import CohortAnalyzer

    df = _load_df(req.run_id)
    ca = CohortAnalyzer()
    result = ca.ltv_cohorts(
        df,
        cohort_col=req.cohort_col,
        entity_col=req.entity_col,
        activity_col=req.activity_col,
        value_col=req.value_col,
        max_periods=req.max_periods,
    )
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


def _load_df(run_id: Optional[str]) -> pd.DataFrame:
    if run_id:
        for suffix in ["_cleaned.csv", "_sample.csv"]:
            path = f"data/uploads/{run_id}{suffix}"
            if os.path.exists(path):
                return pd.read_csv(path)
    upload_dir = "data/uploads"
    if os.path.exists(upload_dir):
        files = sorted(
            [f for f in os.listdir(upload_dir) if f.endswith(".csv")],
            key=lambda f: os.path.getmtime(os.path.join(upload_dir, f)), reverse=True,
        )
        if files:
            return pd.read_csv(os.path.join(upload_dir, files[0]))
    raise HTTPException(404, detail="No dataset available.")
