"""
api/routes/stats.py
---------------------
Statistical analysis endpoints.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/stats", tags=["Statistical Analysis"])
logger = logging.getLogger("dipex.api.stats")


class DescribeRequest(BaseModel):
    run_id: Optional[str] = None
    columns: Optional[List[str]] = None


class HypothesisRequest(BaseModel):
    run_id: Optional[str] = None
    test: str                  # one_sample_t | two_sample_t | chi_square | anova | mann_whitney | kruskal
    col_a: str
    col_b: Optional[str] = None
    group_col: Optional[str] = None
    pop_mean: float = 0.0
    alpha: float = 0.05


class RegressionRequest(BaseModel):
    run_id: Optional[str] = None
    target: str
    features: Optional[List[str]] = None
    model_type: str = "ols"    # ols | logistic | ridge | lasso


@router.post("/describe")
async def describe(req: DescribeRequest):
    """Full descriptive statistics for numeric columns."""
    from stats.descriptive import DescriptiveStats
    df = _load_df(req.run_id)
    ds = DescriptiveStats()
    report = ds.analyze(df, columns=req.columns)
    summary_df = ds.to_dataframe(report)
    return {
        "dataset_summary": report["dataset_summary"],
        "summary_table": summary_df.to_dict(orient="records"),
        "columns": list(report["columns"].keys()),
    }


@router.post("/hypotheses")
async def hypothesis_test(req: HypothesisRequest):
    """Run hypothesis tests."""
    from stats.hypothesis_tests import HypothesisTester
    df = _load_df(req.run_id)
    ht = HypothesisTester(alpha=req.alpha)

    test = req.test.lower()
    if test == "one_sample_t":
        if req.col_a not in df.columns:
            raise HTTPException(400, detail=f"Column '{req.col_a}' not found")
        return ht.one_sample_t(df[req.col_a], popmean=req.pop_mean)
    elif test == "two_sample_t":
        if not req.col_b:
            raise HTTPException(400, detail="col_b required for two_sample_t")
        return ht.two_sample_t(df[req.col_a], df[req.col_b])
    elif test == "mann_whitney":
        if not req.col_b:
            raise HTTPException(400, detail="col_b required for mann_whitney")
        return ht.mann_whitney_u(df[req.col_a], df[req.col_b])
    elif test == "anova":
        if not req.group_col:
            raise HTTPException(400, detail="group_col required for ANOVA")
        groups = {str(k): grp[req.col_a] for k, grp in df.groupby(req.group_col)}
        return ht.one_way_anova(groups)
    elif test == "kruskal":
        if not req.group_col:
            raise HTTPException(400, detail="group_col required for kruskal")
        groups = {str(k): grp[req.col_a] for k, grp in df.groupby(req.group_col)}
        return ht.kruskal_wallis(groups)
    elif test == "pearson":
        if not req.col_b:
            raise HTTPException(400, detail="col_b required for pearson")
        return ht.pearson_correlation(df[req.col_a], df[req.col_b])
    elif test == "spearman":
        if not req.col_b:
            raise HTTPException(400, detail="col_b required for spearman")
        return ht.spearman_correlation(df[req.col_a], df[req.col_b])
    else:
        raise HTTPException(400, detail=f"Unknown test '{test}'")


@router.post("/regression")
async def regression(req: RegressionRequest):
    """Fit a regression model."""
    from stats.regression import RegressionEngine
    df = _load_df(req.run_id)
    engine = RegressionEngine()

    mt = req.model_type.lower()
    if mt == "ols":
        result = engine.ols(df, target=req.target, features=req.features)
    elif mt == "logistic":
        result = engine.logistic(df, target=req.target, features=req.features)
    elif mt == "ridge":
        result = engine.ridge(df, target=req.target, features=req.features)
    elif mt == "lasso":
        result = engine.lasso(df, target=req.target, features=req.features)
    else:
        raise HTTPException(400, detail=f"Unknown model_type '{mt}'")

    return result.to_dict()


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
