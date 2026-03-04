"""
api/routes/drift.py
---------------------
Distribution drift detection endpoints.

POST /drift/detect      — run PSI + KL + JS + Wasserstein between reference and current datasets
POST /drift/correlation — correlation matrix for a dataset
POST /drift/permutation — permutation test
POST /drift/residuals   — residual diagnostics for a fitted model
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/drift", tags=["Drift & Advanced Stats"])
logger = logging.getLogger("dipex.api.drift")


class DriftRequest(BaseModel):
    reference_run_id: str
    current_run_id: str
    columns: Optional[List[str]] = None
    n_bins: int = 10


class CorrelationRequest(BaseModel):
    run_id: Optional[str] = None
    method: str = "spearman"          # pearson | spearman | kendall
    target: Optional[str] = None
    columns: Optional[List[str]] = None


class PermutationRequest(BaseModel):
    run_id: Optional[str] = None
    test: str = "two_sample_mean"     # two_sample_mean | correlation | anova
    col_a: str = ""
    col_b: Optional[str] = None
    group_col: Optional[str] = None
    n_permutations: int = 5000
    alpha: float = 0.05


class MultipleCorrectionRequest(BaseModel):
    p_values: List[float]
    method: str = "fdr_bh"            # bonferroni | holm | fdr_bh | fdr_by
    alpha: float = 0.05


@router.post("/detect")
async def detect_drift(req: DriftRequest):
    """Run full drift detection between reference and current dataset."""
    from stats.drift_detection import DriftDetector

    ref_df = _load_df(req.reference_run_id)
    cur_df = _load_df(req.current_run_id)

    dd = DriftDetector(n_bins=req.n_bins)
    result = dd.detect(ref_df, cur_df, columns=req.columns)

    return {
        "reference_run_id": req.reference_run_id,
        "current_run_id": req.current_run_id,
        **result,
    }


@router.post("/correlation")
async def correlation(req: CorrelationRequest):
    """Compute correlation matrix with p-values and VIF."""
    from stats.correlation import CorrelationAnalyzer

    df = _load_df(req.run_id)
    ca = CorrelationAnalyzer()
    report = ca.analyze(df, target=req.target, method=req.method)
    return report


@router.post("/permutation")
async def permutation_test(req: PermutationRequest):
    """Run a permutation test."""
    from stats.permutation_tests import PermutationTester

    df = _load_df(req.run_id)
    pt = PermutationTester(n_permutations=req.n_permutations)

    test = req.test.lower()
    if test == "two_sample_mean":
        if not req.col_a or not req.col_b:
            raise HTTPException(400, detail="col_a and col_b required for two_sample_mean")
        return pt.two_sample_mean(df[req.col_a], df[req.col_b], alpha=req.alpha)
    elif test == "correlation":
        if not req.col_a or not req.col_b:
            raise HTTPException(400, detail="col_a and col_b required for correlation test")
        return pt.correlation(df[req.col_a], df[req.col_b], alpha=req.alpha)
    elif test == "anova":
        if not req.group_col or not req.col_a:
            raise HTTPException(400, detail="col_a and group_col required for ANOVA")
        groups = {str(k): grp[req.col_a] for k, grp in df.groupby(req.group_col)}
        return pt.anova(groups, alpha=req.alpha)
    else:
        raise HTTPException(400, detail=f"Unknown test '{test}'")


@router.post("/multiple-correction")
async def multiple_correction(req: MultipleCorrectionRequest):
    """Apply multiple testing correction to a list of p-values."""
    from stats.permutation_tests import MultipleTestingCorrector
    mtc = MultipleTestingCorrector()
    return mtc.correct(req.p_values, method=req.method, alpha=req.alpha)


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
    raise HTTPException(404, detail="Dataset not found.")
