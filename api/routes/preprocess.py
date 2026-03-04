"""
api/routes/preprocess.py
--------------------------
Preprocessing endpoint — clean + feature-engineer a dataset for a run_id.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

router = APIRouter(prefix="/preprocess", tags=["Preprocessing"])
logger = logging.getLogger("dipex.api.preprocess")


class PreprocessRequest(BaseModel):
    run_id: str
    target_column: Optional[str] = "target"
    overrides: Optional[Dict[str, Any]] = None


@router.post("")
async def preprocess(req: PreprocessRequest):
    """Clean and feature-engineer a dataset for the given run_id."""
    import yaml
    from preprocessing.cleaner import DataCleaner
    from preprocessing.feature_engineer import FeatureEngineer

    data_path = f"data/uploads/{req.run_id}_sample.csv"
    if not os.path.exists(data_path):
        raise HTTPException(404, detail=f"Dataset for run_id '{req.run_id}' not found at {data_path}")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if req.overrides:
        config.update(req.overrides)

    df = pd.read_csv(data_path)

    cleaner = DataCleaner.from_config(config)
    df_clean, clean_report = cleaner.clean(df, run_id=req.run_id)

    fe = FeatureEngineer.from_config(config)
    df_eng, fe_report = fe.engineer(df_clean, run_id=req.run_id, target_col=req.target_column)

    # Save cleaned dataset
    clean_path = f"data/uploads/{req.run_id}_cleaned.csv"
    df_eng.to_csv(clean_path, index=False)

    return {
        "run_id": req.run_id,
        "rows_before": clean_report.rows_before,
        "rows_after": clean_report.rows_after,
        "duplicates_removed": clean_report.duplicates_removed,
        "features_added": fe_report.features_added,
        "cleaning_operations": len(clean_report.imputation_log) + len(clean_report.capping_log),
        "output_path": clean_path,
        "status": "CLEANED",
    }
