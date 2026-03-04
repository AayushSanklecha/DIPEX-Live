from __future__ import annotations

"""
Pydantic schemas for the reporting service (STEP 9).
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class VerifiedReportInput(BaseModel):
    verified_insights: List[str] = Field(
        ..., description="List of text insights that have been fully verified."
    )
    approved_metrics: Dict[str, float] = Field(
        ..., description="Approved, aggregate metrics (no raw data)."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Final confidence score in [0,1]."
    )
    qa_passed: bool = Field(
        ..., description="Whether all QA/validation gates have passed."
    )
    dataset_hash: Optional[str] = Field(
        default=None,
        description="Optional hash of the underlying dataset snapshot for logging only.",
    )


class GeneratedReport(BaseModel):
    correlation_id: str
    report: str
    model_name: str
    confidence_score: float
    qa_passed: bool

