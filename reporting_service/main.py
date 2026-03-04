from __future__ import annotations

"""
FastAPI entrypoint for the governed LLM Reporting Service (STEP 9 & STEP 11).
"""

import logging
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from .config import (
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
)
from .llm_provider import get_llm_provider
from .prompt_builder import build_governed_prompt
from .output_validator import generate_with_validation
from .cost_tracker import log_usage
from .schemas import VerifiedReportInput, GeneratedReport
from . import logger as structured_logger

app = FastAPI(title="Governed LLM Reporting Service")

log = logging.getLogger(__name__)


@app.post("/generate-report", response_model=GeneratedReport)
async def generate_report(payload: VerifiedReportInput) -> GeneratedReport:
    """
    Generates a governed analytics report using ONLY verified inputs and
    approved metrics. Enforces QA and confidence thresholds before calling
    the LLM and applies strict output validation afterwards.
    """
    if not payload.qa_passed:
        raise HTTPException(
            status_code=400,
            detail="QA/validation has not passed; reporting is not permitted.",
        )

    if payload.confidence_score < CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Confidence score {payload.confidence_score:.3f} is below "
                f"threshold {CONFIDENCE_THRESHOLD:.3f}; reporting is not permitted."
            ),
        )

    # Build governed prompt – NEVER include raw data.
    prompt = build_governed_prompt(
        verified_insights=payload.verified_insights,
        approved_metrics=payload.approved_metrics,
        confidence_score=payload.confidence_score,
        qa_passed=payload.qa_passed,
    )

    provider = get_llm_provider()
    correlation_id = str(uuid.uuid4())

    try:
        text, tokens_used = generate_with_validation(
            provider=provider,
            prompt=prompt,
            approved_metrics=payload.approved_metrics,
            confidence_score=payload.confidence_score,
            qa_passed=payload.qa_passed,
        )
    except ValueError as exc:
        log.error("Reporting LLM output rejected after validation: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Generated report did not satisfy validation constraints.",
        ) from exc

    # Cost & token tracking
    usage_record = log_usage(
        prompt=prompt,
        response_text=text,
        model_name=MODEL_NAME,
        request_id=correlation_id,
        extra={
            "qa_passed": payload.qa_passed,
            "confidence_score": payload.confidence_score,
        },
    )

    # Structured logging (governance log)
    structured_logger.log_event(
        correlation_id=correlation_id,
        dataset_hash=payload.dataset_hash,
        confidence_score=payload.confidence_score,
        qa_passed=payload.qa_passed,
        model_name=MODEL_NAME,
        tokens_used=usage_record["total_tokens"],
        validation_status="ACCEPTED",
        extra={"request_type": "generate-report"},
    )

    return GeneratedReport(
        correlation_id=correlation_id,
        report=text,
        model_name=MODEL_NAME,
        confidence_score=payload.confidence_score,
        qa_passed=payload.qa_passed,
    )

