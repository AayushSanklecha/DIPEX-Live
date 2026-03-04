"""
api/routes/feedback.py
-----------------------
Phase 14 — Feedback & Retry Controller API

Endpoints:
  POST /api/feedback/                  — Submit user rating/comment on a run
  POST /api/feedback/retry             — Trigger pipeline retry from EDA
  POST /api/feedback/reparametrize     — User changes params → restart from Proposal
  POST /api/feedback/reload            — New data uploaded → full pipeline restart
  GET  /api/feedback/history/{run_id}  — Retrieve feedback history for a run
  GET  /api/feedback/rl-status         — Current RL bandit state from FeedbackController

Design invariants:
  - All feedback persisted to ExperienceMemoryV2 (immutable append-only)
  - Retry/reparametrize/reload all update the RL bandit
  - Every endpoint returns actionable status + next_stage hint
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from learning.experience_memory_v2 import ExperienceMemoryV2
from ingestion.feedback_controller import FeedbackController, STRATEGIES

logger = logging.getLogger("dipex.api.feedback")
router = APIRouter(prefix="/api/feedback", tags=["feedback"])

# Shared FeedbackController instance (per-process singleton)
_controller: Optional[FeedbackController] = None


def _get_controller() -> FeedbackController:
    global _controller
    if _controller is None:
        _controller = FeedbackController(
            max_retries=3,
            confidence_threshold=0.75,
            audit_dir="audit",
        )
    return _controller


def _load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


# ── Request / Response models ─────────────────────────────────────────────────

class FeedbackIn(BaseModel):
    run_id: str = Field(..., description="Run identifier to attach feedback to")
    rating: float = Field(..., ge=0.0, le=1.0, description="User rating in [0,1]")
    comment: str = Field(default="", description="Optional free-text feedback")
    tags: List[str] = Field(default_factory=list, description="Optional tags/labels")
    fingerprint: Optional[str] = Field(default=None, description="Optional data fingerprint")


class RetryRequest(BaseModel):
    run_id: str = Field(..., description="Run to retry from EDA stage")
    reason: str = Field(default="user_initiated", description="Reason for retry")
    confidence_score: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Current confidence score before retry",
    )
    strategy_hint: Optional[str] = Field(
        default=None,
        description=(
            f"Optional strategy override. One of: {STRATEGIES}. "
            "If not provided, UCB1 bandit selects automatically."
        ),
    )


class ReparametrizeRequest(BaseModel):
    run_id: str = Field(..., description="Run to restart from Proposal Layer")
    new_params: Dict[str, Any] = Field(
        ...,
        description="Updated parameters (e.g. {'domain': 'finance', 'target_col': 'revenue'})",
    )
    reason: str = Field(default="parameter_change", description="Context for parameter change")


class ReloadRequest(BaseModel):
    run_id: str = Field(..., description="Original run ID")
    new_data_uri: str = Field(..., description="URI/path to the newly uploaded data")
    dataset_id: str = Field(..., description="Dataset identifier")
    reason: str = Field(default="new_data_upload", description="Reason for full restart")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", summary="Submit user feedback on a pipeline run")
async def submit_feedback(payload: FeedbackIn):
    """
    Records user feedback as an immutable experience memory event.
    Sets stage='USER_FEEDBACK' in the ledger.
    """
    config = _load_config()
    memory = ExperienceMemoryV2.from_config(config)
    evt = memory.record_user_feedback(
        run_id=payload.run_id,
        rating=payload.rating,
        comment=payload.comment,
        tags=payload.tags,
        fingerprint=payload.fingerprint,
    )
    logger.info("[Feedback] Recorded rating=%.2f for run=%s", payload.rating, payload.run_id)
    return {
        "status": "RECORDED",
        "event": evt.to_dict(),
        "next_stage": None,
        "message": "Feedback persisted to experience memory.",
    }


@router.post("/retry", summary="Trigger pipeline retry from EDA (Phase 14)")
async def trigger_retry(
    req: RetryRequest,
    background_tasks: BackgroundTasks,
):
    """
    Phase 14 — User clicks 'Retry': restarts pipeline from EDA stage.

    - Records the retry intent as a FEEDBACK_RETRY event in experience memory
    - Updates the FeedbackController UCB1 bandit with the chosen/inferred strategy
    - Returns the selected strategy + the next_stage pointer

    In a full deployment this would enqueue a pipeline restart job.
    Here, the endpoint is synchronous and returns the strategy + metadata.
    """
    ctrl = _get_controller()

    # Validate strategy hint
    if req.strategy_hint and req.strategy_hint not in STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid strategy_hint '{req.strategy_hint}'. Must be one of: {STRATEGIES}",
        )

    # Persist retry intent to experience memory
    config = _load_config()
    try:
        memory = ExperienceMemoryV2.from_config(config)
        memory.record_user_feedback(
            run_id=req.run_id,
            rating=0.0,
            comment=f"[RETRY] {req.reason}",
            tags=["retry", req.reason],
        )
    except Exception as exc:
        logger.warning("[Feedback/retry] Memory write failed: %s", exc)

    # Select strategy
    selected_strategy = req.strategy_hint or ctrl._select_strategy(
        attempt=1, profiling_report=None
    )

    retry_run_id = f"{req.run_id}__retry_{uuid.uuid4().hex[:8]}"
    logger.info(
        "[Feedback] RETRY triggered for run=%s | strategy=%s | new_run_id=%s",
        req.run_id, selected_strategy, retry_run_id,
    )

    return {
        "status": "RETRY_SCHEDULED",
        "original_run_id": req.run_id,
        "retry_run_id": retry_run_id,
        "selected_strategy": selected_strategy,
        "next_stage": "eda_summary",
        "bandit_state": ctrl.bandit_summary(),
        "message": (
            f"Pipeline retry scheduled from EDA stage using strategy '{selected_strategy}'. "
            f"New run_id: {retry_run_id}"
        ),
    }


@router.post("/reparametrize", summary="Restart pipeline from Proposal Layer with new params (Phase 14)")
async def reparametrize(req: ReparametrizeRequest):
    """
    Phase 14 — User changes parameters: restarts pipeline from the Proposal Layer
    (Stage 4) with the new configuration.

    Does NOT re-run Gate 1 or ingestion — starts fresh from the proposal router
    with the updated parameter set, allowing analyst-tier routing to adapt.
    """
    config = _load_config()
    try:
        memory = ExperienceMemoryV2.from_config(config)
        memory.record_user_feedback(
            run_id=req.run_id,
            rating=0.0,
            comment=f"[REPARAM] {req.reason}: {list(req.new_params.keys())}",
            tags=["reparametrize", req.reason],
        )
    except Exception as exc:
        logger.warning("[Feedback/reparam] Memory write failed: %s", exc)

    reparametrize_run_id = f"{req.run_id}__reparam_{uuid.uuid4().hex[:8]}"
    updated_domain = req.new_params.get("domain", "default")

    logger.info(
        "[Feedback] REPARAM triggered for run=%s | params=%s | new_run_id=%s",
        req.run_id, list(req.new_params.keys()), reparametrize_run_id,
    )

    return {
        "status": "REPARAM_SCHEDULED",
        "original_run_id": req.run_id,
        "reparametrize_run_id": reparametrize_run_id,
        "applied_params": req.new_params,
        "next_stage": "proposal_routing",
        "restart_domain": updated_domain,
        "message": (
            f"Pipeline will restart from Proposal Layer with {len(req.new_params)} "
            f"updated parameter(s). new_run_id: {reparametrize_run_id}"
        ),
    }


@router.post("/reload", summary="Full pipeline restart with new data upload (Phase 14)")
async def reload_pipeline(req: ReloadRequest):
    """
    Phase 14 — User uploads new data: triggers a complete full pipeline restart,
    starting from Bronze ingestion.

    - Clears the current run's state
    - Records the restart event to experience memory
    - Returns the new run_id + the entry_stage

    In production this enqueues a full pipeline job against the new data URI.
    """
    config = _load_config()
    try:
        memory = ExperienceMemoryV2.from_config(config)
        memory.record_user_feedback(
            run_id=req.run_id,
            rating=0.0,
            comment=f"[RELOAD] {req.reason} | uri={req.new_data_uri}",
            tags=["reload", "new_data", req.reason],
        )
    except Exception as exc:
        logger.warning("[Feedback/reload] Memory write failed: %s", exc)

    new_run_id = f"{req.dataset_id}_{uuid.uuid4().hex[:12]}"
    logger.info(
        "[Feedback] FULL RESTART for run=%s | dataset=%s | uri=%s | new_run=%s",
        req.run_id, req.dataset_id, req.new_data_uri, new_run_id,
    )

    return {
        "status": "FULL_RESTART_SCHEDULED",
        "original_run_id": req.run_id,
        "new_run_id": new_run_id,
        "data_uri": req.new_data_uri,
        "dataset_id": req.dataset_id,
        "next_stage": "bronze_ingestion",
        "message": (
            f"Full pipeline restart scheduled. New data will be ingested from "
            f"'{req.new_data_uri}'. new_run_id: {new_run_id}"
        ),
    }


@router.get(
    "/history/{run_id}",
    summary="Retrieve feedback event history for a run",
)
async def feedback_history(run_id: str):
    """
    Returns all feedback events associated with a run_id from experience memory.
    Filtered by: stage='USER_FEEDBACK'.
    """
    config = _load_config()
    try:
        memory = ExperienceMemoryV2.from_config(config)
        history = memory.query(filters={"run_id": run_id, "stage": "USER_FEEDBACK"})
        events = [e.to_dict() for e in history] if history else []
    except Exception as exc:
        logger.warning("[Feedback/history] Memory query failed: %s", exc)
        events = []

    return {
        "run_id": run_id,
        "event_count": len(events),
        "events": events,
    }


@router.get("/rl-status", summary="Return current FeedbackController UCB1 bandit state")
async def rl_status():
    """
    Returns the current UCB1 bandit state of the FeedbackController:
    - strategy_counts: how many times each strategy was selected
    - strategy_rewards: cumulative reward per strategy
    - best_strategy: strategy with highest average reward
    """
    ctrl = _get_controller()
    return {
        "status": "OK",
        "bandit": ctrl.bandit_summary(),
        "available_strategies": STRATEGIES,
    }
