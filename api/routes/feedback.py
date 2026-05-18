"""
api/routes/feedback.py
-----------------------
POST /api/pipeline/feedback — Analyst satisfaction signal endpoint.

Records post-run user feedback (happy/unhappy) and translates it into RL reward
signals that are appended to the feedback log for PPO replay buffer training.

Endpoint contract:
  POST /api/pipeline/feedback
  Body (JSON):
    {
      "run_id":              "uuid",          # required
      "happy":               true,            # required — true = satisfied, false = not satisfied
      "reason":              "...",           # optional free text (max 500 chars)
      "plan_rejection_count": 0,             # optional — how many times user rejected plan before run
      "rerun_requested":     false            # optional — user wants pipeline re-executed
    }

  Response:
    {
      "recorded": true,
      "run_id":   "uuid",
      "rl_reward_delta": 0.20,
      "reward_components": { ... },
      "feedback_id": "uuid"
    }

Reward logic:
  - happy=True   → +0.20 reward (user_satisfied bonus)
  - happy=False  → −0.20 reward (user_dissatisfied penalty)
  - plan_rejection_count > 0 → −0.08 per rejection, capped at 3
  - Existing pipeline reward (from audit log) is loaded and the delta is combined

Storage:
  Feedback persisted to audit/feedback_log.jsonl — one JSON object per line.
  Format is compatible with the PPO replay buffer loader.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("dipex.api.feedback")
router = APIRouter(prefix="/api", tags=["Feedback"])

# ── Constants ─────────────────────────────────────────────────────────────────
_FEEDBACK_LOG = "audit/feedback_log.jsonl"
_AUDIT_LOG    = "audit/audit.jsonl"

# Reward weights (additive on top of pipeline execution reward)
W_USER_SATISFIED    =  0.20
W_USER_DISSATISFIED = -0.20
W_PLAN_REJECTED_PER = -0.08   # per rejection, max 3
W_PLAN_APPROVED     =  0.05   # if plan was approved before run


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_audit_entry(run_id: str) -> Optional[Dict[str, Any]]:
    """Load the pipeline run audit entry for a given run_id."""
    if not os.path.exists(_AUDIT_LOG):
        return None
    try:
        with open(_AUDIT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("run_id") == run_id:
                        return entry
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("Could not read audit log: %s", exc)
    return None


def _compute_reward_delta(
    happy: bool,
    plan_rejection_count: int,
    plan_was_approved: bool,
) -> Dict[str, float]:
    """
    Compute RL reward delta from feedback signals.

    Returns a dict of named reward components and their total.
    """
    components: Dict[str, float] = {}

    # User satisfaction signal
    if happy:
        components["user_satisfied"] = W_USER_SATISFIED
    else:
        components["user_dissatisfied"] = W_USER_DISSATISFIED

    # Plan rejection penalty (capped at 3 rejections)
    if plan_rejection_count > 0:
        rejection_penalty = W_PLAN_REJECTED_PER * min(plan_rejection_count, 3)
        components["plan_rejection_penalty"] = round(rejection_penalty, 4)

    # Plan approval bonus
    if plan_was_approved:
        components["plan_approval_bonus"] = W_PLAN_APPROVED

    total = float(np.clip(sum(components.values()), -1.0, 1.0))
    components["total_delta"] = round(total, 4)

    return components


def _append_feedback_log(entry: Dict[str, Any]) -> None:
    """Append a feedback entry to the JSONL feedback log."""
    os.makedirs("audit", exist_ok=True)
    try:
        with open(_FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.error("Failed to write feedback log: %s", exc)
        raise


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/pipeline/feedback",
    summary="Record analyst satisfaction feedback and compute RL reward delta",
    response_model=None,
)
async def pipeline_feedback(request: Request) -> Dict[str, Any]:
    """
    **Analyst Satisfaction Feedback Endpoint**

    Called from the frontend after the user views pipeline results and clicks
    👍 (happy) or 👎 (not satisfied / wants re-run).

    - Links feedback to the originating run via `run_id`
    - Retrieves pipeline metrics from the audit log
    - Computes RL reward delta and appends to feedback_log.jsonl
    - Returns the computed reward components for UI display

    The feedback log is structured for replay into the PPO trainer's
    experience buffer during periodic model retraining.
    """
    # ── Parse body ────────────────────────────────────────────────────────────
    try:
        body: Dict[str, Any] = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}") from exc

    run_id: str = (body.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=422, detail="'run_id' is required")

    happy: bool = bool(body.get("happy", True))
    reason: str = str(body.get("reason") or "")[:500].strip()
    plan_rejection_count: int = int(body.get("plan_rejection_count", 0))
    rerun_requested: bool = bool(body.get("rerun_requested", False))

    # ── Load pipeline run context from audit log ───────────────────────────────
    audit_entry = _load_audit_entry(run_id)
    plan_was_approved = bool((audit_entry or {}).get("plan_approved", False))
    dataset_id = (audit_entry or {}).get("dataset_id", "unknown")
    source_kind = (audit_entry or {}).get("source_kind", "unknown")
    gate_decision = (audit_entry or {}).get("gate_decision", "UNKNOWN")
    prior_reward = float((audit_entry or {}).get("confidence_score", 0.5))

    # ── Compute RL reward delta ───────────────────────────────────────────────
    reward_components = _compute_reward_delta(
        happy=happy,
        plan_rejection_count=plan_rejection_count,
        plan_was_approved=plan_was_approved,
    )
    total_delta = reward_components["total_delta"]
    combined_reward = float(np.clip(prior_reward + total_delta, 0.0, 1.0))

    # ── Build feedback record ─────────────────────────────────────────────────
    feedback_id = str(uuid.uuid4())
    feedback_entry = {
        # Identifiers
        "feedback_id": feedback_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "source_kind": source_kind,
        # Analyst signal
        "happy": happy,
        "reason": reason,
        "plan_rejection_count": plan_rejection_count,
        "rerun_requested": rerun_requested,
        "plan_was_approved": plan_was_approved,
        # RL signals
        "reward_delta": total_delta,
        "reward_components": reward_components,
        "combined_reward": combined_reward,
        "prior_pipeline_reward": prior_reward,
        # Pipeline context (for PPO replay)
        "gate_decision": gate_decision,
        "pipeline_context": {
            "quality_score":      (audit_entry or {}).get("quality_score", 0.0),
            "confidence_score":   prior_reward,
            "model_metrics":      (audit_entry or {}).get("model_metrics", {}),
            "retry_count":        (audit_entry or {}).get("retry_count", 0),
            "row_count":          (audit_entry or {}).get("row_count", 0),
            "col_count":          (audit_entry or {}).get("col_count", 0),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Schema marker for PPO replay buffer
        "schema_version": "feedback_v1",
    }

    # ── Persist feedback ──────────────────────────────────────────────────────
    try:
        _append_feedback_log(feedback_entry)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {exc}"
        ) from exc

    logger.info(
        "Feedback recorded — run_id=%s happy=%s delta=%.3f combined=%.3f",
        run_id, happy, total_delta, combined_reward,
    )

    return {
        "recorded": True,
        "feedback_id": feedback_id,
        "run_id": run_id,
        "happy": happy,
        "rl_reward_delta": total_delta,
        "combined_reward": combined_reward,
        "reward_components": reward_components,
        "rerun_requested": rerun_requested,
    }


@router.get(
    "/pipeline/feedback/summary",
    summary="Aggregate feedback statistics across all runs",
    response_model=None,
)
async def feedback_summary() -> Dict[str, Any]:
    """
    Returns aggregated satisfaction metrics from the feedback log.
    Used by the analytics dashboard to display analyst satisfaction trends.
    """
    if not os.path.exists(_FEEDBACK_LOG):
        return {
            "total_feedback": 0,
            "happy_count": 0,
            "unhappy_count": 0,
            "satisfaction_rate": 0.0,
            "avg_reward_delta": 0.0,
            "rerun_rate": 0.0,
            "plan_rejection_avg": 0.0,
        }

    entries = []
    try:
        with open(_FEEDBACK_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as exc:
        logger.warning("Could not read feedback log: %s", exc)
        return {"error": str(exc)}

    if not entries:
        return {"total_feedback": 0}

    happy_count = sum(1 for e in entries if e.get("happy", False))
    unhappy_count = len(entries) - happy_count
    avg_delta = sum(e.get("reward_delta", 0.0) for e in entries) / len(entries)
    rerun_count = sum(1 for e in entries if e.get("rerun_requested", False))
    avg_rejections = sum(e.get("plan_rejection_count", 0) for e in entries) / len(entries)

    return {
        "total_feedback": len(entries),
        "happy_count": happy_count,
        "unhappy_count": unhappy_count,
        "satisfaction_rate": round(happy_count / len(entries), 4),
        "avg_reward_delta": round(avg_delta, 4),
        "rerun_rate": round(rerun_count / len(entries), 4),
        "plan_rejection_avg": round(avg_rejections, 3),
    }
