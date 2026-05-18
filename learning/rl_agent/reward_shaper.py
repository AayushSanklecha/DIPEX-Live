"""
learning/rl_agent/reward_shaper.py
-------------------------------------
10-component shaped reward function for the PPO pipeline agent.

Base reward formula (from pipeline execution):
  R_base = (0.30 × data_health/100)
         + (0.25 × model_auc)
         + (0.20 × pipeline_success)
         - (0.15 × quarantine_frac)
         - (0.10 × min(drift_psi/0.25, 1.0))
         + (0.08 × user_approved_plan)
         - (0.07 × retry_count/3)

Analyst satisfaction delta (applied post-run from feedback):
  R_delta = (0.20 × user_satisfied)        # from POST /api/pipeline/feedback
          - (0.08 × plan_rejection_count)   # times user said "try again" on plan
          + (0.05 × instruction_followed)   # instructions were parsed and applied

All components are bounded so total R ∈ [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RewardComponents:
    # ── Base pipeline execution signals ───────────────────────────────────────
    data_health_bonus: float = 0.0
    model_auc_bonus: float = 0.0
    pipeline_success_bonus: float = 0.0
    quarantine_penalty: float = 0.0
    drift_penalty: float = 0.0
    user_approval_bonus: float = 0.0
    retry_penalty: float = 0.0
    # ── Analyst satisfaction signals (post-run from feedback endpoint) ────────
    user_satisfied_bonus: float = 0.0
    plan_rejection_penalty: float = 0.0
    instruction_followed_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "data_health_bonus":         round(self.data_health_bonus, 4),
            "model_auc_bonus":           round(self.model_auc_bonus, 4),
            "pipeline_success_bonus":    round(self.pipeline_success_bonus, 4),
            "quarantine_penalty":        round(self.quarantine_penalty, 4),
            "drift_penalty":             round(self.drift_penalty, 4),
            "user_approval_bonus":       round(self.user_approval_bonus, 4),
            "retry_penalty":             round(self.retry_penalty, 4),
            # Analyst satisfaction
            "user_satisfied_bonus":      round(self.user_satisfied_bonus, 4),
            "plan_rejection_penalty":    round(self.plan_rejection_penalty, 4),
            "instruction_followed_bonus": round(self.instruction_followed_bonus, 4),
            "total":                     round(self.total, 4),
        }


class RewardShaper:
    """
    Computes a shaped, continuous reward ∈ [0, 1] from pipeline run metrics,
    extended with analyst satisfaction signals from the feedback loop.

    Weights designed to:
      - Reward healthy data (data_health is the strongest positive signal)
      - Reward good model performance (AUC is second strongest)
      - Penalize data loss (quarantine is the strongest penalty)
      - Penalize instability (drift, repeated retries)
      - Reward analyst satisfaction (+0.20 for happy, −0.20 for unhappy)
    """

    # ── Base pipeline weights ──────────────────────────────────────────────────
    W_HEALTH    = 0.30
    W_AUC       = 0.25
    W_SUCCESS   = 0.20
    W_QUARANT   = 0.15
    W_DRIFT     = 0.10
    W_APPROVAL  = 0.08
    W_RETRY     = 0.07
    # ── Analyst satisfaction weights ───────────────────────────────────────
    W_SATISFIED     = 0.20    # user clicked 👍 satisfied
    W_REJECTION     = 0.08    # per-plan-rejection penalty
    W_INSTR_APPLIED = 0.05    # instructions were parsed and applied

    def compute(
        self,
        data_health: float = 50.0,
        model_auc: float = 0.5,
        pipeline_success: bool = True,
        quarantine_frac: float = 0.0,
        drift_psi: Optional[float] = None,
        user_approved_plan: bool = False,
        retry_count: int = 0,
        # ── Analyst satisfaction signals (optional, provided post-run) ───────
        user_satisfied: Optional[bool] = None,
        plan_rejection_count: int = 0,
        instruction_followed: bool = False,
    ) -> RewardComponents:
        """
        Compute shaped reward from pipeline run outcome.

        Parameters
        ----------
        data_health          : AnalystBrain health score 0-100
        model_auc            : model ROC-AUC 0-1 (0.5 if no model)
        pipeline_success     : True if pipeline completed without FAIL gate
        quarantine_frac      : fraction of rows quarantined (0-1)
        drift_psi            : Population Stability Index (None = no drift detected)
        user_approved_plan   : True if user explicitly approved the pre-analysis plan
        retry_count          : number of pipeline retries (0-3)
        user_satisfied       : True=happy, False=unhappy, None=no feedback yet
        plan_rejection_count : number of times user rejected the plan before approving
        instruction_followed : True if user instructions were successfully parsed and applied

        Returns
        -------
        RewardComponents with summed total clipped to [0, 1]
        """
        comp = RewardComponents()

        # ── Base pipeline signals ───────────────────────────────────────────────
        comp.data_health_bonus      = self.W_HEALTH  * float(np.clip(data_health / 100.0, 0, 1))
        comp.model_auc_bonus        = self.W_AUC     * float(np.clip(model_auc, 0, 1))
        comp.pipeline_success_bonus = self.W_SUCCESS * (1.0 if pipeline_success else 0.0)
        comp.user_approval_bonus    = self.W_APPROVAL * (1.0 if user_approved_plan else 0.0)

        # Penalties
        comp.quarantine_penalty = self.W_QUARANT * float(np.clip(quarantine_frac, 0, 1))
        drift_norm              = float(np.clip((drift_psi or 0.0) / 0.25, 0, 1))
        comp.drift_penalty      = self.W_DRIFT   * drift_norm
        comp.retry_penalty      = self.W_RETRY   * float(np.clip(retry_count / 3.0, 0, 1))

        # ── Analyst satisfaction signals ─────────────────────────────────────────
        if user_satisfied is True:
            comp.user_satisfied_bonus = self.W_SATISFIED          # full bonus
        elif user_satisfied is False:
            comp.user_satisfied_bonus = -self.W_SATISFIED         # symmetric penalty
        # else None → no feedback received yet, no change

        if plan_rejection_count > 0:
            rejection_penalty = self.W_REJECTION * min(plan_rejection_count, 3)
            comp.plan_rejection_penalty = rejection_penalty        # stored as positive, subtracted below

        if instruction_followed:
            comp.instruction_followed_bonus = self.W_INSTR_APPLIED

        # ── Total (clipped to [0, 1]) ────────────────────────────────────────────────
        raw = (
            comp.data_health_bonus
            + comp.model_auc_bonus
            + comp.pipeline_success_bonus
            + comp.user_approval_bonus
            + comp.user_satisfied_bonus          # ±0.20
            + comp.instruction_followed_bonus    # +0.05
            - comp.quarantine_penalty
            - comp.drift_penalty
            - comp.retry_penalty
            - comp.plan_rejection_penalty        # −0.08 per rejection
        )
        comp.total = float(np.clip(raw, 0.0, 1.0))
        return comp

    def from_pipeline_result(
        self,
        result_summary: dict,
        analytics: dict,
        user_satisfied: Optional[bool] = None,
        plan_rejection_count: int = 0,
        instruction_followed: bool = False,
    ) -> RewardComponents:
        """
        Convenience factory: extract metrics from PipelineResult.summary() dict.
        Accepts optional analyst satisfaction signals from the feedback loop.
        """
        data_health   = float(analytics.get("data_health_score", 50.0))
        model_metrics = result_summary.get("model_metrics") or {}
        auc           = float(model_metrics.get("roc_auc", model_metrics.get("auc", 0.5)) or 0.5)
        success       = result_summary.get("gate_decision", "FAIL") in ("PASS", "WARN")
        q_rows        = int(result_summary.get("quarantine_rows", 0))
        retry_count   = int(result_summary.get("retry_count", 0))
        quarantine_frac = 0.0  # conservative default when row total unknown

        return self.compute(
            data_health=data_health,
            model_auc=auc,
            pipeline_success=success,
            quarantine_frac=quarantine_frac,
            retry_count=retry_count,
            user_satisfied=user_satisfied,
            plan_rejection_count=plan_rejection_count,
            instruction_followed=instruction_followed,
        )
