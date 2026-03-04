"""
modeling/rl_orchestrator.py
-----------------------------
Production-grade Top-Level RL Orchestrator for DIPEX.

Purpose
-------
Allocates compute budgets across pipeline stages to meet SLA targets.

Architecture
------------
State  : (sla_bucket, size_bucket, drift_level)
           sla_bucket  : "critical" | "standard" | "batch"
           size_bucket : "small" | "medium" | "large"
           drift_level : "none" | "moderate" | "severe"

Actions: (profile_depth, preprocess_depth, model_complexity, validation_depth)
           profile_depth    : "basic" | "full"
           preprocess_depth : "fast" | "thorough"
           model_complexity : "light" | "balanced" | "heavy"
           validation_depth : "quick" | "full"

Reward Function
---------------
  beat_sla (ran within budget)          → +2.0
  quality above threshold               → +3.0 × quality_score
  overtime (exceeded SLA)              → -10.0
  quality below threshold               → -5.0 × (threshold - quality_score)

Persistence: data/rl_orchestrator.json

Usage
-----
    from modeling.rl_orchestrator import get_rl_orchestrator

    orch = get_rl_orchestrator()
    plan = orch.get_plan(sla_minutes=10, row_count=50000, drift_detected=False)
    # plan = {
    #   "profile_depth": "basic",
    #   "preprocess_depth": "fast",
    #   "model_complexity": "light",
    #   "validation_depth": "quick",
    #   "state": "...",
    #   "action_key": "...",
    # }

    # After pipeline completes:
    orch.record_outcome(plan, actual_minutes=8.5, quality_score=0.91, sla_minutes=10)
"""

from __future__ import annotations

import json
import logging
import os
import random
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.modeling.rl_orchestrator")

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_orchestrator.json"
)

# ── Action space ──────────────────────────────────────────────────────────────

_PROFILE_OPTIONS     = ["basic", "full"]
_PREPROCESS_OPTIONS  = ["fast", "thorough"]
_MODEL_OPTIONS       = ["light", "balanced", "heavy"]
_VALIDATION_OPTIONS  = ["quick", "full"]

_ALL_ACTIONS: List[Tuple[str, str, str, str]] = [
    (p, pp, m, v)
    for p, pp, m, v in product(
        _PROFILE_OPTIONS,
        _PREPROCESS_OPTIONS,
        _MODEL_OPTIONS,
        _VALIDATION_OPTIONS,
    )
]  # 2×2×3×2 = 24 actions

_ALPHA:     float = 0.12
_EPSILON:   float = 0.12
_SAVE_PROB: float = 0.10

_QUALITY_THRESHOLD: float = 0.75


def _action_key(action: Tuple[str, str, str, str]) -> str:
    return "::".join(action)


def _parse_action_key(key: str) -> Tuple[str, str, str, str]:
    parts = key.split("::")
    return (parts[0], parts[1], parts[2], parts[3])  # type: ignore[return-value]


def _state_key(sla_minutes: float, row_count: int, drift_detected: bool) -> str:
    sla_b  = "critical" if sla_minutes <= 5 else ("standard" if sla_minutes <= 30 else "batch")
    size_b = "small" if row_count < 10_000 else ("medium" if row_count < 200_000 else "large")
    drift  = "severe" if drift_detected else "none"
    return f"{sla_b}::{size_b}::{drift}"


# ── Sensible defaults per state ───────────────────────────────────────────────

_STATE_PRIORS: Dict[str, Tuple[str, str, str, str]] = {
    # critical SLA → fastest possible
    "critical::small::none":     ("basic", "fast", "light",    "quick"),
    "critical::medium::none":    ("basic", "fast", "light",    "quick"),
    "critical::large::none":     ("basic", "fast", "light",    "quick"),
    "critical::small::severe":   ("basic", "fast", "balanced", "quick"),
    "critical::medium::severe":  ("basic", "fast", "balanced", "quick"),
    "critical::large::severe":   ("basic", "fast", "light",    "quick"),
    # standard SLA → balanced
    "standard::small::none":     ("full",  "thorough", "balanced", "full"),
    "standard::medium::none":    ("full",  "fast",     "balanced", "full"),
    "standard::large::none":     ("basic", "fast",     "balanced", "quick"),
    "standard::small::severe":   ("full",  "thorough", "heavy",    "full"),
    "standard::medium::severe":  ("full",  "thorough", "balanced", "full"),
    "standard::large::severe":   ("full",  "fast",     "balanced", "quick"),
    # batch SLA → full quality
    "batch::small::none":        ("full",  "thorough", "heavy",    "full"),
    "batch::medium::none":       ("full",  "thorough", "heavy",    "full"),
    "batch::large::none":        ("full",  "thorough", "balanced", "full"),
    "batch::small::severe":      ("full",  "thorough", "heavy",    "full"),
    "batch::medium::severe":     ("full",  "thorough", "heavy",    "full"),
    "batch::large::severe":      ("full",  "thorough", "heavy",    "full"),
}


class RLOrchestrator:
    """
    Q-learning orchestrator that learns the optimal compute-budget
    allocation per pipeline-run context.
    """

    def __init__(self, state_path: str = _DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.q: Dict[str, Dict[str, float]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    self.q = json.load(fh)
                logger.info("RLOrchestrator: loaded %d states.", len(self.q))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLOrchestrator: Q-table load failed: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.q, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLOrchestrator: save failed: %s", exc)

    def _init_state(self, state: str) -> None:
        if state not in self.q:
            self.q[state] = {_action_key(a): 0.0 for a in _ALL_ACTIONS}
            # Seed from domain priors for faster cold-start
            prior_action = _STATE_PRIORS.get(state)
            if prior_action:
                self.q[state][_action_key(prior_action)] = 2.0

    def _update(self, state: str, action_key: str, reward: float) -> None:
        self._init_state(state)
        curr = self.q[state].get(action_key, 0.0)
        self.q[state][action_key] = round(curr + _ALPHA * (reward - curr), 6)
        if random.random() < _SAVE_PROB:
            self.save()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_plan(
        self,
        sla_minutes:    float = 30.0,
        row_count:      int   = 10_000,
        drift_detected: bool  = False,
    ) -> Dict[str, Any]:
        """
        Return the RL-selected orchestration plan for this pipeline run.

        Returns
        -------
        {
          "profile_depth":     str,
          "preprocess_depth":  str,
          "model_complexity":  str,
          "validation_depth":  str,
          "state":             str,
          "action_key":        str,
        }
        """
        state = _state_key(sla_minutes, row_count, drift_detected)
        self._init_state(state)

        if random.random() < _EPSILON:
            action = random.choice(_ALL_ACTIONS)
            logger.debug("[RL] Orchestrator exploring: %s | state=%s", action, state)
        else:
            best_key = max(self.q[state], key=self.q[state].__getitem__)
            action   = _parse_action_key(best_key)
            logger.debug(
                "[RL] Orchestrator exploiting: %s | state=%s | Q=%.4f",
                action, state, self.q[state][best_key],
            )

        akey = _action_key(action)
        return {
            "profile_depth":    action[0],
            "preprocess_depth": action[1],
            "model_complexity": action[2],
            "validation_depth": action[3],
            "state":            state,
            "action_key":       akey,
        }

    def record_outcome(
        self,
        plan:          Dict[str, Any],
        actual_minutes: float,
        quality_score: float,
        sla_minutes:   float,
    ) -> None:
        """
        Update Q-table with the observed pipeline run outcome.

        Parameters
        ----------
        actual_minutes : Wall-clock duration of the pipeline run
        quality_score  : Downstream model quality (AUC / R², 0-1)
        sla_minutes    : The SLA budget (as passed to get_plan)
        """
        state      = plan.get("state", "unknown")
        action_key = plan.get("action_key", "")

        # Reward components
        sla_reward = 2.0 if actual_minutes <= sla_minutes else -10.0
        quality_reward = (
            3.0 * quality_score
            if quality_score >= _QUALITY_THRESHOLD
            else -5.0 * (_QUALITY_THRESHOLD - quality_score)
        )
        reward = sla_reward + quality_reward

        self._update(state, action_key, reward)
        logger.info(
            "[RL] Orchestrator outcome: state=%s | action=%s | "
            "actual=%.1fm | sla=%.1fm | quality=%.4f | reward=%.2f",
            state, action_key, actual_minutes, sla_minutes, quality_score, reward,
        )

    def get_policy_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return {state: {best_action, Q_value}} for all learned states."""
        return {
            s: {
                "best_action": max(v, key=v.__getitem__),
                "Q":           v[max(v, key=v.__getitem__)],
            }
            for s, v in self.q.items() if v
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_ORCHESTRATOR: Optional[RLOrchestrator] = None


def get_rl_orchestrator() -> RLOrchestrator:
    """Return the module-level singleton RLOrchestrator."""
    global _ORCHESTRATOR  # noqa: PLW0603
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = RLOrchestrator()
    return _ORCHESTRATOR
