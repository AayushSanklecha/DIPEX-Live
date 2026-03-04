"""
validation/rl_threshold_tuner.py
----------------------------------
Production-grade RL Bandit for Dynamic Validation Threshold Adaptation.

Architecture
------------
State   : "{dataset_id}::{column}"
Actions : null/range tolerance thresholds {0.01, 0.05, 0.10, 0.20, 0.30}

The agent learns which tolerance level leads to the best downstream outcomes
(model F1 / pipeline quality score) without letting bad data through.

Reward Function
---------------
  validation_pass + downstream_success  →  +1.0
  validation_pass + downstream_fail     →  -10.0  (let bad data through!)
  validation_fail (correctly blocked)   →  +0.5
  validation_fail (false-negative)      →  -1.0

The Q-table is persisted between runs so the agent improves continuously.

Usage
-----
    from validation.rl_threshold_tuner import get_rl_tuner

    tuner = get_rl_tuner()
    threshold = tuner.get_threshold("sales_data", "revenue", default=0.10)
    # Use threshold in NullValidator / RangeValidator

    # After downstream model evaluation completes:
    tuner.record_outcome("sales_data", "revenue", threshold,
                         validation_passed=True, downstream_success=True)
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional

logger = logging.getLogger("dipex.validation.rl_threshold_tuner")

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_validation_thresholds.json"
)
_ACTIONS:     List[float] = [0.01, 0.05, 0.10, 0.20, 0.30]
_ALPHA:       float       = 0.15   # learning rate (slightly higher — fast adaptation)
_EPSILON:     float       = 0.12   # exploration rate
_SAVE_PROB:   float       = 0.15


def _ak(threshold: float) -> str:
    """
    Canonical action key — always uses Python's repr to avoid
    '0.10' vs '0.1' float-to-str inconsistency.
    """
    return repr(round(threshold, 6))


class RLThresholdTuner:
    """
    Multi-armed bandit agent that dynamically adjusts validation thresholds
    per (dataset, column) pair to optimise downstream model quality.
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
                logger.info("RLThresholdTuner: Q-table loaded (%d states).", len(self.q))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLThresholdTuner: Q-table load failed: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.q, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLThresholdTuner: Save failed: %s", exc)

    # ── Q-table helpers ───────────────────────────────────────────────────────

    def _state_key(self, dataset_id: str, column: str) -> str:
        return f"{dataset_id}::{column}"

    def _init_state(self, state: str, default_threshold: float) -> None:
        if state not in self.q:
            self.q[state] = {_ak(a): 0.0 for a in _ACTIONS}
            # Seed closest action to config default with small positive prior
            closest = min(_ACTIONS, key=lambda x: abs(x - default_threshold))
            self.q[state][_ak(closest)] = 0.5

    def _update(self, state: str, action_key: str, reward: float) -> None:
        if state not in self.q:
            self._init_state(state, float(action_key))
        if action_key not in self.q[state]:
            self.q[state][action_key] = 0.0
        curr = self.q[state][action_key]
        self.q[state][action_key] = round(curr + _ALPHA * (reward - curr), 6)
        if random.random() < _SAVE_PROB:
            self.save()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_threshold(
        self,
        dataset_id:        str,
        column:            str,
        default:           float = 0.10,
        force_exploration: bool  = False,
    ) -> float:
        """
        Return the RL-selected threshold for a (dataset, column) pair.

        Parameters
        ----------
        dataset_id : Stable dataset identifier
        column     : Column name
        default    : Config-file default (used to seed the Q-table)
        force_exploration : If True, always explore (useful for new datasets)

        Returns
        -------
        float : Threshold value from _ACTIONS
        """
        state = self._state_key(dataset_id, column)
        self._init_state(state, default)

        if force_exploration or random.random() < _EPSILON:
            chosen_f = random.choice(_ACTIONS)
        else:
            best_key  = max(self.q[state], key=self.q[state].__getitem__)
            chosen_f  = float(best_key)  # repr key is directly parseable

        logger.debug("[RL] Threshold for %s::%s → %.4f", dataset_id, column, chosen_f)
        return chosen_f

    def record_outcome(
        self,
        dataset_id:         str,
        column:             str,
        threshold:          float,
        validation_passed:  bool,
        downstream_success: bool = True,
    ) -> None:
        """
        Update the Q-table with the observed outcome of using `threshold`.

        Parameters
        ----------
        validation_passed  : Did the column pass validation with this threshold?
        downstream_success : Did the downstream ML model succeed after this run?
        """
        state      = self._state_key(dataset_id, column)
        action_key = _ak(threshold)

        if validation_passed and downstream_success:
            reward =  1.0
        elif validation_passed and not downstream_success:
            reward = -10.0   # False positive: bad data slipped through
        elif not validation_passed and not downstream_success:
            reward =  0.5    # True negative: correctly blocked
        else:
            reward = -1.0    # False negative: blocked good data

        self._update(state, action_key, reward)
        logger.debug(
            "[RL] Threshold outcome: %s::%s | thresh=%.4f | pass=%s | ds_ok=%s | reward=%.1f",
            dataset_id, column, threshold, validation_passed, downstream_success, reward,
        )

    def get_policy_summary(self) -> Dict[str, str]:
        """Return {state: best_threshold} for all learned states."""
        return {
            s: max(v, key=v.__getitem__)
            for s, v in self.q.items() if v
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_TUNER: Optional[RLThresholdTuner] = None


def get_rl_tuner() -> RLThresholdTuner:
    """Return the module-level singleton RLThresholdTuner."""
    global _TUNER  # noqa: PLW0603
    if _TUNER is None:
        _TUNER = RLThresholdTuner()
    return _TUNER
