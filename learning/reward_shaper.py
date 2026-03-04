"""
learning/reward_shaper.py
--------------------------
Enhancement 4: Multi-Dimensional Reward Shaping.

Replaces the raw confidence_score reward signal with a richer composite reward
that accounts for three dimensions:
  1. Confidence quality  (0.50 weight) — how good is the pipeline's confidence score
  2. Processing speed    (0.25 weight) — was this run fast relative to baseline?
  3. Drift improvement   (0.25 weight) — did drift go down compared to previous run?

This teaches the RL engine to not just chase high confidence, but also reward
efficient, fast, and stability-improving pipeline runs.

Design Principles:
  - All sub-rewards are normalised to [0, 1]
  - Weights are configurable via config.yaml `rl.reward_shaping.*`
  - Falls back gracefully if speed/drift signals are unavailable
  - Integrates with existing ReinforcementUpdateEngine and MetaRLEngine
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger("dipex.reward_shaper")

# Default reward weights (must sum to 1.0)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "confidence": 0.50,
    "speed":      0.25,
    "drift":      0.25,
}

# Speed baseline: runs faster than this (seconds) get full speed bonus
_SPEED_BASELINE_SECONDS: float = 60.0

# Drift improvement: PSI below this is considered "good drift"
_DRIFT_GOOD_PSI: float = 0.10


class RewardShaper:
    """
    Computes a multi-dimensional shaped reward from a pipeline run outcome.

    Usage:
        shaper = RewardShaper(config)
        reward = shaper.compute(
            confidence_score=0.91,
            elapsed_seconds=45.0,
            drift_psi_before=0.25,
            drift_psi_after=0.12,
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("rl", {}).get("reward_shaping", {})
        raw_weights = cfg.get("weights", _DEFAULT_WEIGHTS)

        # Normalise weights so they always sum to 1.0
        total = sum(raw_weights.values()) or 1.0
        self._weights: Dict[str, float] = {
            k: float(v) / total for k, v in raw_weights.items()
        }
        self._speed_baseline = float(cfg.get("speed_baseline_seconds", _SPEED_BASELINE_SECONDS))
        self._drift_good_psi = float(cfg.get("drift_good_psi", _DRIFT_GOOD_PSI))

        logger.info(
            "RewardShaper initialised: weights=%s speed_baseline=%.1fs drift_good_psi=%.2f",
            self._weights, self._speed_baseline, self._drift_good_psi,
        )

    def compute(
        self,
        confidence_score: float,
        elapsed_seconds: Optional[float] = None,
        drift_psi_before: Optional[float] = None,
        drift_psi_after: Optional[float] = None,
    ) -> float:
        """
        Compute the shaped reward in [0, 1].

        Parameters
        ----------
        confidence_score  : Overall pipeline confidence (0.0 – 1.0)
        elapsed_seconds   : Wall-clock time of the pipeline run (None → no speed bonus)
        drift_psi_before  : PSI drift score before this run (None → no drift bonus)
        drift_psi_after   : PSI drift score after this run  (None → no drift bonus)

        Returns
        -------
        float: Composite reward in [0.0, 1.0]
        """
        # 1. Confidence sub-reward (primary signal)
        r_conf = float(max(0.0, min(1.0, confidence_score)))

        # 2. Speed sub-reward (sigmoid curve — exponential penalty for slow runs)
        r_speed = self._compute_speed_reward(elapsed_seconds)

        # 3. Drift improvement sub-reward
        r_drift = self._compute_drift_reward(drift_psi_before, drift_psi_after)

        # Weighted composite
        w = self._weights
        reward = (
            w.get("confidence", 0.50) * r_conf
            + w.get("speed",      0.25) * r_speed
            + w.get("drift",      0.25) * r_drift
        )
        reward = float(max(0.0, min(1.0, reward)))

        logger.debug(
            "RewardShaper: conf=%.3f speed=%.3f drift=%.3f → composite=%.4f",
            r_conf, r_speed, r_drift, reward,
        )
        return reward

    def _compute_speed_reward(self, elapsed_seconds: Optional[float]) -> float:
        """Speed reward: 1.0 for instant, decays sigmoidally toward 0 for slow runs."""
        if elapsed_seconds is None:
            # No timing data — return neutral (neither reward nor penalty)
            return 0.5
        # Normalised: 0 seconds → 1.0, baseline seconds → ~0.5, 3x baseline → ~0.1
        t = max(0.0, float(elapsed_seconds))
        # Exponential decay: reward = exp(-k * t / baseline), k=ln(2) gives 0.5 at baseline
        k = math.log(2.0)
        return float(math.exp(-k * t / max(self._speed_baseline, 1.0)))

    def _compute_drift_reward(
        self,
        psi_before: Optional[float],
        psi_after: Optional[float],
    ) -> float:
        """
        Drift improvement reward.
          - psi went down → positive reward (drift reduced)
          - psi went up   → negative contribution (drift worsened)
          - psi already low → full reward
        """
        if psi_before is None or psi_after is None:
            return 0.5  # neutral when data unavailable

        psi_before = float(max(0.0, psi_before))
        psi_after  = float(max(0.0, psi_after))

        # Already at good drift → full reward
        if psi_after <= self._drift_good_psi:
            return 1.0

        # Improvement ratio: how much did drift reduce?
        if psi_before <= 0.0:
            return 0.5  # no baseline, neutral

        improvement = (psi_before - psi_after) / max(psi_before, 0.001)
        # improvement > 0 → drift reduced (good), improvement < 0 → drift worsened
        return float(max(0.0, min(1.0, 0.5 + 0.5 * improvement)))

    def decompose(
        self,
        confidence_score: float,
        elapsed_seconds: Optional[float] = None,
        drift_psi_before: Optional[float] = None,
        drift_psi_after: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return a breakdown dict of all sub-rewards for audit/logging."""
        return {
            "confidence_reward": float(max(0.0, min(1.0, confidence_score))),
            "speed_reward":      self._compute_speed_reward(elapsed_seconds),
            "drift_reward":      self._compute_drift_reward(drift_psi_before, drift_psi_after),
            "composite_reward":  self.compute(
                confidence_score=confidence_score,
                elapsed_seconds=elapsed_seconds,
                drift_psi_before=drift_psi_before,
                drift_psi_after=drift_psi_after,
            ),
            "weights": dict(self._weights),
        }
