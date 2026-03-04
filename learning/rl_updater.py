"""
learning/rl_updater.py
-----------------------
Closes the feedback loop by translating Hard Gate 1 and verifier results
into a reward signal and updating the RL strategy Q-table.

Reward policy
-------------
+1.0 × confidence_score  — all gates passed with high confidence
+0.25 (fixed)            — gates passed but confidence < threshold (marginal pass)
−0.5 (fixed penalty)     — any gate failure (same magnitude as a hard-reject penalty)
"""

import logging
from typing import Optional

from proposal.rl_strategy import RLStrategyProposal

logger = logging.getLogger(__name__)

# Confidence threshold below which a "pass" is treated as a marginal win
_MARGINAL_CONFIDENCE_THRESHOLD = 0.60


class RLUpdater:
    """
    Translates pipeline outcome into a Q-value update for the RL strategy agent.

    Args:
        strategy_engine: The ``RLStrategyProposal`` instance that holds the
                         persistent Q-table to be updated.
        alpha:           Learning rate passed through to ``update_q_value``.
                         Defaults to 0.1 (conservative, stable convergence).
    """

    def __init__(
        self,
        strategy_engine: Optional[RLStrategyProposal] = None,
        alpha: float = 0.1,
    ) -> None:
        self.strategy_engine: RLStrategyProposal = (
            strategy_engine if strategy_engine is not None else RLStrategyProposal()
        )
        self.alpha = alpha

    def update(
        self,
        run_id: str,
        model_type: str,
        task: str,
        confidence: float,
        all_gates_passed: bool = True,
    ) -> float:
        """
        Computes a scalar reward and applies a Q-table update.

        Reward policy:
            - Hard failure (all_gates_passed=False) → reward = −0.5
            - Marginal pass (confidence < 0.60)     → reward = +0.25
            - Strong pass                            → reward = +confidence

        Args:
            run_id:           Pipeline run identifier (for audit logging).
            model_type:       Algorithm name (Q-table key).
            task:             Task type — "classification" or "regression".
            confidence:       Aggregated confidence score in [0, 1].
            all_gates_passed: False if any verifier gate failed (triggers penalty).

        Returns:
            The scalar reward that was applied.
        """
        if not all_gates_passed:
            reward = -0.5
            outcome = "GATE_FAILURE"
        elif confidence < _MARGINAL_CONFIDENCE_THRESHOLD:
            reward = 0.25
            outcome = "MARGINAL_PASS"
        else:
            reward = confidence
            outcome = "STRONG_PASS"

        logger.info(
            "RL feedback — run_id=%s  model=%s  task=%s  "
            "confidence=%.4f  outcome=%s  reward=%.3f",
            run_id, model_type, task, confidence, outcome, reward,
        )

        self.strategy_engine.update_q_value(
            task_type=task,
            algorithm=model_type,
            reward=reward,
            alpha=self.alpha,
        )
        return reward

    # Kept for backwards compatibility with any existing call sites
    def update_feedback(
        self,
        task_type: str,
        algorithm: str,
        confidence_score: float,
        all_passed: bool,
    ) -> None:
        """Deprecated: use ``update()`` instead."""
        self.update(
            run_id="legacy",
            model_type=algorithm,
            task=task_type,
            confidence=confidence_score,
            all_gates_passed=all_passed,
        )
