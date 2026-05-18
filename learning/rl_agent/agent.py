"""
learning/rl_agent/agent.py
----------------------------
PPOAgent: Main public API for the RL pipeline strategy agent.

Implements the Plan-specified behavior:
  - Shadow mode for first 20 episodes (Thompson Sampling fallback)
  - PPO takes over after 20 real episodes are collected
  - Safety constraints enforced on all actions
  - Rollback if reward drops > 20% over 5 runs
  - State persisted to models/rl_ppo_policy.pkl between runs

Usage::
    agent = PPOAgent.from_config(config)
    action = agent.recommend(pipeline_context)
    # ... run pipeline ...
    agent.record_outcome(result_summary, analytics)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .action_space import ActionSpace, PipelineAction
from .policy_network import PolicyNetwork
from .replay_buffer import ReplayBuffer
from .reward_shaper import RewardShaper
from .state_encoder import StateEncoder
from .value_network import ValueNetwork

logger = logging.getLogger("dipex.learning.rl_agent.agent")

# Threshold for switching from Thompson Sampling to PPO
SHADOW_MODE_EPISODES = 20

# Rollback threshold: if reward drops > 20% over 5 runs, revert weights
ROLLBACK_DROP_THRESHOLD = 0.20
ROLLBACK_WINDOW = 5


class PPOAgent:
    """
    PPO Actor-Critic agent for ADAP pipeline strategy optimization.

    Phase 1 (< 20 episodes): Shadow mode — uses Thompson Sampling from the
    existing ReinforcementUpdateEngine. Observations are recorded but PPO
    weights are not updated.

    Phase 2 (>= 20 episodes): PPO takes over. Actor selects 8-axis action,
    Critic provides V(s) for GAE advantage estimation.

    Safety: Actions are always constraint-checked. If reward drops by >20%
    over 5 runs, weights are reverted to last good checkpoint.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config  = config or {}
        self.policy  = PolicyNetwork()
        self.value   = ValueNetwork()
        self.encoder = StateEncoder()
        self.action_space = ActionSpace()
        self.reward_shaper = RewardShaper()
        self.buffer  = ReplayBuffer(capacity=1000)

        # Attempt to load existing checkpoint
        self._loaded = self.policy.load() and self.value.load()
        self._episode_count = self.policy._episode_count

        # Rollback tracking
        self._recent_rewards: List[float] = []
        self._best_reward: float = -1.0
        self._best_policy_weights: Optional[dict] = None

        # Current episode state
        self._current_state: Optional[np.ndarray] = None
        self._current_action_indices: Optional[List[int]] = None
        self._current_log_prob: float = 0.0
        self._current_value: float    = 0.0

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PPOAgent":
        return cls(config=config)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def in_shadow_mode(self) -> bool:
        """True while PPO is still collecting seed data (< 20 episodes)."""
        return self._episode_count < SHADOW_MODE_EPISODES

    def recommend(
        self,
        pipeline_context: Dict[str, Any],
        greedy: bool = False,
    ) -> PipelineAction:
        """
        Recommend the next pipeline action given the current context.

        In shadow mode, delegates to Thompson Sampling and just records state.
        In PPO mode, uses the actor network to select the action.

        Parameters
        ----------
        pipeline_context: dict with keys:
            n_rows, n_cols, null_rate, anomaly_rate, drift_psi,
            data_health, domain, target_col, prior_confidence,
            quarantine_frac, retry_count
        greedy: if True, take the argmax action (for evaluation)

        Returns
        -------
        PipelineAction with all 8 axes filled
        """
        state = self.encoder.encode(pipeline_context)
        self._current_state = state

        if self.in_shadow_mode:
            # Shadow mode: use defaults + minor randomization
            logger.info(
                "[PPOAgent] Shadow mode (%d/%d episodes) — using default action.",
                self._episode_count, SHADOW_MODE_EPISODES
            )
            indices, action = self.action_space.default_action()
            self._current_action_indices = indices
            self._current_log_prob = self.policy.log_prob(state, indices)
            self._current_value    = self.value.forward(state)
            return action

        # PPO mode: sample from actor distribution
        indices, probs = self.policy.sample_action(state, greedy=greedy)
        indices = self.action_space.constrained_indices(indices)
        action  = self.action_space.decode(indices)

        self._current_action_indices = indices
        self._current_log_prob = self.policy.log_prob(state, indices)
        self._current_value    = self.value.forward(state)

        logger.info(
            "[PPOAgent] Episode %d action: cv=%s/%s conf=%.2f imp=%s outl=%s complexity=%s retries=%d",
            self._episode_count,
            action.cv_folds, action.cv_strategy,
            action.confidence_threshold, action.imputation,
            action.outlier_policy, action.model_complexity, action.retry_budget,
        )
        return action

    def record_outcome(
        self,
        result_summary: Dict[str, Any],
        analytics: Dict[str, Any],
        user_approved_plan: bool = False,
    ) -> Dict[str, Any]:
        """
        Record the outcome of a pipeline run and update the agent.

        Parameters
        ----------
        result_summary : PipelineResult.summary() dict
        analytics      : AnalyticsResult.to_dict() dict
        user_approved_plan : whether user approved the pre-analysis plan

        Returns
        -------
        dict with episode_count, reward, in_shadow_mode, training_metrics
        """
        if self._current_state is None or self._current_action_indices is None:
            logger.warning("[PPOAgent] record_outcome() called without prior recommend() — skipping.")
            return {"error": "recommend() must be called before record_outcome()"}

        # Compute reward
        data_health   = float(analytics.get("data_health_score", 50.0))
        model_metrics = result_summary.get("model_metrics") or {}
        auc = float(model_metrics.get("roc_auc", model_metrics.get("auc", 0.5)) or 0.5)
        success = result_summary.get("gate_decision", "FAIL") in ("PASS", "WARN")
        q_rows  = int(result_summary.get("quarantine_rows", 0))
        retry   = int(result_summary.get("retry_count", 0))

        reward_components = self.reward_shaper.compute(
            data_health=data_health,
            model_auc=auc,
            pipeline_success=success,
            quarantine_frac=min(q_rows / max(q_rows + 100, 1), 1.0),
            user_approved_plan=user_approved_plan,
            retry_count=retry,
        )
        reward = reward_components.total

        # Store transition
        self.buffer.add_transition(
            state          = self._current_state,
            action_indices = self._current_action_indices,
            log_prob       = self._current_log_prob,
            reward         = reward,
            value          = self._current_value,
            done           = True,
        )

        self._episode_count += 1
        self.policy._episode_count = self._episode_count

        # Track rewards for rollback detection
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > ROLLBACK_WINDOW:
            self._recent_rewards.pop(0)

        training_metrics: Dict[str, Any] = {}

        # Update PPO weights once out of shadow mode.
        # Minimum 32 transitions for stable on-policy gradient estimates.
        # (PPO on 4 transitions produces extremely high-variance updates.)
        if not self.in_shadow_mode and self.buffer.n_transitions >= 32:
            try:
                from .ppo_trainer import PPOTrainer
                trainer = PPOTrainer(self.policy, self.value)
                training_metrics = trainer.update(self.buffer)
                self.buffer.clear()

                # Rollback check
                if self._should_rollback(reward):
                    self._rollback()
                else:
                    if reward > self._best_reward:
                        self._best_reward = reward
                        self._save_best_checkpoint()

            except Exception as exc:
                logger.warning("[PPOAgent] PPO update failed: %s", exc)

        # Save updated checkpoint
        try:
            self.policy.save()
            self.value.save()
        except Exception as exc:
            logger.warning("[PPOAgent] Checkpoint save failed: %s", exc)

        # Reset state
        self._current_state          = None
        self._current_action_indices = None

        summary = {
            "episode_count":    self._episode_count,
            "reward":           round(reward, 4),
            "in_shadow_mode":   self.in_shadow_mode,
            "training_metrics": training_metrics,
            "reward_components": reward_components.to_dict(),
        }
        logger.info(
            "[PPOAgent] Episode %d done — reward=%.4f shadow=%s",
            self._episode_count, reward, self.in_shadow_mode,
        )
        return summary

    def get_current_recommendation_summary(self) -> Dict[str, Any]:
        """Return a summary of the current best policy (greedy)."""
        dummy_context = {
            "n_rows": 10000, "n_cols": 20,
            "null_rate": 0.05, "anomaly_rate": 0.02,
            "drift_psi": 0.1, "data_health": 75.0, "domain": "generic",
        }
        state = self.encoder.encode(dummy_context)
        indices, _ = self.policy.sample_action(state, greedy=True)
        action = self.action_space.decode(indices)
        return {
            "in_shadow_mode": self.in_shadow_mode,
            "episode_count": self._episode_count,
            "recommended_action": action.to_dict(),
        }

    # ── Rollback logic ────────────────────────────────────────────────────────

    def _should_rollback(self, current_reward: float) -> bool:
        """Check if reward dropped > 20% compared to best."""
        if self._best_reward <= 0 or len(self._recent_rewards) < ROLLBACK_WINDOW:
            return False
        recent_mean = float(np.mean(self._recent_rewards))
        drop = (self._best_reward - recent_mean) / max(self._best_reward, 1e-8)
        return drop > ROLLBACK_DROP_THRESHOLD

    def _save_best_checkpoint(self) -> None:
        """Save current weights as best checkpoint."""
        import copy
        self._best_policy_weights = {
            "W1": self.policy.W1.copy(), "b1": self.policy.b1.copy(),
            "W2": self.policy.W2.copy(), "b2": self.policy.b2.copy(),
            "heads": [(W.copy(), b.copy()) for W, b in self.policy.heads],
        }

    def _rollback(self) -> None:
        """Revert policy to best checkpoint."""
        if self._best_policy_weights is None:
            return
        w = self._best_policy_weights
        self.policy.W1 = w["W1"].copy()
        self.policy.b1 = w["b1"].copy()
        self.policy.W2 = w["W2"].copy()
        self.policy.b2 = w["b2"].copy()
        self.policy.heads = [(W.copy(), b.copy()) for W, b in w["heads"]]
        logger.warning("[PPOAgent] Rollback triggered — reverted to best checkpoint (reward=%.4f).",
                       self._best_reward)
        self._recent_rewards.clear()
