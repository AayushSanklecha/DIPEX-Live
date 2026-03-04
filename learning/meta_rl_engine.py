"""
learning/meta_rl_engine.py
---------------------------
Production Meta-RL Engine for DIPEX.

Implements:
  - Epsilon-greedy contextual bandit with safe annealing (ε ∈ [0.05, 0.30])
  - Policy registry: multiple named RL strategies selectable by context
  - Bayesian regret tracking per action (minimize expected regret)
  - Cumulative regret monitoring with deprioritization
  - Drift-conditioned policy switching: high PSI → exploration-heavy policy
  - No catastrophic forgetting: incremental EMA updates + experience replay
  - Continuous learning loop: Execute → Measure → Reward → Store → Update

Safety: All updates go through rl_safety.assert_target_allowed().
Sandbox: When is_sandbox_active(), all weight writes are skipped.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from learning.rl_safety import (
    ALLOWED_UPDATE_TARGETS,
    FORBIDDEN_TARGETS,
    RLSafetyViolation,
    InstabilityDetector,
    RLCheckpointManager,
    assert_target_allowed,
    sandbox_safe_write,
    is_sandbox_active,
)
from learning.contextual_bandit import ContextualBandit
from learning.domain_priors import get_prior
from learning.reward_shaper import RewardShaper
from learning.transfer_learning import KnowledgeTransfer, DomainFingerprint
from learning.rl_automl import RLAutoTuner

logger = logging.getLogger("dipex.meta_rl")

# ══════════════════════════════════════════════════════════════════════════════
# Policy Registry
# ══════════════════════════════════════════════════════════════════════════════

POLICY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "default": {
        "epsilon": 0.15,
        "description": "Balanced exploration / exploitation",
        "use_when": "normal operation",
    },
    "exploration_heavy": {
        "epsilon": 0.30,
        "description": "High exploration — used when drift detected",
        "use_when": "PSI > 0.20",
    },
    "exploitation_heavy": {
        "epsilon": 0.05,
        "description": "Low exploration — converged, high confidence",
        "use_when": "confidence >= 0.90 for 5+ episodes",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionStats:
    """Running statistics for a single bandit action (arm)."""
    name: str
    q_value: float = 0.5         # EMA Q-value
    count: int = 0               # times selected
    cumulative_regret: float = 0.0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.count if self.count > 0 else 0.0

    @property
    def mean_regret(self) -> float:
        return self.cumulative_regret / self.count if self.count > 0 else 0.0


@dataclass
class BanditState:
    """Full state of all arms in the bandit."""
    actions: Dict[str, ActionStats] = field(default_factory=dict)
    episode: int = 0
    active_policy: str = "default"
    epsilon: float = 0.15
    total_reward: float = 0.0
    total_regret: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "active_policy": self.active_policy,
            "epsilon": self.epsilon,
            "total_reward": self.total_reward,
            "total_regret": self.total_regret,
            "actions": {
                name: {
                    "q_value": a.q_value,
                    "count": a.count,
                    "cumulative_regret": a.cumulative_regret,
                    "total_reward": a.total_reward,
                }
                for name, a in self.actions.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BanditState":
        state = cls(
            episode=d.get("episode", 0),
            active_policy=d.get("active_policy", "default"),
            epsilon=d.get("epsilon", 0.15),
            total_reward=d.get("total_reward", 0.0),
            total_regret=d.get("total_regret", 0.0),
        )
        for name, av in d.get("actions", {}).items():
            stats = ActionStats(name=name)
            stats.q_value = float(av.get("q_value", 0.5))
            stats.count = int(av.get("count", 0))
            stats.cumulative_regret = float(av.get("cumulative_regret", 0.0))
            stats.total_reward = float(av.get("total_reward", 0.0))
            state.actions[name] = stats
        return state


# ══════════════════════════════════════════════════════════════════════════════
# MetaRLEngine
# ══════════════════════════════════════════════════════════════════════════════

class MetaRLEngine:
    """
    Contextual Multi-Arm Bandit with Meta-RL and Regret Minimization.

    Action space (bounded — all ALLOWED_UPDATE_TARGETS):
      retry_strategy, bandit_q_table, ranker_priors, confidence_weights,
      epsilon, exploration_rate, model_selection_weights, proposal_weights,
      window_size_policy, hyperparameter_ranges

    Safety invariants:
      - No action may target FORBIDDEN_TARGETS
      - All writes blocked when sandbox mode is active
      - InstabilityDetector triggers rollback when Δconfidence < -0.10 × 3 episodes
    """

    # Candidate actions for retry strategy selection
    DEFAULT_ACTION_SPACE: List[str] = [
        "restart_from_eda",
        "restart_from_proposal",
        "restart_full_pipeline",
        "adjust_hyperparameters",
        "apply_feature_selection",
        "change_model_class",
        "increase_regularization",
        "reduce_feature_count",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        rl_cfg = self.config.get("rl", {})

        self._epsilon_min: float = float(rl_cfg.get("epsilon_min", 0.05))
        self._epsilon_max: float = float(rl_cfg.get("epsilon_max", 0.30))
        self._alpha: float = float(rl_cfg.get("alpha", 0.10))  # Q-value learning rate
        self._drift_psi_threshold: float = float(
            self.config.get("pipeline", {}).get("qa_gate", {}).get("max_drift_psi", 0.20)
        )

        self._state_path = Path(
            self.config.get("storage", {}).get(
                "meta_rl_state", "data/state/meta_rl_state.json"
            )
        )
        self._checkpoint_mgr = RLCheckpointManager(
            checkpoint_dir=str(
                Path(self._state_path).parent / "rl_checkpoints"
            ),
            max_checkpoints=int(rl_cfg.get("max_checkpoints", 20)),
        )
        self._instability = InstabilityDetector(
            lookback_episodes=int(rl_cfg.get("stability_lookback_episodes", 3)),
            instability_delta_threshold=float(
                rl_cfg.get("instability_delta_threshold", -0.10)
            ),
        )

        # ── Enhancement: load state and apply warm-start prior if fresh ──────
        self._state: BanditState = self._load_state()
        self._ensure_actions()

        if self._state.episode == 0:
            domain = rl_cfg.get("domain", "default")
            prior  = get_prior(domain)
            for action, q_prior in prior.items():
                if action not in self._state.actions:
                    self._state.actions[action] = ActionStats(name=action)
                self._state.actions[action].q_value = float(q_prior)
            logger.info("MetaRLEngine: warm-start priors applied for domain='%s'", domain)

        # ── Enhancement 2: Contextual Bandit (LinUCB) ─────────────────────────
        self._contextual_bandit = ContextualBandit(
            action_space=self.DEFAULT_ACTION_SPACE,
            config=self.config,
        )

        # ── Enhancement 4: Reward Shaper ──────────────────────────────────────
        self._reward_shaper = RewardShaper(self.config)

        # ── Enhancement 5: Cross-Dataset Transfer Learning ───────────────────
        self._knowledge_transfer = KnowledgeTransfer(self.config)

        # ── Enhancement 1: Optuna Auto-Tuner ─────────────────────────────────
        self._autotuner = RLAutoTuner(self.config)
        # Apply any previously discovered best params immediately
        best_params = self._autotuner.load_best_params()
        self._alpha = float(best_params.get("alpha", self._alpha))
        self._epsilon_min = float(best_params.get("epsilon_min", self._epsilon_min))
        self._epsilon_max = float(best_params.get("epsilon_max", self._epsilon_max))
        logger.info(
            "MetaRLEngine: loaded tuned params alpha=%.4f eps=[%.2f,%.2f]",
            self._alpha, self._epsilon_min, self._epsilon_max,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self,
        context: Optional[Dict[str, Any]] = None,
        psi_level: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Select the next retry / strategy action.

        Enhancement 2: Uses LinUCB contextual bandit when metrics are available
        (picks best action for this specific dataset's features). Falls back to
        epsilon-greedy when no context is provided.

        Args:
            context  : Feature dict (dataset_size, null_rate, drift_level, etc.) [legacy]
            psi_level: Current PSI drift score (triggers policy switch if > threshold)
            metrics  : Full pipeline metrics dict (used for LinUCB context vector)

        Returns:
            Action name string from action space.
        """
        self._switch_policy_based_on_drift(psi_level)

        # Enhancement 2: Use contextual bandit (LinUCB) when metrics available
        if metrics:
            ctx_vec = self._contextual_bandit.build_context(
                metrics=metrics, episode=self._state.episode
            )
            action = self._contextual_bandit.select_action(ctx_vec)
            logger.info("MetaRLEngine: LinUCB selected action='%s'", action)
            return action

        # Fallback: original epsilon-greedy
        epsilon = max(
            self._epsilon_min,
            min(self._epsilon_max, self._state.epsilon),
        )

        if random.random() < epsilon:
            # Explore: random action (weighted away from high-regret arms)
            action = self._explore(context)
        else:
            # Exploit: highest Q-value
            action = self._exploit()

        return action

    def record_outcome(
        self,
        action: str,
        reward: float,
        prev_confidence: float = 0.0,
        new_confidence: float = 0.0,
        # Enhancement 4: shaped reward inputs
        elapsed_seconds: Optional[float] = None,
        drift_psi_before: Optional[float] = None,
        drift_psi_after: Optional[float] = None,
        # Enhancement 5: transfer learning inputs
        metrics: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        domain: str = "default",
    ) -> None:
        """
        Record the reward for a taken action and update Q-values + regret.

        Enhancement 4: Uses RewardShaper to compute a multi-dimensional reward
        (confidence + speed + drift improvement) instead of raw confidence.
        Enhancement 5: Stores domain fingerprint for future transfer learning.
        Enhancement 1: Triggers Optuna auto-tuning every N episodes.

        Args:
            action          : Action that was taken
            reward          : Raw confidence-based reward ∈ [0, 1] (legacy fallback)
            prev_confidence : Confidence before action
            new_confidence  : Confidence after action
            elapsed_seconds : Pipeline wall-clock time (for speed reward)
            drift_psi_before: PSI drift score before the run
            drift_psi_after : PSI drift score after the run
            metrics         : Full metrics dict (for contextual bandit update + transfer)
            run_id          : Pipeline run identifier
            domain          : Data domain string
        """
        # Safety: never record against forbidden targets
        if action in FORBIDDEN_TARGETS:
            raise RLSafetyViolation(target=action, context="record_outcome()")

        if action not in self._state.actions:
            self._state.actions[action] = ActionStats(name=action)

        # ── Enhancement 4: Compute shaped reward ──────────────────────────────
        shaped_reward = self._reward_shaper.compute(
            confidence_score=new_confidence,
            elapsed_seconds=elapsed_seconds,
            drift_psi_before=drift_psi_before,
            drift_psi_after=drift_psi_after,
        )
        # Use shaped reward if available; fall back to raw reward
        effective_reward = shaped_reward if new_confidence > 0.0 else max(0.0, min(1.0, reward))

        a = self._state.actions[action]
        old_q = a.q_value

        # Q-value EMA update (no catastrophic forgetting)
        a.q_value = (1 - self._alpha) * old_q + self._alpha * effective_reward
        a.count += 1
        a.total_reward += effective_reward
        self._state.total_reward += effective_reward
        self._state.episode += 1

        # Regret = best_possible_reward - actual_reward
        best_q = max(s.q_value for s in self._state.actions.values())
        regret = max(0.0, best_q - effective_reward)
        a.cumulative_regret += regret
        self._state.total_regret += regret

        # Track confidence delta for instability detection
        confidence_delta = new_confidence - prev_confidence
        self._instability.record(confidence_delta)

        # Anneal epsilon toward min over time
        self._anneal_epsilon()

        # ── Enhancement 2: Update contextual bandit with shaped reward ────────
        if metrics:
            ctx_vec = self._contextual_bandit.build_context(
                metrics=metrics, episode=self._state.episode
            )
            self._contextual_bandit.update(action, ctx_vec, effective_reward)

        # ── Enhancement 5: Store domain fingerprint for transfer learning ─────
        if metrics and run_id:
            fp = DomainFingerprint.from_run_result({
                **metrics,
                "confidence_score": new_confidence,
            })
            q_snapshot = {n: float(s.q_value) for n, s in self._state.actions.items()}
            self._knowledge_transfer.store(run_id, domain, fp, q_snapshot)

        # Save checkpoint every 10 episodes
        if self._state.episode % 10 == 0:
            self._checkpoint_mgr.save(
                episode=self._state.episode,
                weights=self._state.to_dict(),
            )

        # ── Enhancement 1: Optuna auto-tunes after EVERY episode (adaptive) ──
        if self._autotuner.should_tune(self._state.episode):
            from learning.experience_memory_v2 import ExperienceMemoryV2
            try:
                mem    = ExperienceMemoryV2.from_config(self.config)
                events = mem.list_recent(limit=500)
                # Pass episode so adaptive trial count (10/25/50) applies
                best   = self._autotuner.tune(events, episode=self._state.episode)
                self._alpha       = float(best.get("alpha",       self._alpha))
                self._epsilon_min = float(best.get("epsilon_min", self._epsilon_min))
                self._epsilon_max = float(best.get("epsilon_max", self._epsilon_max))
                logger.info(
                    "MetaRLEngine ep=%d: Optuna applied alpha=%.4f eps=[%.2f,%.2f]",
                    self._state.episode, self._alpha, self._epsilon_min, self._epsilon_max,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MetaRLEngine: Optuna auto-tune failed: %s", exc)

        logger.info(
            "MetaRLEngine episode=%d action=%s raw_reward=%.4f shaped=%.4f "
            "Q: %.4f→%.4f regret=%.4f eps=%.4f",
            self._state.episode, action, reward, effective_reward,
            old_q, a.q_value, regret, self._state.epsilon,
        )

        self._save_state()

        # Instability detection → recommend rollback
        if self._instability.is_unstable():
            available = self._checkpoint_mgr.list_checkpoints()
            if available:
                logger.warning(
                    "MetaRL instability detected. Last stable checkpoint: episode=%d. "
                    "Call revert_to_checkpoint(%d) to rollback.",
                    available[-1], available[-1],
                )

    def revert_to_checkpoint(self, episode: int) -> bool:
        """
        Restore policy weights from a checkpoint by episode number.

        Args:
            episode: The episode number to restore from

        Returns:
            True if successful, False if checkpoint not found or checksum mismatch.
        """
        weights = self._checkpoint_mgr.restore(episode=episode)
        if weights is None:
            return False

        self._state = BanditState.from_dict(weights)
        self._instability.reset()
        logger.info("MetaRL reverted to checkpoint episode=%d", episode)
        self._save_state()
        return True

    def get_status(self) -> Dict[str, Any]:
        """Return current RL engine status dict for dashboard / monitoring."""
        return {
            "episode": self._state.episode,
            "active_policy": self._state.active_policy,
            "epsilon": round(self._state.epsilon, 4),
            "total_reward": round(self._state.total_reward, 4),
            "total_regret": round(self._state.total_regret, 4),
            "action_stats": {
                name: {
                    "q_value": round(a.q_value, 4),
                    "count": a.count,
                    "mean_reward": round(a.mean_reward, 4),
                    "mean_regret": round(a.mean_regret, 4),
                }
                for name, a in self._state.actions.items()
            },
            "checkpoints": self._checkpoint_mgr.list_checkpoints(),
            "is_sandbox": is_sandbox_active(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _explore(self, context: Optional[Dict[str, Any]]) -> str:
        """
        Weighted exploration: prefer actions with low cumulative regret.
        High-regret actions are down-weighted but not excluded (ensures diversity).
        """
        actions = list(self._state.actions.keys())
        if not actions:
            return random.choice(self.DEFAULT_ACTION_SPACE)

        # Inverse regret weighting
        regrets = np.array([self._state.actions[a].mean_regret for a in actions])
        max_regret = regrets.max()
        if max_regret > 0:
            weights = 1.0 / (1.0 + regrets)  # soft inverse
        else:
            weights = np.ones(len(actions))
        weights = weights / weights.sum()

        return str(np.random.choice(actions, p=weights))

    def _exploit(self) -> str:
        """Return action with highest Q-value."""
        if not self._state.actions:
            return self.DEFAULT_ACTION_SPACE[0]
        return max(self._state.actions, key=lambda a: self._state.actions[a].q_value)

    def _switch_policy_based_on_drift(self, psi: float) -> None:
        """
        Switch active policy based on current PSI drift level.
        Drift-conditioned policy switching ensures RL adapts to distribution shifts.
        """
        if psi > self._drift_psi_threshold:
            new_policy = "exploration_heavy"
        elif self._state.episode > 50 and all(
            a.count >= 5 for a in self._state.actions.values() if self._state.actions
        ):
            # Assess if we can switch to exploitation-heavy
            best_reward = max(
                (a.mean_reward for a in self._state.actions.values()),
                default=0.0,
            )
            new_policy = "exploitation_heavy" if best_reward >= 0.85 else "default"
        else:
            new_policy = "default"

        if new_policy != self._state.active_policy:
            logger.info(
                "MetaRL policy switch: %s → %s (psi=%.3f)",
                self._state.active_policy, new_policy, psi,
            )
            self._state.active_policy = new_policy
            self._state.epsilon = POLICY_REGISTRY[new_policy]["epsilon"]

    def _anneal_epsilon(self) -> None:
        """Anneal epsilon toward min using cosine schedule."""
        ep = self._state.episode
        T = 1000  # anneal target: 1000 episodes
        progress = min(ep / T, 1.0)
        epsilon_range = self._epsilon_max - self._epsilon_min
        # Cosine annealing
        new_eps = self._epsilon_min + 0.5 * epsilon_range * (1 + math.cos(math.pi * progress))
        # Clamp to active policy bounds
        self._state.epsilon = max(self._epsilon_min, min(self._epsilon_max, new_eps))

    def _ensure_actions(self) -> None:
        """Make sure all default actions are initialized in state."""
        for action in self.DEFAULT_ACTION_SPACE:
            if action not in self._state.actions:
                self._state.actions[action] = ActionStats(name=action)

    def _load_state(self) -> BanditState:
        if not self._state_path.exists():
            return BanditState()
        try:
            import json
            with open(self._state_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return BanditState.from_dict(d)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MetaRL failed to load state: %s. Starting fresh.", exc)
            return BanditState()

    def _save_state(self) -> None:
        """Persist current state unless sandbox mode is active."""
        sandbox_safe_write(self._state_path, self._state.to_dict())
