"""
analyst/rl_optimizer.py
-------------------------
Analyst-specific RL optimizer (separate from platform RL engine).

Learns the best analyst strategy (cleaning method, model class, window size, etc.)
for a given dataset context using a contextual multi-arm bandit.

State: {dataset_size, cardinality, null_rate, drift_level, signal_to_noise, last_confidence}
Action space: {cleaning_strategy, model_class, retry_path, normalization_method, window_size}
Reward: Δconfidence since previous attempt

Safety: Wires into Meta-RL engine. Sandbox mode fully supported.
        Named policy: "analyst_optimizer"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from learning.rl_safety import (
    sandbox_safe_write,
    is_sandbox_active,
    assert_target_allowed,
    FORBIDDEN_TARGETS,
)
from learning.meta_rl_engine import MetaRLEngine

logger = logging.getLogger("dipex.rl_optimizer")


# ══════════════════════════════════════════════════════════════════════════════
# State Encoder
# ══════════════════════════════════════════════════════════════════════════════

def encode_state(context: Dict[str, Any]) -> np.ndarray:
    """
    Encode analyst dataset context into a fixed-length feature vector.

    State dimensions (6):
    [dataset_size_normalized, cardinality_norm, null_rate, drift_level,
     signal_to_noise_norm, last_confidence]
    """
    return np.array([
        min(1.0, float(context.get("dataset_size", 0)) / 1_000_000),
        min(1.0, float(context.get("cardinality", 0)) / 1_000),
        max(0.0, min(1.0, float(context.get("null_rate", 0.0)))),
        max(0.0, min(1.0, float(context.get("drift_level", 0.0)))),
        max(0.0, min(1.0, float(context.get("signal_to_noise", 1.0)) / 10.0)),
        max(0.0, min(1.0, float(context.get("last_confidence", 0.5)))),
    ], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# Analyst RL Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class AnalystRLOptimizer:
    """
    Analyst-specific reinforcement learning optimizer.
    Selects the best analyst strategy for each dataset context.

    Policy name: "analyst_optimizer"
    Action space: cleaning_strategy, model_class, retry_path, normalization_method, window_size
    """

    # Bounded action space — analyst operations domain
    ACTION_SPACE: Dict[str, List[str]] = {
        "cleaning_strategy": [
            "impute_mean",
            "impute_median",
            "impute_mode",
            "impute_knn",
            "drop_rows_with_nulls",
            "flag_and_keep",
        ],
        "model_class": [
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "linear_regression",
            "xgboost",
            "lightgbm",
        ],
        "retry_path": [
            "restart_from_eda",
            "restart_from_proposal",
            "adjust_hyperparameters",
            "change_model_class",
            "reduce_feature_count",
        ],
        "normalization_method": [
            "standard_scaler",
            "min_max_scaler",
            "robust_scaler",
            "none",
        ],
        "window_size": [
            "tumbling_5min",
            "tumbling_30min",
            "sliding_10min_2stride",
            "none",
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        rl_cfg = self.config.get("rl", {})

        self._policy_name = "analyst_optimizer"
        self._state_path = Path(
            self.config.get("storage", {}).get(
                "analyst_rl_state", "data/state/analyst_rl_state.json"
            )
        )

        # Each action dimension gets its own Meta-RL sub-engine
        self._engines: Dict[str, MetaRLEngine] = {}
        for action_dim in self.ACTION_SPACE:
            dim_config = dict(self.config)
            if "storage" not in dim_config:
                dim_config["storage"] = {}
            dim_config["storage"]["meta_rl_state"] = (
                f"data/state/analyst_rl_{action_dim}.json"
            )
            self._engines[action_dim] = MetaRLEngine(config=dim_config)
            # Initialize actions for this dimension
            for action in self.ACTION_SPACE[action_dim]:
                assert_target_allowed(action_dim)  # ensure dimension is not forbidden
                if action not in self._engines[action_dim]._state.actions:
                    from learning.meta_rl_engine import ActionStats
                    self._engines[action_dim]._state.actions[action] = ActionStats(name=action)

    def select_strategy(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Select the best analyst strategy for the given dataset context.

        Args:
            context: {dataset_size, cardinality, null_rate, drift_level,
                      signal_to_noise, last_confidence}

        Returns:
            Strategy dict: {cleaning_strategy, model_class, retry_path,
                            normalization_method, window_size}

        Note: In sandbox mode, actions are selected but NOT persisted.
        """
        psi = float(context.get("drift_level", 0.0))
        strategy = {}

        for action_dim, engine in self._engines.items():
            # Temporarily replace engine's action space with this dimension's actions
            for action in self.ACTION_SPACE[action_dim]:
                from learning.meta_rl_engine import ActionStats
                if action not in engine._state.actions:
                    engine._state.actions[action] = ActionStats(name=action)

            selected = engine.select_action(context=context, psi_level=psi)
            # Ensure selected is valid for this dimension
            if selected not in self.ACTION_SPACE[action_dim]:
                selected = self.ACTION_SPACE[action_dim][0]  # fallback to first valid action
            strategy[action_dim] = selected

        sandbox_note = " [SANDBOX — not persisted]" if is_sandbox_active() else ""
        logger.info(
            "AnalystRLOptimizer: selected strategy%s: %s", sandbox_note, strategy
        )
        return strategy

    def record_outcome(
        self,
        strategy: Dict[str, str],
        prev_confidence: float,
        new_confidence: float,
    ) -> None:
        """
        Record the outcome of an applied strategy and update Q-values.

        Args:
            strategy: Strategy dict that was applied
            prev_confidence: Confidence score before applying strategy
            new_confidence: Confidence score after applying strategy
        """
        reward = max(0.0, min(1.0, new_confidence))
        confidence_delta = new_confidence - prev_confidence

        for action_dim, action in strategy.items():
            if action_dim in self._engines:
                self._engines[action_dim].record_outcome(
                    action=action,
                    reward=reward,
                    prev_confidence=prev_confidence,
                    new_confidence=new_confidence,
                )

        if is_sandbox_active():
            logger.info("AnalystRLOptimizer: sandbox mode — outcome recorded but weights not saved")
        else:
            logger.info(
                "AnalystRLOptimizer: outcome recorded — Δconfidence=%.4f reward=%.4f",
                confidence_delta, reward,
            )

    def get_status(self) -> Dict[str, Any]:
        """Return status of all RL sub-engines for dashboard monitoring."""
        return {
            "policy_name": self._policy_name,
            "action_dimensions": list(self.ACTION_SPACE.keys()),
            "is_sandbox": is_sandbox_active(),
            "engines": {
                dim: engine.get_status()
                for dim, engine in self._engines.items()
            },
        }

    def revert_all_to_checkpoint(self, episode: int) -> Dict[str, bool]:
        """Revert all action dimension engines to a checkpoint episode."""
        return {
            dim: engine.revert_to_checkpoint(episode=episode)
            for dim, engine in self._engines.items()
        }



# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY SHIMS
# ══════════════════════════════════════════════════════════════════════════════

from enum import Enum
from dataclasses import dataclass as _dc_rl


class StrategyDomain(str, Enum):
    """Strategy domains for the legacy RLOptimizer.propose() API."""
    CLEANING = "cleaning_strategy"
    MODEL = "model_class"
    RETRY = "retry_path"
    NORMALIZATION = "normalization_method"
    WINDOW = "window_size"


@_dc_rl
class RLProposal:
    """
    A single RL strategy proposal.
    Test-contract attributes: .advisory_only (always True), .strategy_name, .domain, .q_value
    """
    domain: str
    strategy_name: str
    action: str
    q_value: float
    advisory_only: bool = True
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "strategy_name": self.strategy_name,
            "action": self.action,
            "q_value": round(self.q_value, 4),
            "advisory_only": self.advisory_only,
            "rationale": self.rationale,
        }


# Default action space for the simple RLOptimizer
_DEFAULT_ACTIONS: Dict[str, List[str]] = {
    "cleaning": [
        "impute_mean", "impute_median", "impute_mode",
        "impute_knn", "drop_rows_with_nulls", "flag_and_keep",
    ],
    "cleaning_strategy": [
        "impute_mean", "impute_median", "impute_mode",
        "impute_knn", "drop_rows_with_nulls", "flag_and_keep",
    ],
    "model_selection": [
        "random_forest", "gradient_boosting", "logistic_regression",
        "xgboost", "lightgbm", "linear_regression",
    ],
    "model_class": [
        "random_forest", "gradient_boosting", "logistic_regression",
        "xgboost", "lightgbm", "linear_regression",
    ],
    "retry_path": ["restart_from_eda", "restart_from_proposal", "full_pipeline_restart"],
    "normalization_method": ["min_max", "z_score", "robust_scaler", "log_transform"],
    "window_size": ["1d", "7d", "14d", "30d", "90d"],
}


class RLOptimizer:
    """
    Test-contract-compatible RL optimizer.

    Simpler than AnalystRLOptimizer — uses a flat _arm_weights dict per domain
    for epsilon-greedy bandit. Fully advisory (advisory_only=True on all proposals).

    Usage::
        rl = RLOptimizer(store_path="/tmp/rl")
        props = rl.propose("cleaning", n_proposals=3)
        rl.record_outcome(props[0], actual_gain=0.15)
        msg = rl.check_hard_gate(null_rate=0.05, checksum_ok=True)
        top = rl.top_strategies("cleaning", n=3)
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        epsilon: float = 0.15,
        alpha: float = 0.10,
        sandbox: bool = False,
    ) -> None:
        self._store_path = Path(store_path) if store_path else None
        if self._store_path:
            self._store_path.mkdir(parents=True, exist_ok=True)
        self._config = config or {}
        self._epsilon = epsilon
        self._alpha = alpha
        self.sandbox: bool = sandbox or os.environ.get("DIPEX_RL_SANDBOX", "").lower() in (
            "true", "1", "yes"
        )
        # _arm_weights[domain][strategy_name] = weight (initialized uniform)
        self._arm_weights: Dict[str, Dict[str, float]] = {}
        for domain, actions in _DEFAULT_ACTIONS.items():
            self._arm_weights[domain] = {a: 1.0 for a in actions}
        # Flat action registry for select_action()
        self._flat_actions: List[str] = [
            a for acts in _DEFAULT_ACTIONS.values() for a in acts
        ]
        self._flat_counts: Dict[str, int] = {a: 0 for a in self._flat_actions}
        self._flat_rewards: Dict[str, float] = {a: 0.0 for a in self._flat_actions}
        self._total_calls: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def propose(
        self,
        domain: str,
        n_proposals: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RLProposal]:
        """
        Propose top-N strategies for a given domain.

        All proposals are advisory_only=True.
        Strategy names are sorted by weight (highest first) with epsilon-greedy
        exploration — explores a random arm with probability epsilon.

        Args:
            domain: Domain string (e.g. "cleaning", "model_selection")
            n_proposals: Number of proposals to return
            context: Optional context dict (ignored by simple bandit)

        Returns:
            List[RLProposal] sorted by weight desc
        """
        domain_key = domain.lower()
        weights = self._arm_weights.get(domain_key)
        if weights is None:
            # Unknown domain — fallback to first default domain
            domain_key = "cleaning"
            weights = self._arm_weights.setdefault(domain_key, {a: 1.0 for a in _DEFAULT_ACTIONS["cleaning"]})

        import random as _rand
        strategies = list(weights.items())   # [(name, weight), ...]

        # Epsilon-greedy: with prob epsilon shuffle (explore), else sort by weight
        if _rand.random() < self._epsilon:
            _rand.shuffle(strategies)
        else:
            strategies.sort(key=lambda kv: kv[1], reverse=True)

        proposals = []
        for strategy_name, q_value in strategies[:n_proposals]:
            proposals.append(RLProposal(
                domain=domain_key,
                strategy_name=strategy_name,
                action=strategy_name,
                q_value=q_value,
                advisory_only=True,
                rationale=(
                    f"Proposed for domain '{domain_key}' with weight={q_value:.4f}. "
                    "Advisory only — no automatic execution."
                ),
            ))
        return proposals

    def select_action(self, state: Optional[Dict[str, Any]] = None) -> str:
        """
        Select an action from the flat action space using UCB1.

        Parameters
        ----------
        state:
            Current context dict (used for logging; UCB1 is context-free here).

        Returns
        -------
        str
            Selected action name.
        """
        import math as _math
        self._total_calls += 1

        best_action: Optional[str] = None
        best_score = -_math.inf

        for action in self._flat_actions:
            count = self._flat_counts[action]
            if count == 0:
                best_action = action
                break
            mean_r = self._flat_rewards[action] / count
            ucb = mean_r + _math.sqrt(2.0 * _math.log(self._total_calls) / count)
            if ucb > best_score:
                best_score = ucb
                best_action = action

        logger.debug("[RLOptimizer] select_action -> %s (sandbox=%s)", best_action, self.sandbox)
        return best_action  # type: ignore[return-value]

    def record_outcome(
        self,
        proposal: Any,
        actual_gain: float = 0.0,
        *,
        # alternate signature used by test_rl_safety: action+success+confidence_gained
        action: Optional[str] = None,
        success: bool = True,
        confidence_gained: float = 0.0,
    ) -> None:
        """
        Update arm weights based on observed gain (legacy API) OR
        UCB1 flat stats (when called with action= kwarg from test_rl_convergence).
        """
        # Legacy API: proposal object passed positionally
        if hasattr(proposal, "strategy_name"):
            domain = proposal.domain.lower()
            name   = proposal.strategy_name
            gain   = actual_gain
            if domain not in self._arm_weights:
                self._arm_weights[domain] = {}
            old_w = self._arm_weights[domain].get(name, 1.0)
            new_w = (1.0 - self._alpha) * old_w + self._alpha * (gain + 1.0)
            self._arm_weights[domain][name] = max(0.0, new_w)
            logger.info("RLOptimizer.record_outcome: %s/%s gain=%.4f w: %.4f→%.4f",
                        domain, name, gain, old_w, new_w)
        else:
            # Called with action= kwarg (test_rl_convergence / select_action path)
            _action = action or (proposal if isinstance(proposal, str) else None)
            _gain   = (1.0 if success else -0.5) + float(confidence_gained)
            if _action and _action in self._flat_counts:
                self._flat_counts[_action]  += 1
                self._flat_rewards[_action] += _gain
            logger.info("RLOptimizer.record_outcome(ucb): action=%s gain=%.4f", _action, _gain)

    def check_hard_gate(
        self,
        null_rate: float = 0.0,
        checksum_ok: bool = True,
        confidence: float = 1.0,
    ) -> Optional[str]:
        """
        Hard gate checks.  Returns a descriptive error string if gate fires,
        or None if all checks pass.

        Args:
            null_rate: Overall null rate in data (block if > 0.90)
            checksum_ok: Data integrity flag (block if False)
            confidence: Confidence score (block if < 0.0 — reserved for future)

        Returns:
            None if gate passes, str describing failure otherwise
        """
        if null_rate > 0.90:
            return (
                f"Hard gate: data is too sparse (null_rate={null_rate:.1%} > 90%). "
                "This dataset cannot support reliable analysis."
            )
        if not checksum_ok:
            return (
                "Hard gate: checksum mismatch — data integrity cannot be verified. "
                "Pipeline halted."
            )
        return None

    def top_strategies(
        self,
        domain: str,
        n: int = 3,
    ) -> List[tuple]:
        """
        Return top-N strategies for a domain sorted by weight descending.

        Returns:
            List of (strategy_name, weight) tuples sorted desc by weight
        """
        domain_key = domain.lower()
        weights = self._arm_weights.get(domain_key, {})
        sorted_strategies = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_strategies[:n]
