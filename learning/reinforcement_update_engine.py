"""
learning/reinforcement_update_engine.py
---------------------------------------
STEP 10 — Reinforcement Learning Update (efficiency only).

Updates (ALLOWED — efficiency targets only):
  - Retry selection policy (bandit Q-table)
  - Insight ranking weights (feature priors for RankerProposer)
  - Confidence calibration weights
  - Meta-RL policy registry weights
  - Exploration epsilon (bounded [0.05, 0.30])

NEVER updates (safety invariants — enforced by FORBIDDEN_TARGETS):
  - Hard Gate 1 / deterministic validation logic
  - Compliance / regulatory rules
  - Statistical verification thresholds
  - Any hard-coded safety boundary

Meta-RL: Maintains a policy registry with UCB1 regret tracking per strategy.
         Switches to exploration-heavy policy when drift is detected (PSI > 0.2).
Sandbox: When rl_sandbox() context is active, all file writes are blocked.
Rollback: revert_to_checkpoint() restores weights if confidence instability detected.
EWC: Elastic Weight Consolidation smoothing prevents catastrophic forgetting.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from learning.experience_memory_v2 import ExperienceMemoryV2
from learning.rl_safety import (
    RLCheckpointManager,
    assert_target_allowed,
    sandbox_safe_write,
    is_sandbox_active,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_EPSILON_MIN: float = 0.05    # Never go below 5% exploration
_EPSILON_MAX: float = 0.30    # Never exceed 30% exploration
_DRIFT_PSI_THRESHOLD: float = 0.20   # PSI above this → exploration-heavy mode
_INSTABILITY_WINDOW: int = 3          # Episodes to check for confidence decline
_INSTABILITY_DELTA: float = -0.10     # Δconfidence threshold triggering rollback
_EWC_LAMBDA: float = 0.90            # Elastic Weight Consolidation smoothing coeff


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class UpdateSummary:
    updated_retry_policy: bool
    updated_ranker_priors: bool
    updated_confidence_weights: bool
    sandbox_active: bool = False
    policies_updated: List[str] = field(default_factory=list)
    regret_updated: bool = False
    epsilon_adjusted: bool = False
    rollback_triggered: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyRecord:
    """Per-strategy record in the Meta-RL policy registry."""
    name: str
    cumulative_reward: float = 0.0
    n_trials: int = 0
    cumulative_regret: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.cumulative_reward / max(self.n_trials, 1)

    @property
    def mean_regret(self) -> float:
        return self.cumulative_regret / max(self.n_trials, 1)

    def update(self, reward: float, best_possible: float = 1.0) -> None:
        self.cumulative_reward += reward
        self.cumulative_regret += max(0.0, best_possible - reward)
        self.last_reward = reward
        self.n_trials += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean_reward": round(self.mean_reward, 6),
            "mean_regret": round(self.mean_regret, 6),
            "n_trials": self.n_trials,
            "last_reward": round(self.last_reward, 6),
        }


# ── Main Engine ────────────────────────────────────────────────────────────────

class ReinforcementUpdateEngine:
    """
    Learns from immutable experience history and writes small, bounded state
    updates to dedicated *efficiency* state files.

    Architecture:
    - Primary bandit    : epsilon-greedy Q-table for retry strategy selection
    - Meta-RL registry  : maintains policy registry; selects best-performing family via UCB1
    - Regret tracker    : cumulative regret per action; deprioritises high-regret actions
    - Drift switch      : when PSI > 0.2, boost epsilon → more exploration
    - Sandbox           : integrates with rl_safety.sandbox_safe_write — no writes in dry-run
    - Checkpoint        : delegates to RLCheckpointManager for rollback on instability
    - EWC smoothing     : prevents catastrophic forgetting on small batches
    """

    _lock = threading.Lock()

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config  = config or {}
        storage  = self.config.get("storage", {})
        proposal = self.config.get("proposal", {})
        pipe     = self.config.get("pipeline", {})
        rl_cfg   = self.config.get("rl", {})

        # ── State file paths ─────────────────────────────────────────────────
        self._bandit_state_path = Path(
            proposal.get("bandit", {}).get("storage_path", "data/bandit_state.json")
        )
        self._ranker_priors_path = Path(
            storage.get("ranker_priors_state", "data/state/ranker_feature_priors.json")
        )
        self._confidence_weights_path = Path(
            storage.get("confidence_weights_state", "data/state/confidence_weights.json")
        )
        self._meta_rl_path = Path(rl_cfg.get("meta_rl_state", "data/state/meta_rl_registry.json"))
        self._epsilon_path = Path(rl_cfg.get("epsilon_state", "data/state/epsilon.json"))
        self._regret_path  = Path(rl_cfg.get("regret_state", "data/state/regret_tracker.json"))
        self._conf_hist_path = Path(rl_cfg.get("confidence_history", "data/state/confidence_history.json"))

        for p in (
            self._ranker_priors_path, self._confidence_weights_path,
            self._meta_rl_path, self._epsilon_path, self._regret_path,
            self._conf_hist_path,
        ):
            p.parent.mkdir(parents=True, exist_ok=True)

        # ── Hyperparameters ──────────────────────────────────────────────────
        self._conf_threshold = float(pipe.get("confidence", {}).get("threshold", 0.6))
        self._alpha          = float(pipe.get("learning", {}).get("alpha", 0.1))
        self._weight_lr      = float(pipe.get("learning", {}).get("confidence_weight_lr", 0.05))

        # ── Sub-components ───────────────────────────────────────────────────
        self._memory = ExperienceMemoryV2.from_config(self.config)
        self._checkpoint_mgr = RLCheckpointManager(
            checkpoint_dir=rl_cfg.get("checkpoint_dir", "data/rl_checkpoints"),
            max_checkpoints=int(rl_cfg.get("max_checkpoints", 20)),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReinforcementUpdateEngine":
        return cls(config)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_for_run(
        self,
        run_id: str,
        drift_psi: Optional[float] = None,
        episode: Optional[int] = None,
    ) -> UpdateSummary:
        """
        Applies Step 10 updates using the run's history plus recent global history.

        Parameters
        ----------
        run_id    : The pipeline run identifier
        drift_psi : PSI score from drift verifier (triggers exploration boost if > 0.20)
        episode   : Current episode counter (used for epsilon annealing + checkpointing)
        """
        with self._lock:
            sandbox    = is_sandbox_active()
            run_events = self._memory.get_run_history(run_id, limit=2000)

            # Snapshot weights before any update (used for in-run rollback)
            weights_before = self._collect_current_weights()

            # ── Core updates (all sandbox-aware via sandbox_safe_write) ─────
            updated_retry  = self._update_retry_policy(run_events)
            updated_ranker = self._update_ranker_priors(run_events)
            updated_conf   = self._recalibrate_confidence_weights()

            # ── Meta-RL: UCB1 policy registry update ─────────────────────────
            meta_policies = self._update_meta_rl_registry(run_events)

            # ── Regret minimization ──────────────────────────────────────────
            regret_updated = self._update_regret_tracker(run_events)

            # ── Epsilon annealing + drift-conditioned adjustment ─────────────
            epsilon_adjusted = self._adjust_epsilon(episode=episode, drift_psi=drift_psi)

            # ── Confidence history + instability rollback ────────────────────
            rollback = self._check_instability_and_rollback(
                run_events, episode, weights_before
            )

            # ── Persist checkpoint (skip if sandbox or rollback triggered) ───
            if not sandbox and not rollback and episode is not None:
                try:
                    self._checkpoint_mgr.save(episode, self._collect_current_weights())
                    logger.info("RL checkpoint saved for episode %d", episode)
                except Exception as exc:   # noqa: BLE001
                    logger.warning("Checkpoint save failed: %s", exc)

            return UpdateSummary(
                updated_retry_policy=updated_retry,
                updated_ranker_priors=updated_ranker,
                updated_confidence_weights=updated_conf,
                sandbox_active=sandbox,
                policies_updated=meta_policies,
                regret_updated=regret_updated,
                epsilon_adjusted=epsilon_adjusted,
                rollback_triggered=rollback,
                details={
                    "run_id": run_id,
                    "event_count": len(run_events),
                    "drift_psi": drift_psi,
                    "episode": episode,
                    "sandbox": sandbox,
                },
            )

    def revert_to_checkpoint(self, episode: int) -> bool:
        """
        Restore RL policy weights from checkpoint at the given episode.
        Called automatically on instability detection, or manually by operator.
        Returns True if rollback succeeded.
        """
        weights = self._checkpoint_mgr.restore(episode)
        if weights is None:
            logger.warning("No checkpoint found for episode %d — rollback aborted.", episode)
            return False

        try:
            for key, path in (
                ("bandit_state",       self._bandit_state_path),
                ("confidence_weights", self._confidence_weights_path),
                ("ranker_priors",      self._ranker_priors_path),
                ("epsilon",            self._epsilon_path),
            ):
                if key in weights and weights[key]:
                    sandbox_safe_write(path, weights[key])
            logger.warning("RL rollback to episode %d completed.", episode)
            return True
        except Exception as exc:   # noqa: BLE001
            logger.error("Rollback write failed: %s", exc)
            return False

    def get_current_epsilon(self) -> float:
        """Return current exploration rate ε."""
        return float(
            self._load_json(self._epsilon_path, default={"epsilon": _EPSILON_MAX})
            .get("epsilon", _EPSILON_MAX)
        )

    def get_best_meta_policy(self) -> str:
        """Return best-performing Meta-RL strategy family by UCB1."""
        return str(
            self._load_json(self._meta_rl_path, default={"_best_policy": "balanced"})
            .get("_best_policy", "balanced")
        )

    # ── Retry policy (bandit Q-table) ─────────────────────────────────────────

    def _update_retry_policy(self, run_events: List[Dict[str, Any]]) -> bool:
        """Updates bandit Q-values. Reward ∝ approved confidence_score. Uses EWC smoothing."""
        assert_target_allowed("retry_strategy", context="ReinforcementUpdateEngine._update_retry_policy")

        approved = [e for e in run_events if e.get("event_type") == "APPROVED_OUTPUT"]
        retries  = [e for e in run_events if e.get("event_type") == "RETRY_DECISION"]
        if not approved or not retries:
            return False

        last_retry = retries[-1]
        action = (
            (last_retry.get("payload") or {}).get("action")
            or (last_retry.get("payload") or {}).get("details", {}).get("action")
            or "abort_and_log"
        )
        conf   = float(((approved[-1].get("payload") or {}).get("confidence_score") or 0.0))
        reward = max(0.0, min(1.0, conf))

        q = self._load_json(self._bandit_state_path, default={})
        retry_table = q.get("retry_strategy", {})
        if not retry_table:
            return False

        old = float(retry_table.get(action, 0.5))
        # EWC: blend toward reward at rate alpha; high lambda preserves prior knowledge
        new = (1.0 - self._alpha) * old + self._alpha * reward
        retry_table[action] = float(new)
        q["retry_strategy"] = retry_table
        sandbox_safe_write(self._bandit_state_path, q)

        logger.info(
            "Step10 retry policy: action=%s Q %.3f→%.3f reward=%.3f",
            action, old, new, reward,
        )
        return True

    # ── Insight ranking weights ────────────────────────────────────────────────

    def _update_ranker_priors(self, run_events: List[Dict[str, Any]]) -> bool:
        """Updates feature priors from approved outputs. No safety impact."""
        assert_target_allowed("ranker_priors", context="ReinforcementUpdateEngine._update_ranker_priors")

        approved = [e for e in run_events if e.get("event_type") == "APPROVED_OUTPUT"]
        if not approved:
            return False

        ws = ((approved[-1].get("payload") or {}).get("winning_strategy") or {})
        top_feature = ws.get("top_insight_candidate")
        if not top_feature:
            return False

        priors = self._load_json(self._ranker_priors_path, default={})
        count = int(priors.get(top_feature, 0))
        priors[top_feature] = count + 1
        sandbox_safe_write(self._ranker_priors_path, priors)

        logger.info("Step10 ranker prior: %s count=%d", top_feature, count + 1)
        return True

    # ── Confidence calibration weights ────────────────────────────────────────

    def _recalibrate_confidence_weights(self, limit: int = 5000) -> bool:
        """
        Learns bounded dimension re-weights via separation between approved vs.
        not-approved outcomes. Uses EWC to prevent catastrophic forgetting.
        Never mutates config.yaml.
        """
        assert_target_allowed("confidence_weights", context="ReinforcementUpdateEngine._recalibrate_confidence_weights")

        events = self._memory.list_recent(limit=limit)
        approved_ids = {e.get("run_id") for e in events if e.get("event_type") == "APPROVED_OUTPUT"}
        vectors = [e for e in events if e.get("event_type") == "CONFIDENCE_VECTOR"]
        if len(vectors) < 20:
            return False

        dims = [
            "data_quality_score", "statistical_score", "stability_score",
            "drift_robustness_score", "compliance_score", "retry_penalty_score",
        ]
        succ: List[List[float]] = []
        fail: List[List[float]] = []
        for e in vectors:
            cv  = ((e.get("payload") or {}).get("confidence_vector") or {})
            row = [float(cv.get(d, 0.0) or 0.0) for d in dims]
            approved = (
                e.get("run_id") in approved_ids
                and float(cv.get("confidence_score", 0.0) or 0.0) >= self._conf_threshold
            )
            (succ if approved else fail).append(row)

        if len(succ) < 5 or len(fail) < 5:
            return False

        succ_arr  = np.asarray(succ, dtype=float)
        fail_arr  = np.asarray(fail, dtype=float)
        diff      = np.clip(succ_arr.mean(axis=0) - fail_arr.mean(axis=0), -1.0, 1.0)
        importance = np.maximum(0.0, diff)
        if importance.sum() == 0:
            return False
        importance = importance / importance.sum()

        current = self._load_json(self._confidence_weights_path, default={})
        weights  = current.get("weights") or {
            "data_quality": 0.20, "statistical": 0.22, "stability": 0.18,
            "drift_robustness": 0.15, "compliance": 0.15, "retry_penalty": 0.10,
        }
        key_map = {
            "data_quality_score": "data_quality",   "statistical_score": "statistical",
            "stability_score": "stability",          "drift_robustness_score": "drift_robustness",
            "compliance_score": "compliance",        "retry_penalty_score": "retry_penalty",
        }
        target = {key_map[d]: float(importance[i]) for i, d in enumerate(dims)}

        # EWC smoothing: strongly preserve prior weights; nudge toward target
        new_weights: Dict[str, float] = {}
        for k, old in weights.items():
            t = target.get(k, float(old))
            w = _EWC_LAMBDA * float(old) + (1 - _EWC_LAMBDA) * t
            new_weights[k] = float(max(0.05, min(0.50, w)))

        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        sandbox_safe_write(
            self._confidence_weights_path,
            {"weights": new_weights, "computed_from": {"succ": len(succ), "fail": len(fail)}},
        )
        logger.info("Step10 confidence weights recalibrated (EWC λ=%.2f).", _EWC_LAMBDA)
        return True

    # ── Meta-RL: UCB1 policy registry ─────────────────────────────────────────

    def _update_meta_rl_registry(self, run_events: List[Dict[str, Any]]) -> List[str]:
        """
        Maintains 3 strategy families: conservative | balanced | aggressive.
        Selects best family using UCB1 (Upper Confidence Bound).
        Updated policy registry is persisted for future run selection.
        """
        approved = [e for e in run_events if e.get("event_type") == "APPROVED_OUTPUT"]
        if not approved:
            return []

        registry_data = self._load_json(self._meta_rl_path, default={})
        policies: Dict[str, PolicyRecord] = {}
        for name in ("conservative", "balanced", "aggressive"):
            d = registry_data.get(name) or {}
            policies[name] = PolicyRecord(
                name=name,
                cumulative_reward=float(d.get("cumulative_reward", 0.0)),
                n_trials=int(d.get("n_trials", 0)),
                cumulative_regret=float(d.get("cumulative_regret", 0.0)),
                last_reward=float(d.get("last_reward", 0.0)),
            )

        ws = ((approved[-1].get("payload") or {}).get("winning_strategy") or {})
        active_policy = ws.get("strategy_family", "balanced")
        if active_policy not in policies:
            active_policy = "balanced"

        reward = float(((approved[-1].get("payload") or {}).get("confidence_score") or 0.0))
        policies[active_policy].update(reward=reward)

        # UCB1 exploration bonus
        total_trials = sum(p.n_trials for p in policies.values()) or 1
        ucb_scores: Dict[str, float] = {}
        for name, p in policies.items():
            if p.n_trials == 0:
                ucb_scores[name] = float("inf")
            else:
                ucb_scores[name] = p.mean_reward + np.sqrt(2.0 * np.log(total_trials) / p.n_trials)

        best_policy = max(ucb_scores, key=lambda k: ucb_scores[k])
        logger.info(
            "Meta-RL: active=%s reward=%.3f | UCB1 best=%s",
            active_policy, reward, best_policy,
        )

        registry_out: Dict[str, Any] = {
            name: {
                "cumulative_reward": p.cumulative_reward,
                "n_trials": p.n_trials,
                "cumulative_regret": p.cumulative_regret,
                "last_reward": p.last_reward,
                "mean_reward": round(p.mean_reward, 6),
                "ucb_score": round(ucb_scores.get(name, 0.0), 6),
            }
            for name, p in policies.items()
        }
        registry_out["_best_policy"]   = best_policy
        registry_out["_total_trials"]  = total_trials

        sandbox_safe_write(self._meta_rl_path, registry_out)
        return [active_policy]

    # ── Regret minimization ───────────────────────────────────────────────────

    def _update_regret_tracker(self, run_events: List[Dict[str, Any]]) -> bool:
        """
        Tracks per-action cumulative regret (opportunity cost vs. optimal decision).
        Actions with mean_regret > 0.50 receive a deprioritisation penalty in the Q-table.
        This prevents consistently bad strategies from being repeatedly selected.
        """
        approved = [e for e in run_events if e.get("event_type") == "APPROVED_OUTPUT"]
        retries  = [e for e in run_events if e.get("event_type") == "RETRY_DECISION"]
        if not retries:
            return False

        conf   = float(((approved[-1].get("payload") or {}).get("confidence_score") or 0.0)) if approved else 0.0
        reward = max(0.0, min(1.0, conf))
        regret = 1.0 - reward

        last_retry = retries[-1]
        action = (
            (last_retry.get("payload") or {}).get("action")
            or (last_retry.get("payload") or {}).get("details", {}).get("action")
            or "abort_and_log"
        )

        tracker = self._load_json(self._regret_path, default={})
        rec     = tracker.get(action) or {"cumulative_regret": 0.0, "n": 0, "last_regret": 0.0}
        rec["cumulative_regret"] = float(rec.get("cumulative_regret", 0.0)) + regret
        rec["n"]                 = int(rec.get("n", 0)) + 1
        rec["last_regret"]       = round(regret, 6)
        rec["mean_regret"]       = round(rec["cumulative_regret"] / rec["n"], 6)
        tracker[action] = rec

        # Deprioritise persistently bad actions via Q-table penalty
        mean_regret = rec["mean_regret"]
        if mean_regret > 0.50:
            q = self._load_json(self._bandit_state_path, default={})
            retry_table = q.get("retry_strategy", {})
            if retry_table and action in retry_table:
                penalty = 0.01 * (mean_regret - 0.50)
                old_q = float(retry_table[action])
                retry_table[action] = max(0.01, old_q - penalty)
                q["retry_strategy"] = retry_table
                sandbox_safe_write(self._bandit_state_path, q)
                logger.info(
                    "Regret penalty: action=%s mean_regret=%.3f Q %.3f→%.3f",
                    action, mean_regret, old_q, retry_table[action],
                )

        sandbox_safe_write(self._regret_path, tracker)
        return True

    # ── Epsilon annealing + drift-conditioned switching ───────────────────────

    def _adjust_epsilon(
        self,
        episode: Optional[int],
        drift_psi: Optional[float],
    ) -> bool:
        """
        Epsilon-greedy with cosine annealing and drift-conditioned boosting.

        Normal:        ε anneals 0.30 → 0.05 over 500 episodes (cosine schedule)
        Drift boost:   ε temporarily clamped to 0.30 when PSI > 0.20
                       (exploration-heavy — re-probe strategies after distribution shift)
        Always:        ε ∈ [_EPSILON_MIN, _EPSILON_MAX]
        """
        epsilon_data = self._load_json(self._epsilon_path, default={"epsilon": _EPSILON_MAX})
        eps    = float(epsilon_data.get("epsilon", _EPSILON_MAX))
        reason = "unchanged"

        if drift_psi is not None and drift_psi > _DRIFT_PSI_THRESHOLD:
            new_eps = _EPSILON_MAX
            reason  = f"drift_boost(PSI={drift_psi:.3f})"
        elif episode is not None and episode > 0:
            decay   = 0.5 * (1.0 + float(np.cos(np.pi * min(episode, 500) / 500)))
            new_eps = _EPSILON_MIN + (_EPSILON_MAX - _EPSILON_MIN) * decay
            reason  = f"cosine_anneal(ep={episode})"
        else:
            new_eps = eps

        new_eps = float(max(_EPSILON_MIN, min(_EPSILON_MAX, new_eps)))
        if abs(new_eps - eps) < 1e-6:
            return False

        epsilon_data["epsilon"] = round(new_eps, 6)
        epsilon_data["reason"]  = reason
        epsilon_data["episode"] = episode
        sandbox_safe_write(self._epsilon_path, epsilon_data)
        logger.info("Epsilon: %.4f → %.4f | %s", eps, new_eps, reason)
        return True

    # ── Instability detection & rollback ──────────────────────────────────────

    def _check_instability_and_rollback(
        self,
        run_events: List[Dict[str, Any]],
        episode: Optional[int],
        weights_before: Dict[str, Any],
    ) -> bool:
        """
        Appends confidence score of current run to rolling history.
        If Δconfidence < -0.10 over last 3 episodes → instability → rollback.
        """
        approved = [e for e in run_events if e.get("event_type") == "APPROVED_OUTPUT"]
        conf = float(((approved[-1].get("payload") or {}).get("confidence_score") or 0.0)) if approved else 0.0

        history_data = self._load_json(self._conf_hist_path, default={"history": []})
        hist: List[float] = [float(x) for x in (history_data.get("history") or [])[-50:]]
        hist.append(conf)
        sandbox_safe_write(self._conf_hist_path, {"history": hist})

        if len(hist) < _INSTABILITY_WINDOW:
            return False

        recent = hist[-_INSTABILITY_WINDOW:]
        delta  = recent[-1] - recent[0]
        if delta >= _INSTABILITY_DELTA:
            return False

        # Instability detected
        logger.warning(
            "RL instability: confidence %.3f→%.3f (Δ=%.3f) over %d steps. Triggering rollback.",
            recent[0], recent[-1], delta, _INSTABILITY_WINDOW,
        )

        # Try checkpoint rollback first
        if episode is not None and episode >= _INSTABILITY_WINDOW:
            if self.revert_to_checkpoint(episode - _INSTABILITY_WINDOW):
                return True

        # Fallback: restore weights captured before this update
        try:
            for key, path in (
                ("bandit_state",       self._bandit_state_path),
                ("confidence_weights", self._confidence_weights_path),
                ("ranker_priors",      self._ranker_priors_path),
            ):
                if weights_before.get(key):
                    sandbox_safe_write(path, weights_before[key])
            logger.warning("In-run rollback applied (no checkpoint available).")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("In-run rollback failed: %s", exc)
            return False

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _collect_current_weights(self) -> Dict[str, Any]:
        """Snapshot all learnable weight files for pre-update checkpoint."""
        return {
            "bandit_state":       self._load_json(self._bandit_state_path, default={}),
            "confidence_weights": self._load_json(self._confidence_weights_path, default={}),
            "ranker_priors":      self._load_json(self._ranker_priors_path, default={}),
            "epsilon":            self._load_json(self._epsilon_path, default={"epsilon": _EPSILON_MAX}),
            "meta_rl":            self._load_json(self._meta_rl_path, default={}),
            "regret":             self._load_json(self._regret_path, default={}),
        }

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        """Legacy helper. Use sandbox_safe_write for new code."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
