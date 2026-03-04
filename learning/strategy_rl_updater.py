"""
learning/strategy_rl_updater.py
--------------------------------
STEP 10 — Reinforcement Learning Update.

Updates:
  - Retry selection policy        (bandit Q-table: retry_strategy)
  - Insight ranking weights       (feature-level importance counts)
  - Confidence calibration stats  (for future calibration of scores)

Critically:
  - NEVER touches deterministic validation logic (Hard Gate 1).
  - NEVER touches compliance / regulatory rules (RegulatoryEngine).
  - Learning only optimizes efficiency and strategy, not safety.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RLUpdateContext:
    """Minimal context needed for strategy RL updates."""

    run_id: str
    confidence_score: float
    retry_count: int
    model_type: Optional[str]
    task: Optional[str]
    top_insight_feature: Optional[str]


class StrategyRLUpdater:
    """
    Applies small, conservative RL-style updates to:

      - Bandit Q-table (retry_strategy, model_family)
      - Insight ranking weights (per-feature)
      - Confidence calibration stats

    All updates are stored in separate JSON state files and do NOT alter:
      - Validation thresholds
      - Deterministic validation logic
      - Compliance/regulatory rule definitions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        bandit_cfg = self.config.get("proposal", {}).get("bandit", {})
        storage_cfg = self.config.get("storage", {})

        self._bandit_path = Path(bandit_cfg.get("storage_path", "data/bandit_state.json"))
        self._insight_weights_path = Path(
            storage_cfg.get("insight_weights_path", "data/insight_weights.json")
        )
        self._conf_calib_path = Path(
            storage_cfg.get("confidence_calibration_path", "data/confidence_calibration.json")
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StrategyRLUpdater":
        return cls(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, ctx: RLUpdateContext) -> None:
        """
        Main entry point after a run has been successfully approved and stored.
        """
        try:
            self._update_bandit_policies(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StrategyRLUpdater: bandit update failed: %s", exc)

        try:
            self._update_insight_weights(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StrategyRLUpdater: insight weights update failed: %s", exc)

        try:
            self._update_confidence_calibration(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StrategyRLUpdater: confidence calibration update failed: %s", exc)

    # ------------------------------------------------------------------
    # Bandit Q-table updates
    # ------------------------------------------------------------------

    def _load_bandit(self) -> Dict[str, Dict[str, float]]:
        if not self._bandit_path.exists():
            return {}
        try:
            with open(self._bandit_path, "r") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StrategyRLUpdater: could not load bandit state: %s", exc)
            return {}

    def _save_bandit(self, q_table: Dict[str, Dict[str, float]]) -> None:
        self._bandit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._bandit_path, "w") as f:
            json.dump(q_table, f, indent=2)

    def _soft_update(self, old: float, reward: float, alpha: float = 0.1) -> float:
        return (1.0 - alpha) * old + alpha * reward

    def _update_bandit_policies(self, ctx: RLUpdateContext) -> None:
        """
        Updates:
          - retry_strategy: reward high confidence with fewer retries
          - model_family: lightly reward model_type used in high-confidence runs
        """
        q_table = self._load_bandit()
        if not q_table:
            return

        # Reward design: encourage high confidence with fewer retries
        # Reward ∈ [0,1]: confidence - penalty_per_retry
        penalty_per_retry = 0.1
        reward = max(0.0, min(1.0, ctx.confidence_score - penalty_per_retry * ctx.retry_count))

        # Retry strategy context
        retry_ctx = q_table.get("retry_strategy", {})
        if retry_ctx:
            # Conservatively reward exponential_backoff when retries were needed,
            # and immediate_retry when no retry was needed.
            if ctx.retry_count == 0:
                preferred = "immediate_retry"
            elif ctx.retry_count <= 2:
                preferred = "exponential_backoff"
            else:
                preferred = "abort_and_log"

            if preferred in retry_ctx:
                old_val = float(retry_ctx.get(preferred, 0.0))
                retry_ctx[preferred] = self._soft_update(old_val, reward)
            q_table["retry_strategy"] = retry_ctx

        # Model family context
        model_ctx = q_table.get("model_family", {})
        if model_ctx and ctx.model_type in model_ctx:
            old_val = float(model_ctx.get(ctx.model_type, 0.0))
            model_ctx[ctx.model_type] = self._soft_update(old_val, reward)
            q_table["model_family"] = model_ctx

        self._save_bandit(q_table)

    # ------------------------------------------------------------------
    # Insight ranking weights
    # ------------------------------------------------------------------

    def _load_insight_weights(self) -> Dict[str, float]:
        if not self._insight_weights_path.exists():
            return {}
        try:
            with open(self._insight_weights_path, "r") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StrategyRLUpdater: could not load insight weights: %s", exc)
            return {}

    def _save_insight_weights(self, weights: Dict[str, float]) -> None:
        self._insight_weights_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._insight_weights_path, "w") as f:
            json.dump(weights, f, indent=2)

    def _update_insight_weights(self, ctx: RLUpdateContext) -> None:
        """
        Lightly rewards features that frequently appear as top_insight_candidate
        in high-confidence runs.
        """
        if not ctx.top_insight_feature:
            return
        if ctx.confidence_score < 0.6:
            # Only reward when confidence at least marginal
            return

        weights = self._load_insight_weights()
        key = ctx.top_insight_feature
        old = float(weights.get(key, 0.0))
        # Increment by a scaled reward; very conservative to avoid runaway
        increment = max(0.01, min(0.1, ctx.confidence_score / 10.0))
        weights[key] = old + increment
        self._save_insight_weights(weights)

    # ------------------------------------------------------------------
    # Confidence calibration stats
    # ------------------------------------------------------------------

    def _load_conf_calibration(self) -> Dict[str, Any]:
        if not self._conf_calib_path.exists():
            return {}
        try:
            with open(self._conf_calib_path, "r") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("StrategyRLUpdater: could not load confidence calibration: %s", exc)
            return {}

    def _save_conf_calibration(self, data: Dict[str, Any]) -> None:
        self._conf_calib_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._conf_calib_path, "w") as f:
            json.dump(data, f, indent=2)

    def _update_confidence_calibration(self, ctx: RLUpdateContext) -> None:
        """
        Tracks simple moving statistics for confidence scores across
        successful runs, as a basis for future calibration.
        """
        data = self._load_conf_calibration()
        count = int(data.get("count", 0))
        mean_conf = float(data.get("mean_confidence", 0.0))

        # Online mean update
        new_count = count + 1
        new_mean = (mean_conf * count + ctx.confidence_score) / new_count

        data["count"] = new_count
        data["mean_confidence"] = new_mean
        self._save_conf_calibration(data)

