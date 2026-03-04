"""
verifier/retry_engine.py
------------------------
Step 6 — Retry Engine.

Triggered when the aggregated Confidence Score is below a configured
threshold. Selects a retry strategy (immediate_retry, exponential_backoff,
abort_and_log) via the contextual bandit and records the attempt for
audit and for the Retry Penalty Score on the next run.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Actions the engine can recommend
ACTION_RETRY_IMMEDIATE = "immediate_retry"
ACTION_RETRY_BACKOFF = "exponential_backoff"
ACTION_ABORT_AND_LOG = "abort_and_log"


@dataclass
class RetryDecision:
    """Result of evaluating whether to retry and which strategy to use."""

    triggered: bool          # True if confidence was below threshold
    action: str              # immediate_retry | exponential_backoff | abort_and_log
    should_retry: bool       # True if caller should schedule a retry
    backoff_seconds: float   # Suggested delay before retry (0 for immediate)
    attempt: int             # Current attempt index (1-based for this run)
    max_attempts: int
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "action": self.action,
            "should_retry": self.should_retry,
            "backoff_seconds": self.backoff_seconds,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "reason": self.reason,
            "details": self.details,
        }


class RetryEngine:
    """
    Invoked when confidence_score < threshold. Tracks per-run attempt counts,
    consults bandit state for retry_strategy, and returns a RetryDecision.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        pipe = self.config.get("pipeline", {})
        retry_cfg = pipe.get("retry_engine", {})
        self._confidence_threshold: float = float(
            retry_cfg.get("confidence_threshold", 0.6)
        )
        self._max_retries: int = int(retry_cfg.get("max_retries", 3))
        self._backoff_base_seconds: float = float(
            retry_cfg.get("backoff_base_seconds", 2.0)
        )
        self._state_path: Path = Path(
            retry_cfg.get("state_path", "data/retry_state.json")
        )
        self._bandit_path: Path = Path(
            self.config.get("proposal", {}).get("bandit", {}).get(
                "storage_path", "data/bandit_state.json"
            )
        )
        self._load_attempts()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RetryEngine":
        return cls(config)

    def _load_attempts(self) -> None:
        """Load run_id -> attempt_count from state file."""
        self._attempts: Dict[str, int] = {}
        if self._state_path.exists():
            try:
                with open(self._state_path, "r") as f:
                    self._attempts = json.load(f)
            except Exception as exc:
                logger.warning("Retry state file unreadable: %s", exc)

    def _save_attempts(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(self._attempts, f, indent=2)

    def _get_attempt(self, run_id: str) -> int:
        return self._attempts.get(run_id, 0)

    def _increment_attempt(self, run_id: str) -> int:
        self._attempts[run_id] = self._attempts.get(run_id, 0) + 1
        self._save_attempts()
        return self._attempts[run_id]

    def _select_strategy(self) -> str:
        """Select retry strategy from bandit Q-table (retry_strategy context)."""
        if not self._bandit_path.exists():
            return random.choice([
                ACTION_RETRY_IMMEDIATE,
                ACTION_RETRY_BACKOFF,
                ACTION_ABORT_AND_LOG,
            ])
        try:
            with open(self._bandit_path, "r") as f:
                q = json.load(f)
            strategies = q.get("retry_strategy", {})
            if not strategies:
                return ACTION_ABORT_AND_LOG
            actions = list(strategies.keys())
            weights = [max(0.01, strategies.get(a, 0.5)) for a in actions]
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(actions, weights=probs, k=1)[0]
        except Exception as exc:
            logger.warning("Could not read bandit state for retry strategy: %s", exc)
            return ACTION_ABORT_AND_LOG

    def _select_policy(self, context: str) -> Optional[str]:
        """
        Generic helper to select a policy from the bandit Q-table for a given context
        (e.g. transformation_policy, window_policy, model_family, feature_subset_policy).
        """
        if not self._bandit_path.exists():
            return None
        try:
            with open(self._bandit_path, "r") as f:
                q = json.load(f)
            options = q.get(context, {})
            if not options:
                return None
            actions = list(options.keys())
            weights = [max(0.01, options.get(a, 0.5)) for a in actions]
            total = sum(weights)
            probs = [w / total for w in weights]
            return random.choices(actions, weights=probs, k=1)[0]
        except Exception as exc:
            logger.warning("Could not read bandit state for context '%s': %s", context, exc)
            return None

    def evaluate(
        self,
        run_id: str,
        confidence_score: float,
        confidence_vector: Optional[Dict[str, Any]] = None,
    ) -> RetryDecision:
        """
        If confidence_score < threshold, trigger retry logic: increment attempt,
        select strategy, and return a RetryDecision. Otherwise return a decision
        that triggered=False and should_retry=False.
        """
        below = confidence_score < self._confidence_threshold
        attempt_before = self._get_attempt(run_id)

        if not below:
            return RetryDecision(
                triggered=False,
                action=ACTION_ABORT_AND_LOG,
                should_retry=False,
                backoff_seconds=0.0,
                attempt=attempt_before,
                max_attempts=self._max_retries,
                reason="Confidence score above threshold; no retry needed.",
                details={"confidence_score": confidence_score},
            )

        attempt = self._increment_attempt(run_id)
        strategy = self._select_strategy()

        if attempt > self._max_retries:
            strategy = ACTION_ABORT_AND_LOG
            should_retry = False
            backoff_seconds = 0.0
            reason = (
                f"Confidence below threshold (%.4f < %.4f) but max_retries (%d) reached; abort."
                % (confidence_score, self._confidence_threshold, self._max_retries)
            )
        elif strategy == ACTION_RETRY_IMMEDIATE:
            should_retry = True
            backoff_seconds = 0.0
            reason = (
                "Confidence below threshold; retry immediately (strategy=immediate_retry)."
            )
        elif strategy == ACTION_RETRY_BACKOFF:
            should_retry = True
            backoff_seconds = self._backoff_base_seconds ** min(attempt, 5)
            reason = (
                "Confidence below threshold; retry after %.1fs backoff (strategy=exponential_backoff)."
                % backoff_seconds
            )
        else:
            should_retry = False
            backoff_seconds = 0.0
            reason = (
                "Confidence below threshold; abort and log (strategy=abort_and_log)."
            )

        # Build intelligent adjustment plan (Step 7)
        cv = confidence_vector or {}
        dq = float(cv.get("data_quality_score", 1.0))
        stat = float(cv.get("statistical_score", 1.0))
        stab = float(cv.get("stability_score", 1.0))
        drift = float(cv.get("drift_robustness_score", 1.0))
        comp = float(cv.get("compliance_score", 1.0))
        gate1_decision = cv.get("details", {}).get("gate1_decision")

        # Decide restart stage
        if gate1_decision == "REJECT" or dq < 0.7:
            restart_stage = "FULL"       # Potential schema / hard data issues
        elif stat < 0.7 or stab < 0.7:
            restart_stage = "PROPOSAL"   # Strategy / model / feature issues
        else:
            restart_stage = "EDA"        # Default: re-examine profiling/EDA

        adjustments: Dict[str, Any] = {}
        # Different transformation / encoding when compliance or statistical issues
        if comp < 0.8 or stat < 0.7:
            adjustments["transformation_policy"] = self._select_policy("transformation_policy")
            adjustments["encoding_policy"] = self._select_policy("encoding_policy")
        # Different window size when drift robustness is weak
        if drift < 0.8:
            adjustments["window_policy"] = self._select_policy("window_policy")
        # Alternative model / feature subset when stability or statistical weak
        if stat < 0.7 or stab < 0.7:
            adjustments["model_family"] = self._select_policy("model_family")
            adjustments["feature_subset_policy"] = self._select_policy("feature_subset_policy")

        logger.warning(
            "RetryEngine triggered run_id=%s attempt=%d confidence=%.4f strategy=%s should_retry=%s restart=%s",
            run_id, attempt, confidence_score, strategy, should_retry, restart_stage,
        )

        return RetryDecision(
            triggered=True,
            action=strategy,
            should_retry=should_retry,
            backoff_seconds=backoff_seconds,
            attempt=attempt,
            max_attempts=self._max_retries,
            reason=reason,
            details={
                "confidence_score": confidence_score,
                "confidence_threshold": self._confidence_threshold,
                "confidence_vector": cv,
                "plan": {
                    "restart_stage": restart_stage,
                    "adjustments": adjustments,
                },
            },
        )

    def current_attempt(self, run_id: str) -> int:
        """Return current attempt count for run_id (for use in confidence vector)."""
        return self._get_attempt(run_id)
