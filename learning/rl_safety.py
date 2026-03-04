"""
learning/rl_safety.py
---------------------
Production-grade RL safety infrastructure for DIPEX.

Provides:
  - FORBIDDEN_TARGETS: the immutable set of components RL can never modify
  - RLSafetyViolation: exception raised when RL attempts to touch forbidden targets
  - RLCheckpointManager: save/restore policy weight checkpoints with versioning
  - RLSandboxGuard: context manager that prevents weight persistence
  - rl_safe: decorator to enforce safety on any update function

Design principle:
  RL can only improve EFFICIENCY (fewer retries, better hypotheses).
  RL can NEVER modify safety-critical components: schema validators, hard gates,
  compliance rules, statistical verification logic. These invariants are enforced
  at code level, not configuration.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("dipex.rl_safety")

# ══════════════════════════════════════════════════════════════════════════════
# IMMUTABLE FORBIDDEN TARGETS — enforced by code, not configuration
# ══════════════════════════════════════════════════════════════════════════════

FORBIDDEN_TARGETS: Set[str] = frozenset({  # type: ignore[assignment]
    # Deterministic validation (Hard Gate 1)
    "schema_validators",
    "hard_gate_1",
    "validation_rules",
    "null_threshold",
    "type_enforcement",
    "data_contracts",
    # Statistical verification (Hard Gate 2)
    "hard_gate_2",
    "statistical_verification",
    "statistical_verifier",
    "permutation_validation",
    "leakage_detection",
    # Compliance & regulatory
    "compliance_rules",
    "regulatory_engine",
    "domain_verifier_rules",
    "pii_detector",
    # Immutability layer
    "immutability_guard",
    "bronze_layer",
    "layer_write_guard",
    # Audit & logging
    "audit_trail",
    "audit_log",
    # Security
    "jwt_auth",
    "rbac",
})

# Keys that RL is ALLOWED to touch
ALLOWED_UPDATE_TARGETS: Set[str] = frozenset({
    "retry_strategy",
    "bandit_q_table",
    "ranker_priors",
    "confidence_weights",
    "epsilon",
    "exploration_rate",
    "model_selection_weights",
    "proposal_weights",
    "window_size_policy",
    "hyperparameter_ranges",
})


# ══════════════════════════════════════════════════════════════════════════════
# RLSafetyViolation
# ══════════════════════════════════════════════════════════════════════════════

class RLSafetyViolation(RuntimeError):
    """
    Raised when RL attempts to modify a FORBIDDEN_TARGET.
    Logged at CRITICAL level; caught by pipeline bridge to halt RL step.
    """
    def __init__(self, target: str, context: str = "") -> None:
        self.target = target
        self.context = context
        msg = (
            f"[RL SAFETY VIOLATION] RL attempted to modify forbidden target: '{target}'. "
            f"Context: {context or 'none'}. "
            "Safety invariant enforced — update blocked and logged."
        )
        super().__init__(msg)
        logger.critical(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Safety Enforcement Utilities
# ══════════════════════════════════════════════════════════════════════════════

def assert_target_allowed(target: str, context: str = "") -> None:
    """
    Raise RLSafetyViolation if target is in FORBIDDEN_TARGETS.
    Call this before every RL weight update.
    """
    if target in FORBIDDEN_TARGETS:
        raise RLSafetyViolation(target=target, context=context)


def rl_safe(func: Callable) -> Callable:
    """
    Decorator that enforces the `target` argument of an update function
    is not in FORBIDDEN_TARGETS before execution.

    Usage:
        @rl_safe
        def update_policy(target: str, value: Any): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        target = kwargs.get("target") or (args[1] if len(args) > 1 else None)
        if target:
            assert_target_allowed(str(target), context=f"{func.__name__}()")
        return func(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# RLCheckpointManager — save/restore per-episode policy weights
# ══════════════════════════════════════════════════════════════════════════════

class RLCheckpointManager:
    """
    Thread-safe checkpoint manager for RL policy weights.
    Maintains a rolling window of N checkpoints for rollback capability.

    Features:
    - SHA-256 integrity hash on each checkpoint
    - Versioned filenames: checkpoint_{episode:06d}.json
    - Automatic pruning when max_checkpoints exceeded
    """

    def __init__(
        self,
        checkpoint_dir: str = "data/rl_checkpoints",
        max_checkpoints: int = 20,
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_checkpoints
        self._lock = threading.Lock()

    def save(self, episode: int, weights: Dict[str, Any]) -> Path:
        """Save a checkpoint for the given episode. Returns the file path."""
        with self._lock:
            payload = {
                "episode": episode,
                "weights": copy.deepcopy(weights),
            }
            content = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            checksum = hashlib.sha256(content.encode()).hexdigest()
            payload["checksum"] = checksum

            path = self._dir / f"checkpoint_{episode:06d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            logger.info("RL checkpoint saved: episode=%d path=%s", episode, path)
            self._prune()
            return path

    def restore(self, episode: int) -> Optional[Dict[str, Any]]:
        """
        Restore weights from a checkpoint by episode number.
        Returns None if not found or checksum mismatch.
        """
        path = self._dir / f"checkpoint_{episode:06d}.json"
        if not path.exists():
            logger.warning("RL checkpoint not found for episode=%d", episode)
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            stored_checksum = payload.pop("checksum", None)
            verify_content = json.dumps(
                {"episode": payload["episode"], "weights": payload["weights"]},
                sort_keys=True, ensure_ascii=False,
            )
            computed = hashlib.sha256(verify_content.encode()).hexdigest()

            if stored_checksum and computed != stored_checksum:
                logger.error(
                    "RL checkpoint CHECKSUM MISMATCH for episode=%d — refusing restore",
                    episode,
                )
                return None

            logger.info("RL checkpoint restored: episode=%d", episode)
            return payload["weights"]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to restore RL checkpoint episode=%d: %s", episode, exc)
            return None

    def list_checkpoints(self) -> List[int]:
        """Return sorted list of available checkpoint episode numbers."""
        checkpoints = sorted(
            int(p.stem.replace("checkpoint_", ""))
            for p in self._dir.glob("checkpoint_*.json")
        )
        return checkpoints

    def _prune(self) -> None:
        """Remove oldest checkpoints when exceeding max_checkpoints."""
        checkpoints = self.list_checkpoints()
        while len(checkpoints) > self._max:
            oldest = checkpoints.pop(0)
            oldest_path = self._dir / f"checkpoint_{oldest:06d}.json"
            oldest_path.unlink(missing_ok=True)
            logger.debug("RL checkpoint pruned: episode=%d", oldest)


# ══════════════════════════════════════════════════════════════════════════════
# RLSandboxGuard — dry-run context manager
# ══════════════════════════════════════════════════════════════════════════════

_sandbox_active = threading.local()


def is_sandbox_active() -> bool:
    """Returns True if we are currently running inside an RL sandbox context."""
    return getattr(_sandbox_active, "active", False)


@contextmanager
def rl_sandbox():
    """
    Context manager that activates RL sandbox mode.
    Inside this context, rl_write_guard blocks file writes.

    Usage:
        with rl_sandbox():
            engine.update_for_run(run_id)  # computes but doesn't persist
    """
    _sandbox_active.active = True
    logger.info("RL sandbox mode ACTIVATED — no weights will be persisted")
    try:
        yield
    finally:
        _sandbox_active.active = False
        logger.info("RL sandbox mode DEACTIVATED")


def sandbox_safe_write(path: Path, data: Any) -> bool:
    """
    Write JSON data to path UNLESS sandbox mode is active.
    Returns True if write happened, False if blocked by sandbox.
    """
    if is_sandbox_active():
        logger.debug(
            "RL sandbox BLOCKED write to %s (data would have been: %s)",
            path,
            str(data)[:200],
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Instability Detection
# ══════════════════════════════════════════════════════════════════════════════

class InstabilityDetector:
    """
    Monitors rolling confidence deltas to detect RL instability.

    Triggers if: mean(confidence_delta[-lookback:]) < instability_threshold
    This prevents the RL engine from continuing to degrade performance.
    """

    def __init__(
        self,
        lookback_episodes: int = 3,
        instability_delta_threshold: float = -0.10,
    ) -> None:
        self._lookback = lookback_episodes
        self._threshold = instability_delta_threshold
        self._history: List[float] = []

    def record(self, confidence_delta: float) -> None:
        """Record the change in confidence for the latest episode."""
        self._history.append(confidence_delta)

    def is_unstable(self) -> bool:
        """Return True if recent episodes show sustained confidence degradation."""
        if len(self._history) < self._lookback:
            return False
        recent = self._history[-self._lookback:]
        mean_delta = sum(recent) / len(recent)
        unstable = mean_delta < self._threshold
        if unstable:
            logger.warning(
                "RL instability detected: mean_confidence_delta=%.4f over %d episodes "
                "(threshold=%.4f). Rollback recommended.",
                mean_delta, self._lookback, self._threshold,
            )
        return unstable

    def reset(self) -> None:
        """Reset history after a successful rollback."""
        self._history.clear()
