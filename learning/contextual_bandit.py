"""
learning/contextual_bandit.py
------------------------------
Enhancement 2: Contextual Bandit with LinUCB Algorithm.

Replaces the simple epsilon-greedy bandit with a LinUCB (Linear Upper Confidence
Bound) contextual bandit which uses dataset features to pick the optimal action.

Production invariants (industry-grade requirements):
  - np.linalg.solve(A, b) instead of inv(A) @ b — numerically stable (LU decomp)
  - Ridge regularisation (λI) on A — prevents singular matrix on sparse data
  - Thread-safe: all mutations protected by threading.RLock()
  - All features normalised to [0,1] before LinUCB model — equal-scale guarantee
  - Graceful degradation on LinAlgError — returns neutral 0.5 score
  - Dimension validation on load — detects and handles corrupt state files
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.contextual_bandit")

# Context feature names — order MUST be stable across serialisation/deserialisation
_CONTEXT_FEATURES = [
    "null_rate",
    "drift_psi",
    "row_count_log",
    "col_count_norm",
    "confidence_score",
    "schema_complexity",
    "is_first_run",
]
_N_FEATURES = len(_CONTEXT_FEATURES)

# Per-feature normalisation bounds [lo, hi] → maps raw value to [0, 1]
# Reflects realistic data ranges in the DIPEX domain.
_FEATURE_BOUNDS: List[tuple] = [
    (0.0, 1.0),    # null_rate            ∈ [0, 1]
    (0.0, 1.0),    # drift_psi (PSI)      ∈ [0, 1]
    (0.0, 7.0),    # row_count_log        ∈ [0, log10(10M)=7]
    (0.0, 1.0),    # col_count_norm       ∈ [0, 1]
    (0.0, 1.0),    # confidence_score     ∈ [0, 1]
    (0.0, 1.0),    # schema_complexity    normalised
    (0.0, 1.0),    # is_first_run         ∈ {0, 1}
]

# LinUCB exploration coefficient — higher = more exploration vs exploitation
_ALPHA_UCB: float = 1.0

# Ridge regularisation coefficient — initialise A = λI to prevent singularity
# even before any observations arrive. Typical range: 0.1–10.0.
_LAMBDA_REG: float = 1.0


def _normalise(raw: np.ndarray) -> np.ndarray:
    """Min-max normalise each feature dimension to [0, 1] using domain bounds."""
    out = np.empty_like(raw, dtype=float)
    for i, (lo, hi) in enumerate(_FEATURE_BOUNDS):
        spread = hi - lo
        out[i] = (raw[i] - lo) / spread if spread > 1e-9 else 0.0
    return np.clip(out, 0.0, 1.0)


class LinUCBArm:
    """
    A single arm (action) in the LinUCB bandit.

    State is two matrices maintained per action:
      A : n×n regularised covariance matrix (λI + Σ xxᵀ)
      b : n-dim reward accumulation vector   (Σ r·x)

    θ = A⁻¹b is the learned weight vector (estimated via solve, not inversion).

    UCB score: θᵀx + α · √(xᵀ A⁻¹ x)
    Both terms computed via np.linalg.solve for numerical stability.
    """

    __slots__ = ("name", "_alpha", "_n", "_A", "_b")

    def __init__(self, name: str, n_features: int, alpha: float = _ALPHA_UCB) -> None:
        self.name   = name
        self._alpha = alpha
        self._n     = n_features
        # Regularised identity: prevents singular A on first call (sparse data)
        self._A: np.ndarray = np.identity(self._n, dtype=float) * _LAMBDA_REG
        self._b: np.ndarray = np.zeros(self._n, dtype=float)

    def ucb_score(self, x: np.ndarray) -> float:
        """
        Compute LinUCB upper confidence bound score for context x.

        Implementation note:
          Instead of computing inv(A) explicitly (O(n³), numerically fragile),
          we use np.linalg.solve which applies LU decomposition:
            θ = solve(A, b)    is equivalent to A⁻¹b
            v = solve(A, x)    gives A⁻¹x for the confidence term xᵀA⁻¹x
          This is numerically stable and faster for n < 100.
        """
        try:
            theta   = np.linalg.solve(self._A, self._b)     # A⁻¹b
            A_inv_x = np.linalg.solve(self._A, x)            # A⁻¹x
            exploit = float(theta @ x)
            explore = float(math.sqrt(max(0.0, float(x @ A_inv_x))))
            return exploit + self._alpha * explore
        except np.linalg.LinAlgError:
            # Singular or ill-conditioned A — return neutral score so arm is explored
            logger.warning(
                "LinUCBArm [%s]: singular matrix encountered — returning neutral UCB",
                self.name,
            )
            return 0.5 + self._alpha * 0.1  # Slight bias toward exploration

    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Sherman-Morrison-compatible rank-1 update.
        A ← A + xxᵀ,  b ← b + r·x
        """
        self._A = self._A + np.outer(x, x)
        self._b = self._b + reward * x

    def theta(self) -> np.ndarray:
        """Return current weight vector θ = A⁻¹b (stable via solve)."""
        try:
            return np.linalg.solve(self._A, self._b)
        except np.linalg.LinAlgError:
            return np.zeros(self._n, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "A":    self._A.tolist(),
            "b":    self._b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], n_features: int, alpha: float) -> "LinUCBArm":
        arm = cls(d["name"], n_features, alpha)
        try:
            A = np.array(d["A"], dtype=float)
            b = np.array(d["b"], dtype=float)
            if A.shape == (n_features, n_features) and b.shape == (n_features,):
                arm._A = A
                arm._b = b
            else:
                logger.warning(
                    "LinUCBArm [%s]: dimension mismatch on load "
                    "(expected %d, got A=%s b=%s) — resetting",
                    d.get("name", "?"), n_features, A.shape, b.shape,
                )
        except (ValueError, KeyError) as exc:
            logger.warning("LinUCBArm: corrupt state entry, resetting: %s", exc)
        return arm


class ContextualBandit:
    """
    LinUCB Contextual Bandit for DIPEX pipeline action selection.

    Thread-safe: all arm mutations and file I/O are protected by threading.RLock().
    Features are normalised before LinUCB sees them (equal-scale guarantee).

    Usage:
        bandit = ContextualBandit(action_space, config)
        context = bandit.build_context(metrics, episode)
        action  = bandit.select_action(context)
        # ... run pipeline using selected action ...
        bandit.update(action, context, reward)
    """

    def __init__(
        self,
        action_space: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (config or {}).get("rl", {}).get("contextual_bandit", {})
        self._alpha   = float(cfg.get("alpha_ucb", _ALPHA_UCB))
        self._actions = list(action_space)
        self._path    = Path(cfg.get("state_path", "data/state/contextual_bandit.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock    = threading.RLock()  # Reentrant lock for thread safety

        with self._lock:
            self._arms: Dict[str, LinUCBArm] = self._load()
            for action in action_space:
                if action not in self._arms:
                    self._arms[action] = LinUCBArm(action, _N_FEATURES, self._alpha)

    # ── Public API ────────────────────────────────────────────────────────────

    def build_context(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        episode: int = 0,
    ) -> np.ndarray:
        """
        Build a normalised context vector from pipeline run metrics.

        All values are min-max normalised to [0, 1] per _FEATURE_BOUNDS
        to ensure no single feature dominates due to scale differences.
        """
        m = metrics or {}
        row_count = max(1, int(float(m.get("rows_ingested", m.get("row_count", 1)) or 1)))
        raw = np.array([
            float(m.get("null_rate",         0.0) or 0.0),
            float(m.get("drift_score",       m.get("drift_psi", 0.0)) or 0.0),
            math.log10(row_count),
            float(m.get("col_count",         10)) / 100.0,
            float(m.get("confidence_score",  0.0) or 0.0),
            float(m.get("schema_complexity", float(m.get("col_count", 10)) / 10)) / 10.0,
            1.0 if episode == 0 else 0.0,
        ], dtype=float)
        return _normalise(raw)

    def select_action(self, context: Optional[np.ndarray] = None) -> str:
        """
        Return the highest-UCB action for this context. Thread-safe.
        Falls back to uniform context vector if none is provided.
        """
        if context is None:
            context = np.ones(_N_FEATURES, dtype=float) / math.sqrt(_N_FEATURES)

        with self._lock:
            scores: Dict[str, float] = {
                name: arm.ucb_score(context)
                for name, arm in self._arms.items()
            }

        best = max(scores, key=lambda k: scores[k])
        logger.info(
            "ContextualBandit.select_action: '%s' (UCB=%.4f, n_arms=%d)",
            best, scores[best], len(scores),
        )
        return best

    def update(self, action: str, context: np.ndarray, reward: float) -> None:
        """Update the selected arm with observed reward. Thread-safe."""
        with self._lock:
            if action not in self._arms:
                self._arms[action] = LinUCBArm(action, _N_FEATURES, self._alpha)
            self._arms[action].update(context, float(max(0.0, min(1.0, reward))))
            self._save()
        logger.debug("ContextualBandit.update: arm='%s' reward=%.4f", action, reward)

    def get_arm_weights(self) -> Dict[str, float]:
        """Return dict action → expected scalar reward (θ · unit vector)."""
        unit = np.ones(_N_FEATURES, dtype=float) / math.sqrt(_N_FEATURES)
        with self._lock:
            return {name: float(arm.theta() @ unit) for name, arm in self._arms.items()}

    # ── Private I/O (must be called inside self._lock) ───────────────────────

    def _load(self) -> Dict[str, LinUCBArm]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            arms = {
                d["name"]: LinUCBArm.from_dict(d, _N_FEATURES, self._alpha)
                for d in data.get("arms", [])
            }
            logger.info("ContextualBandit: loaded %d arms from %s", len(arms), self._path)
            return arms
        except Exception as exc:  # noqa: BLE001
            logger.warning("ContextualBandit: failed to load state (%s) — starting fresh", exc)
            return {}

    def _save(self) -> None:
        try:
            data = {"arms": [arm.to_dict() for arm in self._arms.values()]}
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("ContextualBandit: failed to save state: %s", exc)
