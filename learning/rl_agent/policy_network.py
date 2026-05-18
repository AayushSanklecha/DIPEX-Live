"""
learning/rl_agent/policy_network.py
--------------------------------------
Actor (Policy) Network for the PPO agent.

Architecture: 3-layer MLP with LayerNorm
  STATE_DIM → 256 → 256 → n_actions_per_axis

Improvements over naive baseline (elite-grade):
  - Hidden size: 64 → 256 (standard for real PPO deployments)
  - Orthogonal initialization (Hu et al., OpenAI Procgen paper)
  - Layer normalization on hidden representations (stabilizes training
    across diverse state distributions)
  - Small output head initialization (scale=0.01) — critical for PPO
    to start with near-uniform action distributions (exploration)
  - Multi-head output: one softmax head per action axis (8 axes)
  - ReLU activation in hidden layers
  - No external deep-learning dependency at inference: numpy-only forward pass
  - PyTorch used in ppo_trainer.py for gradient computation

For training, see ppo_trainer.py (PyTorch — required, no SPSA fallback).
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .action_space import AXIS_SIZES, N_AXES
from .state_encoder import STATE_DIM

logger = logging.getLogger("dipex.learning.rl_agent.policy_network")

# Module-level isolated RNG — never pollutes global numpy random state
_POLICY_RNG = np.random.default_rng(42)

# ── Network dimensions ────────────────────────────────────────────────────────
HIDDEN_SIZE = 256   # Upgraded: 64 → 256 (industry standard for PPO)
MODEL_PATH = Path(os.path.dirname(__file__)) / ".." / ".." / "models" / "rl_ppo_policy.pkl"


# ── Numpy activation functions ────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-9)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalization: normalize across the feature dimension."""
    mu = x.mean()
    sigma = x.std() + eps
    return (x - mu) / sigma


def _orthogonal_init(shape: tuple, scale: float = 1.0) -> np.ndarray:
    """
    Orthogonal initialization (Saxe et al., 2013).
    Critical for PPO: avoids gradient explosion / vanishing in deep networks
    and produces better feature representations at initialization.
    """
    rows, cols = shape
    flat = np.random.standard_normal((max(rows, cols), min(rows, cols)))
    U, _, Vt = np.linalg.svd(flat, full_matrices=False)
    W = U if rows >= cols else Vt
    W = W[:rows, :cols]
    return (W * scale).astype(np.float32)


class PolicyNetwork:
    """
    Lightweight numpy-based actor network for inference.

    Architecture (upgraded — elite grade):
      Input  (STATE_DIM=12)
        → FC1 (256) + LayerNorm + ReLU
        → FC2 (256) + LayerNorm + ReLU
        → 8 independent softmax heads (one per action axis)

    Weights trained by ppo_trainer.py (PyTorch).
    Numpy weights synced back after each PPO update for zero-dependency inference.
    """

    def __init__(self) -> None:
        # ── Hidden layers: orthogonal init with scale=sqrt(2) for ReLU ───────
        self.W1 = _orthogonal_init((HIDDEN_SIZE, STATE_DIM), scale=math.sqrt(2))
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.W2 = _orthogonal_init((HIDDEN_SIZE, HIDDEN_SIZE), scale=math.sqrt(2))
        self.b2 = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # ── Layer norm parameters (learned scale=1, bias=0 at init) ───────────
        self.ln1_g = np.ones(HIDDEN_SIZE, dtype=np.float32)
        self.ln1_b = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.ln2_g = np.ones(HIDDEN_SIZE, dtype=np.float32)
        self.ln2_b = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # ── Multi-head output: scale=0.01 → near-uniform initial distribution ─
        # Critical: large output weights cause early policy collapse (low entropy)
        self.heads: List[Tuple[np.ndarray, np.ndarray]] = [
            (_orthogonal_init((sz, HIDDEN_SIZE), scale=0.01),
             np.zeros(sz, dtype=np.float32))
            for sz in AXIS_SIZES
        ]
        self._episode_count = 0

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _layer_norm_apply(
        self, x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Affine layer norm: γ·LayerNorm(x) + β."""
        mu = x.mean()
        sigma = np.sqrt(((x - mu) ** 2).mean() + eps)
        return gamma * ((x - mu) / sigma) + beta

    def forward(self, state: np.ndarray) -> List[np.ndarray]:
        """
        Forward pass: returns list of probability distributions, one per axis.
        Each element shape: (axis_size,) — softmax probability vector.
        """
        x = self.W1 @ state + self.b1
        x = self._layer_norm_apply(x, self.ln1_g, self.ln1_b)
        x = _relu(x)

        x = self.W2 @ x + self.b2
        x = self._layer_norm_apply(x, self.ln2_g, self.ln2_b)
        x = _relu(x)

        probs = [_softmax(W @ x + b) for W, b in self.heads]
        return probs

    def sample_action(
        self, state: np.ndarray, greedy: bool = False
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Sample action indices from the policy distributions.

        Parameters
        ----------
        state  : STATE_DIM-D normalized state vector
        greedy : if True, take argmax (exploitation); if False, sample (exploration)

        Returns
        -------
        (indices, probs_list) — integer indices per axis, probability vectors
        """
        probs = self.forward(state)
        if greedy:
            indices = [int(np.argmax(p)) for p in probs]
        else:
            # Use isolated module RNG — not global np.random (reproducibility fix)
            indices = [
                int(_POLICY_RNG.choice(len(p), p=p / (p.sum() + 1e-9)))
                for p in probs
            ]
        return indices, probs

    def log_prob(self, state: np.ndarray, indices: List[int]) -> float:
        """Compute sum of log-probabilities of a given action under the current policy."""
        probs = self.forward(state)
        log_p = sum(
            float(np.log(probs[i][indices[i]] + 1e-8))
            for i in range(N_AXES)
        )
        return log_p

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """Serialize all network weights to a pickle checkpoint."""
        save_path = Path(path or MODEL_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        weights = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "ln1_g": self.ln1_g, "ln1_b": self.ln1_b,
            "ln2_g": self.ln2_g, "ln2_b": self.ln2_b,
            "heads": self.heads,
            "episode_count": self._episode_count,
            "hidden_size": HIDDEN_SIZE,
        }
        with open(save_path, "wb") as f:
            pickle.dump(weights, f, protocol=5)
        logger.info("[PolicyNetwork] Saved (episode=%d, hidden=%d) → %s",
                    self._episode_count, HIDDEN_SIZE, save_path)
        return str(save_path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load network weights from checkpoint. Returns True if successful."""
        load_path = Path(path or MODEL_PATH)
        if not load_path.exists():
            logger.debug("[PolicyNetwork] No checkpoint at %s — using orthogonal init.",
                         load_path)
            return False
        try:
            import sys
            import numpy.core.numeric
            import numpy.core.multiarray
            sys.modules.setdefault("numpy._core", sys.modules.get("numpy.core"))
            sys.modules.setdefault("numpy._core.numeric", sys.modules.get("numpy.core.numeric"))
            sys.modules.setdefault("numpy._core.multiarray", sys.modules.get("numpy.core.multiarray"))
            
            with open(load_path, "rb") as f:
                weights = pickle.load(f)

            # Handle old 64-unit checkpoints gracefully (shape mismatch → reinit)
            saved_hidden = weights.get("hidden_size", 64)
            if saved_hidden != HIDDEN_SIZE:
                logger.warning(
                    "[PolicyNetwork] Checkpoint hidden_size=%d ≠ current=%d — "
                    "discarding checkpoint and using fresh init.",
                    saved_hidden, HIDDEN_SIZE,
                )
                return False

            self.W1 = weights["W1"]; self.b1 = weights["b1"]
            self.W2 = weights["W2"]; self.b2 = weights["b2"]
            self.ln1_g = weights.get("ln1_g", np.ones(HIDDEN_SIZE, dtype=np.float32))
            self.ln1_b = weights.get("ln1_b", np.zeros(HIDDEN_SIZE, dtype=np.float32))
            self.ln2_g = weights.get("ln2_g", np.ones(HIDDEN_SIZE, dtype=np.float32))
            self.ln2_b = weights.get("ln2_b", np.zeros(HIDDEN_SIZE, dtype=np.float32))
            self.heads = weights["heads"]
            self._episode_count = weights.get("episode_count", 0)
            logger.info("[PolicyNetwork] Loaded checkpoint: episode=%d, hidden=%d, path=%s",
                        self._episode_count, HIDDEN_SIZE, load_path)
            return True
        except Exception as exc:
            logger.warning("[PolicyNetwork] Load failed: %s — reinitializing.", exc)
            return False

    @property
    def is_trained(self) -> bool:
        """True if the agent has collected ≥ 20 real episodes."""
        return self._episode_count >= 20
