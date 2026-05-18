"""
learning/rl_agent/value_network.py
-------------------------------------
Critic (Value) Network for the PPO agent.

Architecture (upgraded — elite grade):
  STATE_DIM → 256 → 256 → 1

Improvements:
  - Hidden size: 64 → 256 (matches upgraded policy network)
  - Orthogonal initialization for hidden layers (scale=sqrt(2) for ReLU)
  - Small output init (scale=1.0 for value head — less aggressive than policy)
  - Layer normalization on hidden layers
  - Outputs a scalar state-value estimate V(s) ∈ ℝ (unbounded linear head)
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .state_encoder import STATE_DIM

logger = logging.getLogger("dipex.learning.rl_agent.value_network")

HIDDEN_SIZE = 256  # Upgraded: 64 → 256 (matches PolicyNetwork)
MODEL_PATH = Path(os.path.dirname(__file__)) / ".." / ".." / "models" / "rl_ppo_value.pkl"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _orthogonal_init(shape: tuple, scale: float = 1.0) -> np.ndarray:
    """Orthogonal initialization (Saxe et al., 2013) — optimal for deep networks."""
    rows, cols = shape
    flat = np.random.standard_normal((max(rows, cols), min(rows, cols)))
    U, _, Vt = np.linalg.svd(flat, full_matrices=False)
    W = U if rows >= cols else Vt
    W = W[:rows, :cols]
    return (W * scale).astype(np.float32)


class ValueNetwork:
    """
    Lightweight numpy-based critic network for inference.

    Architecture:
      Input (STATE_DIM=12)
        → FC1 (256) + LayerNorm + ReLU
        → FC2 (256) + LayerNorm + ReLU
        → FC_out (1): linear — V(s) ∈ ℝ

    Weights trained via PyTorch in ppo_trainer.py and synced back to numpy
    for zero-dependency inference.
    """

    def __init__(self) -> None:
        # ── Hidden layers: orthogonal with scale=sqrt(2) for ReLU ─────────────
        self.W1 = _orthogonal_init((HIDDEN_SIZE, STATE_DIM), scale=math.sqrt(2))
        self.b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.W2 = _orthogonal_init((HIDDEN_SIZE, HIDDEN_SIZE), scale=math.sqrt(2))
        self.b2 = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # ── Layer norm parameters ──────────────────────────────────────────────
        self.ln1_g = np.ones(HIDDEN_SIZE, dtype=np.float32)
        self.ln1_b = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self.ln2_g = np.ones(HIDDEN_SIZE, dtype=np.float32)
        self.ln2_b = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # ── Output head: scale=1.0 (value estimates can be large at init) ──────
        self.Wo = _orthogonal_init((1, HIDDEN_SIZE), scale=1.0)
        self.bo = np.zeros(1, dtype=np.float32)

    def _layer_norm_apply(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        mu = x.mean()
        sigma = np.sqrt(((x - mu) ** 2).mean() + eps)
        return gamma * ((x - mu) / sigma) + beta

    def forward(self, state: np.ndarray) -> float:
        """Estimate V(s) for a given state vector."""
        x = self.W1 @ state + self.b1
        x = self._layer_norm_apply(x, self.ln1_g, self.ln1_b)
        x = _relu(x)

        x = self.W2 @ x + self.b2
        x = self._layer_norm_apply(x, self.ln2_g, self.ln2_b)
        x = _relu(x)

        v = float((self.Wo @ x + self.bo)[0])
        return v

    def save(self, path: Optional[str] = None) -> str:
        save_path = Path(path or MODEL_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        weights = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "ln1_g": self.ln1_g, "ln1_b": self.ln1_b,
            "ln2_g": self.ln2_g, "ln2_b": self.ln2_b,
            "Wo": self.Wo, "bo": self.bo,
            "hidden_size": HIDDEN_SIZE,
        }
        with open(save_path, "wb") as f:
            pickle.dump(weights, f, protocol=5)
        logger.info("[ValueNetwork] Saved → %s", save_path)
        return str(save_path)

    def load(self, path: Optional[str] = None) -> bool:
        load_path = Path(path or MODEL_PATH)
        if not load_path.exists():
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
            saved_hidden = weights.get("hidden_size", 64)
            if saved_hidden != HIDDEN_SIZE:
                logger.warning(
                    "[ValueNetwork] Checkpoint hidden_size=%d ≠ current=%d — discarding.",
                    saved_hidden, HIDDEN_SIZE,
                )
                return False

            self.W1 = weights["W1"]; self.b1 = weights["b1"]
            self.W2 = weights["W2"]; self.b2 = weights["b2"]
            self.ln1_g = weights.get("ln1_g", np.ones(HIDDEN_SIZE, dtype=np.float32))
            self.ln1_b = weights.get("ln1_b", np.zeros(HIDDEN_SIZE, dtype=np.float32))
            self.ln2_g = weights.get("ln2_g", np.ones(HIDDEN_SIZE, dtype=np.float32))
            self.ln2_b = weights.get("ln2_b", np.zeros(HIDDEN_SIZE, dtype=np.float32))
            self.Wo = weights["Wo"]; self.bo = weights["bo"]
            logger.info("[ValueNetwork] Loaded checkpoint from %s", load_path)
            return True
        except Exception as exc:
            logger.warning("[ValueNetwork] Load failed: %s — reinitializing.", exc)
            return False
