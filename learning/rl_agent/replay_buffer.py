"""
learning/rl_agent/replay_buffer.py
-------------------------------------
On-policy trajectory ring buffer for PPO training.

Stores (state, action_indices, log_prob, reward, value, done) tuples.
Supports GAE (Generalized Advantage Estimation) computation.

PPO hyperparameters (elite-grade defaults):
  γ  (gamma)  = 0.99   — discount factor
  λ  (lambda) = 0.95   — GAE bias-variance tradeoff
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


GAMMA  = 0.99
LAMBDA = 0.95


@dataclass
class Transition:
    state: np.ndarray
    action_indices: List[int]
    log_prob: float
    reward: float
    value: float
    done: bool


class ReplayBuffer:
    """
    Fixed-capacity on-policy ring buffer for PPO trajectories.

    Capacity is measured in episodes (not transitions).
    When capacity is reached, oldest episodes are discarded.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._episodes: List[List[Transition]] = []
        self._current_episode: List[Transition] = []

    # ── Episode management ───────────────────────────────────────────────────

    def add_transition(
        self,
        state: np.ndarray,
        action_indices: List[int],
        log_prob: float,
        reward: float,
        value: float,
        done: bool = False,
    ) -> None:
        """Add a single step transition to the current episode."""
        self._current_episode.append(Transition(
            state=state.copy(),
            action_indices=list(action_indices),
            log_prob=float(log_prob),
            reward=float(reward),
            value=float(value),
            done=bool(done),
        ))
        if done:
            self.commit_episode()

    def commit_episode(self) -> None:
        """Commit the current episode to the buffer."""
        if self._current_episode:
            self._episodes.append(list(self._current_episode))
            self._current_episode = []
            # Ring buffer: evict oldest if over capacity
            if len(self._episodes) > self.capacity:
                self._episodes.pop(0)

    # ── Batch extraction ─────────────────────────────────────────────────────

    def compute_gae(self, next_value: float = 0.0) -> List[Tuple[np.ndarray, List[int], float, float, float]]:
        """
        Compute GAE advantages and returns over all stored transitions.
        Returns list of (state, action_indices, log_prob, advantage, return_)
        """
        results = []
        for episode in self._episodes:
            states    = [t.state          for t in episode]
            actions   = [t.action_indices for t in episode]
            log_probs = [t.log_prob       for t in episode]
            rewards   = [t.reward         for t in episode]
            values    = [t.value          for t in episode]
            dones     = [t.done           for t in episode]

            T = len(episode)
            advantages = np.zeros(T, dtype=np.float32)
            returns    = np.zeros(T, dtype=np.float32)

            gae = 0.0
            nv  = next_value if not dones[-1] else 0.0
            for t in reversed(range(T)):
                next_val = nv if t == T - 1 else values[t + 1]
                delta = rewards[t] + GAMMA * next_val * (1 - float(dones[t])) - values[t]
                gae   = delta + GAMMA * LAMBDA * (1 - float(dones[t])) * gae
                advantages[t] = gae
                returns[t]    = advantages[t] + values[t]

            # Normalize advantages
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for i in range(T):
                results.append((states[i], actions[i], log_probs[i],
                                float(advantages[i]), float(returns[i])))

        return results

    def get_mini_batches(self, batch_size: int = 32) -> List[List[Tuple]]:
        """Shuffle transitions and split into mini-batches."""
        all_transitions = self.compute_gae()
        np.random.shuffle(all_transitions)  # type: ignore[arg-type]
        return [
            all_transitions[i: i + batch_size]
            for i in range(0, len(all_transitions), batch_size)
        ]

    def clear(self) -> None:
        """Clear all stored episodes (call after PPO update)."""
        self._episodes.clear()
        self._current_episode.clear()

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def n_transitions(self) -> int:
        return sum(len(ep) for ep in self._episodes)

    def summary(self) -> Dict[str, Any]:
        all_rewards = [t.reward for ep in self._episodes for t in ep]
        return {
            "n_episodes": self.n_episodes,
            "n_transitions": self.n_transitions,
            "mean_reward": round(float(np.mean(all_rewards)), 4) if all_rewards else 0.0,
            "std_reward":  round(float(np.std(all_rewards)), 4)  if all_rewards else 0.0,
            "capacity": self.capacity,
        }
