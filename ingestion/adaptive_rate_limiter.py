"""
ingestion/adaptive_rate_limiter.py
------------------------------------
Production-grade Q-Learning Rate Limiter for API and DB data extraction.

Design
------
State   : source identifier (URL or "backend_host")
Actions :
  - API backoff_base multiplier: {1.0, 1.5, 2.0, 3.0, 5.0}
  - DB  chunk_size:              {1_000, 5_000, 10_000, 25_000, 50_000}

The Q-table is persisted per session to data/rl_rate_limiter.json and
improves with every pipeline run across the lifetime of the service.

Reward Function
---------------
  success + low latency → +reward (inversely proportional to ms)
  hard failure (429/timeout/exception) → -10 penalty
  partial failure (errors in result) → -2 penalty

Usage
-----
    from ingestion.adaptive_rate_limiter import RLRateLimiter, get_rl_agent

    agent = get_rl_agent()

    # Before API request:
    backoff = agent.get_api_backoff(url)

    # After request completes:
    agent.record_api_outcome(url, backoff, success=True, latency_ms=320, has_errors=False)

    # Before DB read:
    chunk_size = agent.get_db_chunk_size(backend, host)

    # After DB read completes:
    agent.record_db_outcome(backend, host, chunk_size, success=True, latency_ms=1800)
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional

logger = logging.getLogger("dipex.ingestion.adaptive_rate_limiter")

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_rate_limiter.json"
)

_API_ACTIONS:  List[float] = [1.0, 1.5, 2.0, 3.0, 5.0]  # backoff_base values
_DB_ACTIONS:   List[int]   = [1_000, 5_000, 10_000, 25_000, 50_000]  # chunk sizes

_ALPHA:    float = 0.1    # learning rate
_GAMMA:    float = 0.6    # discount factor
_EPSILON:  float = 0.15   # exploration rate (15 % random exploration)
_SAVE_PROB: float = 0.10  # probability of persisting Q-table on each update


class RLRateLimiter:
    """
    Q-Learning agent that adapts API backoff and DB chunk sizing
    to maximise ingestion throughput while minimising failures.
    """

    def __init__(self, state_path: str = _DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.q: Dict[str, Dict[str, float]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    self.q = json.load(fh)
                logger.info("RLRateLimiter: Q-table loaded (%d states).", len(self.q))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLRateLimiter: Could not load Q-table: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.q, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLRateLimiter: Could not save Q-table: %s", exc)

    # ── Q-table helpers ───────────────────────────────────────────────────────

    def _init_state(self, state: str, actions: List) -> None:
        if state not in self.q:
            self.q[state] = {str(a): 0.0 for a in actions}

    def _choose(self, state: str, actions: List) -> str:
        """Epsilon-greedy action selection."""
        self._init_state(state, actions)
        if random.random() < _EPSILON:
            return str(random.choice(actions))
        return max(self.q[state], key=self.q[state].__getitem__)

    def _update(self, state: str, action: str, reward: float) -> None:
        """Q(s,a) ← Q(s,a) + α · (r − Q(s,a))  [single-step bandit update]"""
        self._init_state(state, [])
        if action not in self.q[state]:
            self.q[state][action] = 0.0
        curr = self.q[state][action]
        self.q[state][action] = round(curr + _ALPHA * (reward - curr), 6)
        if random.random() < _SAVE_PROB:
            self.save()

    # ── API interface ─────────────────────────────────────────────────────────

    def get_api_backoff(self, url: str) -> float:
        """Return RL-selected backoff_base for an API source."""
        state  = f"api::{url}"
        action = self._choose(state, _API_ACTIONS)
        return float(action)

    def record_api_outcome(
        self,
        url:        str,
        backoff:    float,
        success:    bool,
        latency_ms: float,
        has_errors: bool = False,
    ) -> None:
        """Record the result of an API read and update Q-table."""
        state = f"api::{url}"
        if not success:
            reward = -10.0
        elif has_errors:
            reward = -2.0
        else:
            # Throughput reward: inversely proportional to latency (capped)
            reward = min(1000.0 / max(latency_ms, 10.0), 10.0)
        self._update(state, str(backoff), reward)
        logger.debug(
            "[RL] API %s | backoff=%.1f | latency=%.0fms | reward=%.2f",
            url, backoff, latency_ms, reward,
        )

    # ── DB interface ──────────────────────────────────────────────────────────

    def get_db_chunk_size(self, backend: str, host: str) -> int:
        """Return RL-selected chunk_size for a DB source."""
        state  = f"db::{backend}::{host}"
        action = self._choose(state, _DB_ACTIONS)
        return int(action)

    def record_db_outcome(
        self,
        backend:    str,
        host:       str,
        chunk_size: int,
        success:    bool,
        latency_ms: float,
    ) -> None:
        """Record the result of a DB read and update Q-table."""
        state = f"db::{backend}::{host}"
        if not success:
            reward = -10.0
        else:
            reward = min(1000.0 / max(latency_ms, 10.0), 10.0)
        self._update(state, str(chunk_size), reward)
        logger.debug(
            "[RL] DB %s@%s | chunk=%d | latency=%.0fms | reward=%.2f",
            backend, host, chunk_size, latency_ms, reward,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_best_actions(self) -> Dict[str, str]:
        """Return a dict of {state: best_action} for all learned states."""
        return {s: max(v, key=v.__getitem__) for s, v in self.q.items() if v}


# ── Module-level singleton ────────────────────────────────────────────────────

_AGENT: Optional[RLRateLimiter] = None


def get_rl_agent() -> RLRateLimiter:
    """Return the module-level singleton RLRateLimiter."""
    global _AGENT  # noqa: PLW0603
    if _AGENT is None:
        _AGENT = RLRateLimiter()
    return _AGENT
