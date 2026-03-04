"""
ingestion/pipeline_bridge_helpers.py
--------------------------------------
Helper utilities for PipelineBridge — specifically the bandit Q-table
that drives the Intelligent Retry Engine's strategy selection.

Architecture
------------
The retry engine in PipelineBridge uses a contextual bandit (Q-learning)
to select which data transformation strategy to try on each retry attempt.
Q-values are persisted across pipeline runs so the agent continually
improves its strategy preferences.

Q-table schema (JSON)
---------------------
{
    "<strategy_name>": <float Q-value>,
    ...
}

Where Q-value > 0.5 = preferred, < 0.5 = penalised.
Initial prior = 0.5 (neutral / unexplored).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger("dipex.ingestion.pipeline_bridge_helpers")

# Default strategy set (must match PipelineBridge._select_retry_strategy candidates)
_DEFAULT_STRATEGIES = [
    "resample_oversample",
    "feature_prune_aggressive",
    "impute_knn",
    "outlier_winsorize",
    "scale_robust",
]
_INITIAL_Q = 0.5   # neutral prior for unexplored strategies


def _bandit_state_path(config: Dict[str, Any]) -> str:
    """Resolve the bandit Q-table file path from config or default."""
    state_dir = (
        config.get("storage", {}).get("state_dir")
        or config.get("pipeline", {}).get("state_dir")
        or "data/state"
    )
    return os.path.join(state_dir, "bandit_state.json")


def load_bandit_state(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Load the retry-strategy Q-table from disk.

    Returns a dict mapping strategy name → Q-value (float).
    If the file does not exist or is corrupt, returns an empty dict
    (callers treat missing keys as the neutral prior _INITIAL_Q).

    Parameters
    ----------
    config : DIPEX pipeline config dict (from config.yaml).

    Returns
    -------
    dict[str, float]
        strategy_name → Q-value
    """
    path = _bandit_state_path(config)
    if not os.path.exists(path):
        logger.debug("Bandit state not found at %s — starting fresh.", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Validate: must be dict[str, float]
        q_table = {str(k): float(v) for k, v in raw.items()}
        logger.debug("Bandit state loaded: %d strategies from %s", len(q_table), path)
        return q_table
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bandit state corrupt at %s (%s) — resetting to empty.", path, exc)
        return {}


def save_bandit_state(config: Dict[str, Any], q_table: Dict[str, float]) -> None:
    """
    Persist the retry-strategy Q-table to disk atomically.

    Parameters
    ----------
    config  : DIPEX pipeline config dict.
    q_table : strategy_name → Q-value mapping to save.
    """
    path = _bandit_state_path(config)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(q_table, f, indent=2, sort_keys=True)
        # Atomic rename (POSIX + Windows)
        os.replace(tmp_path, path)
        logger.debug("Bandit state saved: %d strategies to %s", len(q_table), path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bandit state save failed: %s", exc)


def update_bandit_q(
    q_table: Dict[str, float],
    strategy: str,
    reward: float,
    alpha: float = 0.15,
    gamma: float = 0.0,
) -> Dict[str, float]:
    """
    Apply a single Q-learning update for one strategy.

    Q(s, a) ← Q(s, a) + α × (reward − Q(s, a))
    (γ=0 since we treat each retry as a one-step bandit problem)

    Parameters
    ----------
    q_table  : Current Q-table (mutated in-place and returned).
    strategy : Strategy name that was just executed.
    reward   : Observed reward signal (e.g. confidence_delta, +1/-1).
    alpha    : Learning rate (default 0.15).
    gamma    : Discount factor (0.0 for bandits — no future lookahead).

    Returns
    -------
    Updated q_table dict (same object, mutated).
    """
    current_q = q_table.get(strategy, _INITIAL_Q)
    new_q = current_q + alpha * (reward + gamma * 0.0 - current_q)
    q_table[strategy] = round(new_q, 6)
    logger.debug(
        "Bandit Q-update: strategy=%s reward=%.3f Q: %.4f → %.4f",
        strategy, reward, current_q, new_q,
    )
    return q_table


def record_retry_outcome(
    config: Dict[str, Any],
    strategy: str,
    confidence_before: float,
    confidence_after: float,
) -> None:
    """
    Convenience wrapper: load Q-table, apply update for the strategy used
    in a retry attempt, and persist back to disk.

    The reward signal is the signed confidence improvement Δ:
      - Δ > 0 : strategy helped   → positive reward
      - Δ ≤ 0 : strategy did not help → negative reward proportional to regression

    Parameters
    ----------
    config             : DIPEX pipeline config dict.
    strategy           : Strategy that was applied in the retry.
    confidence_before  : Confidence score before applying the strategy.
    confidence_after   : Confidence score after applying the strategy.
    """
    q_table = load_bandit_state(config)
    reward = confidence_after - confidence_before   # signed confidence delta
    update_bandit_q(q_table, strategy, reward)
    save_bandit_state(config, q_table)
    logger.info(
        "Bandit: strategy=%s delta=%+.4f → Q=%.4f",
        strategy, reward, q_table[strategy],
    )
