"""
proposal/rl_strategy.py
------------------------
RL-based strategy bias proposals using a lightweight Q-table.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RLStrategyProposal:
    """
    Lightweight RL strategy selection via a persistent Q-table.

    Algorithm selection uses a softmax-weighted random draw (exploration),
    where Q-values represent accumulated reward signals from past pipeline runs.
    """

    def __init__(self, storage_path: str = "data/rl_state.json") -> None:
        self.storage_path = Path(storage_path)
        self._load_q_table()

    def _load_q_table(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.q_table: Dict[str, Dict[str, float]] = json.load(f)
                logger.debug("Q-table loaded from %s", self.storage_path)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Q-table corrupt/unreadable (%s). Resetting to defaults.", exc)
                self._default_q_table()
        else:
            self._default_q_table()

    def _default_q_table(self) -> None:
        self.q_table = {
            "regression": {
                "RandomForest": 1.0,
                "GradientBoosting": 1.0,
                "LinearRegression": 1.0,
            },
            "classification": {
                "RandomForest": 1.0,
                "GradientBoosting": 1.0,
                "LogisticRegression": 1.0,
            },
        }

    def _save_q_table(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.q_table, f, indent=2)

    def propose_strategy(self, task_type: str) -> str:
        """
        Selects an algorithm family using softmax-weighted random selection
        over current Q-values (exploration-exploitation balance).
        """
        options = self.q_table.get(task_type, {})
        if not options:
            logger.warning("Unknown task_type '%s'; defaulting to RandomForest.", task_type)
            return "RandomForest"

        algorithms = list(options.keys())
        weights = list(options.values())

        # Softmax: exp(w) weighting avoids the full random collapse
        exp_weights = [pow(2, w) for w in weights]
        total = sum(exp_weights)
        probs = [ew / total for ew in exp_weights]

        chosen = random.choices(algorithms, weights=probs, k=1)[0]
        logger.debug("RL proposal for task=%s → %s", task_type, chosen)
        return chosen

    def update_q_value(
        self, task_type: str, algorithm: str, reward: float, alpha: float = 0.1
    ) -> None:
        """
        Updates the Q-value for an (algorithm, task) pair via a simple
        exponential moving average update rule:
            Q ← (1 − α) · Q + α · reward

        Args:
            alpha: Learning rate (0 = no update; 1 = fully replace with reward).
        """
        if task_type not in self.q_table:
            logger.warning("task_type '%s' not in Q-table; skipping update.", task_type)
            return
        if algorithm not in self.q_table[task_type]:
            logger.warning(
                "Algorithm '%s' not in Q-table for task '%s'; skipping update.",
                algorithm,
                task_type,
            )
            return

        old_val = self.q_table[task_type][algorithm]
        new_val = (1 - alpha) * old_val + alpha * reward
        self.q_table[task_type][algorithm] = new_val
        self._save_q_table()
        logger.info(
            "Q-update: task=%s alg=%s reward=%.3f  Q: %.3f → %.3f",
            task_type, algorithm, reward, old_val, new_val,
        )
