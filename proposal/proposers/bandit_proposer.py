"""
proposal/proposers/bandit_proposer.py
--------------------------------------
Selects strategic policies using a Contextual Bandit (Q-table based).
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)

class BanditProposer(BaseProposer):
    """
    Suggests strategic policies (retry, encoding, etc.) based on past rewards.
    Essentially the Step 4 interface for RLStrategyProposal logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = self.config.get("proposal", {}).get("bandit", {})
        self.storage_path = Path(cfg.get("storage_path", "data/bandit_state.json"))
        self.exploration_rate = float(cfg.get("exploration_rate", 0.1))
        self._load_state()

    def _load_state(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self.q_table: Dict[str, Dict[str, float]] = json.load(f)
            except Exception as e:
                logger.warning("Bandit state corrupt: %s. Resetting.", e)
                self._init_defaults()
        else:
            self._init_defaults()

    def _init_defaults(self) -> None:
        # Default strategic policies for multiple contexts
        self.q_table = {
            "retry_strategy": {
                "exponential_backoff": 1.0,
                "immediate_retry": 0.5,
                "abort_and_log": 0.8
            },
            "encoding_policy": {
                "one_hot": 1.0,
                "label_encoding": 0.9,
                "target_encoding": 1.1
            },
            # Step 7 — Intelligent Retry: additional knobs
            "transformation_policy": {
                "prefer_log_transforms": 1.0,
                "prefer_scaling": 1.0,
                "no_change": 0.8,
            },
            "window_policy": {
                "shorter_windows": 1.0,
                "longer_windows": 1.0,
                "adaptive_windows": 1.1,
            },
            "model_family": {
                "RandomForest": 1.0,
                "GradientBoosting": 1.0,
                "LinearModel": 0.9,
            },
            "feature_subset_policy": {
                "drop_high_correlation": 1.0,
                "drop_high_cardinality": 1.0,
                "keep_all": 0.8,
            }
        }

    def _save_state(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.q_table, f, indent=2)

    def propose(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Suggests policies for given contexts.
        """
        contexts = kwargs.get("contexts", ["retry_strategy", "encoding_policy"])
        suggestions = {}

        for ctx in contexts:
            options = self.q_table.get(ctx, {})
            if not options:
                continue

            # epsilon-greedy / softmax weighted draw
            algs = list(options.keys())
            weights = list(options.values())
            
            # Use power weight for selection pressure
            exp_weights = [pow(2, w) for w in weights]
            total = sum(exp_weights)
            probs = [ew / total for ew in exp_weights]

            chosen = random.choices(algs, weights=probs, k=1)[0]
            suggestions[ctx] = {
                "recommendation": chosen,
                "q_value": round(options[chosen], 4),
                "candidates": options
            }

        return {"strategy_candidates": suggestions}
