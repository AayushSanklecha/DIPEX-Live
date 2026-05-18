"""
learning/rl_agent/action_space.py
-----------------------------------
Defines the 8-axis mixed discrete/continuous action space for the PPO agent.

Action axes:
  0  cv_folds              discrete: {3, 5, 7, 10}
  1  cv_strategy           discrete: {temporal, stratified, kfold, group}
  2  confidence_threshold  continuous: [0.40, 0.90]
  3  imputation            discrete: {median, knn, mice, forward_fill}
  4  outlier_policy        discrete: {winsorize, flag, quarantine, preserve}
  5  model_complexity      discrete: {low, medium, high}
  6  insight_ranker        discrete: {drift_heavy, quality_heavy, balanced}
  7  retry_budget          discrete: {0, 1, 2, 3}

For PPO we discretize confidence_threshold into 10 bins to keep the
action space fully discrete and compatible with categorical sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Action definitions ────────────────────────────────────────────────────────

CV_FOLDS         = [3, 5, 7, 10]
CV_STRATEGIES    = ["temporal", "stratified", "kfold", "group"]
CONFIDENCE_BINS  = [round(0.40 + i * 0.056, 3) for i in range(10)]  # 0.40..0.90
IMPUTATION_STRATS = ["median", "knn", "mice", "forward_fill"]
OUTLIER_POLICIES = ["winsorize", "flag", "quarantine", "preserve"]
MODEL_COMPLEXITY = ["low", "medium", "high"]
INSIGHT_RANKERS  = ["drift_heavy", "quality_heavy", "balanced"]
RETRY_BUDGETS    = [0, 1, 2, 3]

# Axes in order: (axis_name, options_list)
AXES: List[Tuple[str, List[Any]]] = [
    ("cv_folds",             CV_FOLDS),
    ("cv_strategy",          CV_STRATEGIES),
    ("confidence_threshold", CONFIDENCE_BINS),
    ("imputation",           IMPUTATION_STRATS),
    ("outlier_policy",       OUTLIER_POLICIES),
    ("model_complexity",     MODEL_COMPLEXITY),
    ("insight_ranker",       INSIGHT_RANKERS),
    ("retry_budget",         RETRY_BUDGETS),
]

N_AXES = len(AXES)
AXIS_SIZES = [len(opts) for _, opts in AXES]
TOTAL_ACTIONS = sum(AXIS_SIZES)  # flat action count for multi-head softmax


@dataclass
class PipelineAction:
    """Decoded action from integer action indices."""
    cv_folds: int              = 5
    cv_strategy: str           = "stratified"
    confidence_threshold: float = 0.70
    imputation: str            = "median"
    outlier_policy: str        = "winsorize"
    model_complexity: str      = "medium"
    insight_ranker: str        = "balanced"
    retry_budget: int          = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cv_folds": self.cv_folds,
            "cv_strategy": self.cv_strategy,
            "confidence_threshold": self.confidence_threshold,
            "imputation": self.imputation,
            "outlier_policy": self.outlier_policy,
            "model_complexity": self.model_complexity,
            "insight_ranker": self.insight_ranker,
            "retry_budget": self.retry_budget,
        }


class ActionSpace:
    """
    Manages the 8-axis discrete action space.

    The PPO agent uses N_AXES separate softmax heads, one per axis.
    decode() converts axis indices to a PipelineAction.
    """

    N_AXES = N_AXES
    AXIS_SIZES = AXIS_SIZES

    def decode(self, axis_indices: List[int]) -> PipelineAction:
        """Convert list of per-axis indices to a PipelineAction."""
        if len(axis_indices) != N_AXES:
            raise ValueError(f"Expected {N_AXES} axis indices, got {len(axis_indices)}")

        def _clip(idx: int, axis: int) -> int:
            return max(0, min(idx, AXIS_SIZES[axis] - 1))

        return PipelineAction(
            cv_folds             = CV_FOLDS[_clip(axis_indices[0], 0)],
            cv_strategy          = CV_STRATEGIES[_clip(axis_indices[1], 1)],
            confidence_threshold = CONFIDENCE_BINS[_clip(axis_indices[2], 2)],
            imputation           = IMPUTATION_STRATS[_clip(axis_indices[3], 3)],
            outlier_policy       = OUTLIER_POLICIES[_clip(axis_indices[4], 4)],
            model_complexity     = MODEL_COMPLEXITY[_clip(axis_indices[5], 5)],
            insight_ranker       = INSIGHT_RANKERS[_clip(axis_indices[6], 6)],
            retry_budget         = RETRY_BUDGETS[_clip(axis_indices[7], 7)],
        )

    def sample_random(self) -> Tuple[List[int], PipelineAction]:
        """Sample a random action (for exploration or warm-up)."""
        indices = [np.random.randint(0, sz) for sz in AXIS_SIZES]
        return indices, self.decode(indices)

    def default_action(self) -> Tuple[List[int], PipelineAction]:
        """Default safe action (balanced, medium settings)."""
        indices = [
            CV_FOLDS.index(5),
            CV_STRATEGIES.index("stratified"),
            4,  # 0.624 confidence
            IMPUTATION_STRATS.index("median"),
            OUTLIER_POLICIES.index("winsorize"),
            MODEL_COMPLEXITY.index("medium"),
            INSIGHT_RANKERS.index("balanced"),
            RETRY_BUDGETS.index(1),
        ]
        return indices, self.decode(indices)

    def constrained_indices(self, indices: List[int]) -> List[int]:
        """
        Apply safety constraints:
        - confidence_threshold never < 0.40 (index 0 is minimum)
        - retry_budget never > 3 (already capped by axis size)
        """
        # Confidence already starts at 0.40 (index 0), so no lower bound needed.
        # Just clip all indices to valid range.
        return [max(0, min(idx, AXIS_SIZES[i] - 1)) for i, idx in enumerate(indices)]
