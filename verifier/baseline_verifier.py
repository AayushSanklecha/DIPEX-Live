"""
verifier/baseline_verifier.py
------------------------------
Ensures the proposed model beats a naive baseline predictor.
"""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)


class BaselineVerifier:
    """Ensures proposed model beats a naive baseline predictor."""

    def __init__(self, min_improvement: float = 0.05) -> None:
        self.min_improvement = min_improvement

    def verify(
        self,
        model_score: float,
        task_type: str,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compares model score against a naive baseline (mean for regression,
        majority-class for classification).
        """
        if task_type == "regression":
            return self._verify_regression(model_score, y_train)
        else:
            return self._verify_classification(model_score, y_train)

    def _verify_regression(
        self, model_score: float, y_train: np.ndarray
    ) -> Dict[str, Any]:
        baseline_val = float(np.mean(y_train))
        baseline_pred = np.full_like(y_train, baseline_val, dtype=float)
        baseline_score = float(mean_squared_error(y_train, baseline_pred))

        if baseline_score == 0:
            # Target is constant — model already matches baseline perfectly
            passed = model_score == 0.0
            improvement = 0.0
        else:
            improvement = (baseline_score - model_score) / baseline_score
            passed = bool(model_score < baseline_score * (1 - self.min_improvement))

        return {
            "metric": "baseline_improvement",
            "value": float(improvement),
            "passed": passed,
            "detail": (
                f"MSE improvement over mean-baseline: {improvement:.2%} "
                f"(minimum required: {self.min_improvement:.2%})"
            ),
        }

    def _verify_classification(
        self, model_score: float, y_train: np.ndarray
    ) -> Dict[str, Any]:
        # Cast safely to int; handle float targets gracefully
        try:
            y_int = y_train.astype(int)
        except (ValueError, TypeError):
            logger.warning(
                "Could not cast y_train to int for majority-class baseline. "
                "Using mean-rounding fallback."
            )
            y_int = np.round(y_train).astype(int)

        counts = np.bincount(y_int - y_int.min())  # offset to avoid negative indices
        majority_class = int(np.argmax(counts)) + y_int.min()
        baseline_pred = np.full_like(y_train, majority_class)
        baseline_score = float(accuracy_score(y_train, baseline_pred))

        improvement = model_score - baseline_score
        passed = bool(model_score > baseline_score + self.min_improvement)

        return {
            "metric": "baseline_improvement",
            "value": float(improvement),
            "passed": passed,
            "detail": (
                f"Accuracy improvement over majority-class baseline: {improvement:.2%} "
                f"(minimum required: {self.min_improvement:.2%})"
            ),
        }
