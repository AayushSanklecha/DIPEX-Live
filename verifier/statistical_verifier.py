"""
verifier/statistical_verifier.py
---------------------------------
Verifies model results using statistical significance tests.
"""

import logging
from typing import Any, Dict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_MIN_SHAPIRO_SAMPLES: int = 3   # scipy requirement for Shapiro-Wilk
_MAX_SHAPIRO_SAMPLES: int = 5000  # Shapiro becomes unreliable above this


class StatisticalVerifier:
    """Verifies results using statistical significance tests."""

    def __init__(self, p_value_threshold: float = 0.05) -> None:
        self.p_value_threshold = p_value_threshold

    def verify(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
    ) -> Dict[str, Any]:
        """
        Runs significance tests on the model's predictions.

        For regression: Shapiro-Wilk test for residual normality.
        For classification: one-sided binomial test vs random chance.
        """
        n = len(y_true)
        if n == 0:
            logger.warning("StatisticalVerifier received empty arrays; returning not-passed.")
            return {
                "metric": "empty_input",
                "value": 0.0,
                "passed": False,
                "detail": "No samples to evaluate.",
            }

        if task_type == "regression":
            return self._verify_regression(y_true, y_pred)
        else:
            return self._verify_classification(y_true, y_pred)

    def _verify_regression(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        residuals = y_true - y_pred
        sample = residuals[:_MAX_SHAPIRO_SAMPLES]

        if len(sample) < _MIN_SHAPIRO_SAMPLES:
            return {
                "metric": "residual_normality_p_value",
                "value": 1.0,
                "passed": True,
                "detail": (
                    f"Too few samples ({len(sample)}) for Shapiro-Wilk — "
                    "assuming normality."
                ),
            }

        _, p_val = stats.shapiro(sample)
        return {
            "metric": "residual_normality_p_value",
            "value": float(p_val),
            "passed": bool(p_val > self.p_value_threshold),
            "detail": f"Residual normality (Shapiro-Wilk) p-value: {p_val:.4f}",
        }

    def _verify_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        n = len(y_true)
        unique_classes = np.unique(y_true)

        if len(unique_classes) <= 1:
            logger.warning(
                "Only one unique class found in y_true — binomial test not applicable."
            )
            return {
                "metric": "better_than_chance_p_value",
                "value": 1.0,
                "passed": False,
                "detail": "Cannot evaluate: only one class present in y_true.",
            }

        hits = int(np.sum(y_true == y_pred))
        random_chance = 1.0 / len(unique_classes)
        result = stats.binomtest(hits, n, random_chance, alternative="greater")
        p_val = float(result.pvalue)

        return {
            "metric": "better_than_chance_p_value",
            "value": p_val,
            "passed": bool(p_val < self.p_value_threshold),
            "detail": (
                f"Accuracy {hits}/{n} vs random chance {random_chance:.2%} — "
                f"p-value: {p_val:.4f}"
            ),
        }
