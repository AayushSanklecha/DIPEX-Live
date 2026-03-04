"""
verifier/confidence_aggregator.py
----------------------------------
Aggregates all verifier results into a single confidence score.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfidenceAggregator:
    """
    Aggregates all verifier results into a weighted confidence score in [0, 1].

    Weight assignment rationale:
      - statistical (0.30): Core significance test — highest trust signal.
      - baseline    (0.25): Beat a naive predictor — strong quality gate.
      - stability   (0.20): Cross-validation variance — model robustness.
      - domain      (0.15): Domain rule compliance — business safety layer.
      - drift       (0.10): Distribution drift — production health signal.

    Penalty policy:
      If *any* verifier gate fails, the final score is multiplied by 0.5.
      Rationale: a single hard-gate failure is a systemic red flag that
      should substantially lower confidence regardless of what the other
      verifiers report.  The 0.5 penalty is deliberately large to bias
      the system toward caution when a gate fails.
    """

    _FAILURE_PENALTY: float = 0.5

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights: Dict[str, float] = weights or {
            "statistical": 0.30,
            "baseline":    0.25,
            "stability":   0.20,
            "domain":      0.15,
            "drift":       0.10,
        }

    def aggregate(self, verifier_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes a weighted confidence score from individual verifier results.

        Args:
            verifier_results: Mapping of verifier_name → verifier result dict.
                              Each result dict must contain at minimum a boolean ``passed`` key.

        Returns:
            Dict with keys:
              - ``confidence_score``  (float, 0–1)
              - ``all_gates_passed``  (bool)
              - ``vector``            (dict, per-verifier breakdown)
        """
        total_score: float = 0.0
        total_weight: float = 0.0
        all_passed: bool = True
        details: Dict[str, Any] = {}

        for key, res in verifier_results.items():
            weight = self.weights.get(key, 0.0)
            passed = bool(res.get("passed", False))
            score = 1.0 if passed else 0.0

            if not passed:
                all_passed = False

            total_score += score * weight
            total_weight += weight
            details[key] = res

        if total_weight == 0.0:
            logger.warning("ConfidenceAggregator received no weighted verifier results.")
            final_confidence = 0.0
        else:
            final_confidence = total_score / total_weight

        if not all_passed:
            final_confidence *= self._FAILURE_PENALTY
            logger.warning(
                "One or more verifier gates failed; applying %.0f%% confidence penalty.",
                self._FAILURE_PENALTY * 100,
            )

        logger.info("Aggregated confidence score: %.2f%%", final_confidence * 100)

        return {
            "confidence_score": float(final_confidence),
            "all_gates_passed": all_passed,
            "vector": details,
        }
