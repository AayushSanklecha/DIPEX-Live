"""
verifier/confidence_vector.py
-----------------------------
Step 6 — Confidence Vector Aggregation.

Builds a multi-dimensional scoring vector and aggregates into a single
Confidence Score ∈ [0, 1]. Dimensions:

  - Data Quality Score   (from Hard Gate 1)
  - Statistical Score   (from statistical verifier)
  - Stability Score     (from CV / temporal stability)
  - Drift Robustness    (from drift / temporal robustness checks)
  - Compliance Score    (from domain/regulatory verifier)
  - Retry Penalty Score (decay per retry attempt)

Final confidence is a weighted combination, optionally with a global
failure penalty. Used to decide whether to trigger the Retry Engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Default weights: must sum to 1.0 for interpretability
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "data_quality": 0.20,
    "statistical": 0.22,
    "stability": 0.18,
    "drift_robustness": 0.15,
    "compliance": 0.15,
    "retry_penalty": 0.10,
}
_FAILURE_PENALTY: float = 0.5  # multiply final score when any gate failed


@dataclass
class ConfidenceVector:
    """Per-dimension scores and final aggregate."""

    data_quality_score: float = 0.0
    statistical_score: float = 0.0
    stability_score: float = 0.0
    drift_robustness_score: float = 0.0
    compliance_score: float = 0.0
    retry_penalty_score: float = 1.0
    confidence_score: float = 0.0
    all_gates_passed: bool = True
    vector: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_quality_score": round(self.data_quality_score, 6),
            "statistical_score": round(self.statistical_score, 6),
            "stability_score": round(self.stability_score, 6),
            "drift_robustness_score": round(self.drift_robustness_score, 6),
            "compliance_score": round(self.compliance_score, 6),
            "retry_penalty_score": round(self.retry_penalty_score, 6),
            "confidence_score": round(self.confidence_score, 6),
            "all_gates_passed": self.all_gates_passed,
            "vector": {k: round(v, 6) for k, v in self.vector.items()},
            "details": self.details,
        }


class ConfidenceVectorAggregator:
    """
    Builds the 6-dimensional confidence vector from Hard Gate 1,
    Hard Gate 2 / verifier results, and retry attempt count; aggregates
    into a final Confidence Score ∈ [0, 1].
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        conf_cfg = self.config.get("pipeline", {}).get("confidence", {})
        # Load base weights from config, then optional learned override from disk
        base_weights: Dict[str, float] = dict(_DEFAULT_WEIGHTS, **conf_cfg.get("weights", {}))
        learned_weights = self._load_learned_weights()
        self._weights: Dict[str, float] = dict(base_weights, **(learned_weights or {}))
        # Normalise weights to sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}
        self._failure_penalty: float = float(
            conf_cfg.get("failure_penalty", _FAILURE_PENALTY)
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfidenceVectorAggregator":
        return cls(config)

    def _load_learned_weights(self) -> Optional[Dict[str, float]]:
        """
        Loads learned confidence weights from a dedicated state file.
        This never mutates config.yaml; it is an efficiency-only calibration input.
        """
        path_str = (
            (self.config.get("storage", {}) or {}).get("confidence_weights_state")
            or "data/state/confidence_weights.json"
        )
        path = Path(path_str)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            weights = obj.get("weights")
            if isinstance(weights, dict):
                return {k: float(v) for k, v in weights.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load learned confidence weights: %s", exc)
        return None

    def aggregate(
        self,
        gate1_result: Optional[Dict[str, Any]] = None,
        gate2_confidence: Optional[Dict[str, Any]] = None,
        retry_attempt: int = 0,
    ) -> ConfidenceVector:
        """
        Builds per-dimension scores from gate results and retry count,
        then aggregates into a single confidence score.

        Args:
            gate1_result: Hard Gate 1 result (decision, failures, warnings).
            gate2_confidence: Hard Gate 2 confidence dict (vector with passed flags / values).
            retry_attempt: Current retry attempt index (0 = first run).

        Returns:
            ConfidenceVector with all 6 scores and final confidence_score ∈ [0, 1].
        """
        gate1 = gate1_result or {}
        gate2 = gate2_confidence or {}
        vector_raw = gate2.get("vector") or {}

        # 1. Data Quality Score (from Hard Gate 1)
        data_quality_score = self._data_quality_score(gate1)
        # 2. Statistical Score
        statistical_score = self._verifier_dimension_score(vector_raw.get("statistical"))
        # 3. Stability Score
        stability_score = self._verifier_dimension_score(vector_raw.get("stability"))
        # 4. Drift Robustness (drift_robustness or drift)
        drift_res = vector_raw.get("drift_robustness") or vector_raw.get("drift")
        drift_robustness_score = self._verifier_dimension_score(drift_res)
        # 5. Compliance Score
        compliance_score = self._verifier_dimension_score(vector_raw.get("domain"))
        # 6. Retry Penalty Score (decay with attempt count)
        retry_penalty_score = self._retry_penalty_score(retry_attempt)

        all_passed = bool(gate2.get("all_gates_passed", True))
        if gate1.get("decision") == "REJECT":
            all_passed = False

        vector = {
            "data_quality": data_quality_score,
            "statistical": statistical_score,
            "stability": stability_score,
            "drift_robustness": drift_robustness_score,
            "compliance": compliance_score,
            "retry_penalty": retry_penalty_score,
        }

        total = 0.0
        total_weight = 0.0
        for key, weight in self._weights.items():
            val = vector.get(key, 0.0)
            total += val * weight
            total_weight += weight

        confidence_score = total / total_weight if total_weight > 0 else 0.0
        if not all_passed:
            confidence_score *= self._failure_penalty
            logger.warning(
                "One or more gates failed; applying failure penalty (%.0f%%).",
                self._failure_penalty * 100,
            )

        confidence_score = max(0.0, min(1.0, confidence_score))

        out = ConfidenceVector(
            data_quality_score=data_quality_score,
            statistical_score=statistical_score,
            stability_score=stability_score,
            drift_robustness_score=drift_robustness_score,
            compliance_score=compliance_score,
            retry_penalty_score=retry_penalty_score,
            confidence_score=confidence_score,
            all_gates_passed=all_passed,
            vector=vector,
            details={
                "gate1_decision": gate1.get("decision"),
                "gate2_vector_keys": list(vector_raw.keys()),
                "retry_attempt": retry_attempt,
            },
        )
        logger.info(
            "Confidence vector aggregated: score=%.4f (data_quality=%.2f statistical=%.2f "
            "stability=%.2f drift_robust=%.2f compliance=%.2f retry_penalty=%.2f)",
            out.confidence_score,
            out.data_quality_score,
            out.statistical_score,
            out.stability_score,
            out.drift_robustness_score,
            out.compliance_score,
            out.retry_penalty_score,
        )
        return out

    @staticmethod
    def _data_quality_score(gate1: Dict[str, Any]) -> float:
        """Derive [0, 1] from Hard Gate 1: REJECT=0; PASS with no warnings=1."""
        if gate1.get("decision") == "REJECT":
            return 0.0
        n_warn = int(gate1.get("total_warnings", 0))
        if n_warn == 0:
            return 1.0
        # Linear decay: each warning reduces score (floor at 0.5 — PASS with warnings)
        return float(max(0.0, min(1.0, 1.0 - n_warn * 0.05)))

    @staticmethod
    def _verifier_dimension_score(res: Any) -> float:
        """
        Map a verifier result dict to [0, 1].
        - None / missing → 0.5 (neutral / not evaluated)
        - passed=True   → 1.0
        - passed=False  → 0.0
        - p-value field → 1 - p_value (smaller p = stronger result = higher score)
        - improvement   → 0.5 + clamped delta
        Always clamped to [0, 1].
        """
        if res is None:
            return 0.5
        if isinstance(res, (int, float)):
            return float(max(0.0, min(1.0, res)))
        if isinstance(res, dict):
            passed = res.get("passed")
            if passed is True:
                return 1.0
            if passed is False:
                return 0.0
            # Optional: use numeric value for partial score
            val = res.get("value")
            metric = str(res.get("metric", ""))
            if val is not None and isinstance(val, (int, float)):
                if "p_value" in metric:
                    # Lower p-value = stronger evidence → higher score
                    return float(max(0.0, min(1.0, 1.0 - float(val))))
                if "improvement" in metric.lower():
                    return float(max(0.0, min(1.0, 0.5 + float(val))))
                # Generic numeric [0,1] pass-through
                return float(max(0.0, min(1.0, float(val))))
        return 0.5

    @staticmethod
    def _retry_penalty_score(attempt: int) -> float:
        """
        Decay score with retry count.
        0 retries = 1.0; each retry reduces by 0.1 (floor 0.0, not 0.5 —
        allow heavy penalty for many retries).
        Always clamped to [0, 1].
        """
        if attempt <= 0:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - 0.1 * min(attempt, 10))))
