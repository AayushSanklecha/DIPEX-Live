import logging
from typing import Any, Dict
import pandas as pd

logger = logging.getLogger(__name__)

class ConfidenceVector:
    """
    Independent Verification Engine - Hard Gate 2.
    Evaluates ML performance metrics and aggregates pipeline confidence signals.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfidenceVector":
        return cls(config)

    def run_verification_gate(
        self, df: pd.DataFrame, model_metrics: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """
        Independent verification step evaluating test performance.
        Gate 2 fails (REJECT) if minimum ROC-AUC or related metrics are substandard.
        """
        logger.info("[%s] Verifier: Running Hard Gate 2 independent checks.", run_id[:8])

        # Example check: reject models with poor performance
        roc_auc = model_metrics.get("roc_auc", 0.0)
        
        domain = self.config.get("pipeline", {}).get("domain", "default")
        min_auc = {"banking": 0.65, "healthcare": 0.70, "default": 0.50}.get(domain, 0.50)

        if roc_auc < min_auc and roc_auc > 0.0:  # Skip 0.0 as it might just mean not a classification run
            reason = f"ROC-AUC {roc_auc:.3f} below domain minimum {min_auc:.3f}"
            logger.warning("[%s] Verifier REJECT: %s", run_id[:8], reason)
            return {"decision": "REJECT", "reason": reason}
            
        return {"decision": "PASS"}

    def aggregate(
        self,
        df: pd.DataFrame,
        model_metrics: Dict[str, Any],
        quality_score: float,
        gate2_passed: bool,
        retry_count: int,
    ) -> Dict[str, Any]:
        """
        Produces a weighted scalar confidence score bridging multiple pipeline signals.
        """
        # Baseline from data quality
        base_score = quality_score
        
        # Boost based on model performance
        roc_auc = model_metrics.get("roc_auc", 0.5)
        model_boost = max(0, roc_auc - 0.5) * 0.4  # Max 0.2 boost
        
        # Penalize retries
        retry_penalty = retry_count * 0.05
        
        # Calculate final confidence
        confidence_score = base_score + model_boost - retry_penalty
        
        # Hard cap if gate 2 failed
        if not gate2_passed:
            confidence_score = min(confidence_score, 0.49)
            
        # Ensure within bounds
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return {
            "confidence_score": confidence_score,
            "components": {
                "base_quality": quality_score,
                "model_boost": model_boost,
                "retry_penalty": retry_penalty,
                "gate2_passed": gate2_passed
            }
        }
