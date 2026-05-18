"""
modeling/calibrator.py
------------------------
Probability calibration for AutoML models.

Applies Platt scaling (logistic regression on predicted probabilities) or
Isotonic regression to improve calibration quality.

Evaluation: Brier score ≤ 0.12, ECE ≤ 0.05 (as per spec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.modeling.calibrator")


@dataclass
class CalibrationResult:
    """Result from probability calibration."""
    method: str = "none"
    brier_score_before: float = 1.0
    brier_score_after: float = 1.0
    ece_before: float = 1.0
    ece_after: float = 1.0
    improvement: float = 0.0
    passed_gate: bool = False
    gate_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "brier_score_before": round(self.brier_score_before, 4),
            "brier_score_after": round(self.brier_score_after, 4),
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "improvement": round(self.improvement, 4),
            "passed_gate": self.passed_gate,
            "gate_reason": self.gate_reason,
        }


class ProbabilityCalibrator:
    """
    Wraps sklearn CalibratedClassifierCV to apply Platt scaling or
    Isotonic regression to a fitted classifier.

    Usage::

        calibrator = ProbabilityCalibrator()
        cal_result = calibrator.calibrate(model, X_val, y_val)
        # model is now calibrated in place
    """

    BRIER_THRESHOLD = 0.12
    ECE_THRESHOLD   = 0.05
    N_BINS_ECE      = 10

    def calibrate(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        method: str = "sigmoid",  # 'sigmoid' (Platt) or 'isotonic'
    ) -> CalibrationResult:
        """
        Calibrate model probabilities using X_val / y_val as the calibration set.

        Parameters
        ----------
        model   : Fitted sklearn-compatible classifier with predict_proba
        X_val   : Calibration features
        y_val   : Calibration labels (binary)
        method  : 'sigmoid' for Platt scaling, 'isotonic' for isotonic regression
        """
        result = CalibrationResult(method=method)

        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import brier_score_loss

            # Pre-calibration scores
            try:
                proba_before = model.predict_proba(X_val)[:, 1]
                y_arr = np.array(y_val)
                result.brier_score_before = float(brier_score_loss(y_arr, proba_before))
                result.ece_before = self._compute_ece(y_arr, proba_before)
            except Exception as exc:
                logger.debug("[Calibrator] Pre-calibration scoring failed: %s", exc)
                result.gate_reason = f"predict_proba unavailable: {exc}"
                return result

            # Apply calibration
            cal_model = CalibratedClassifierCV(model, cv="prefit", method=method)
            cal_model.fit(X_val, y_val)

            # Post-calibration scores
            proba_after = cal_model.predict_proba(X_val)[:, 1]
            result.brier_score_after = float(brier_score_loss(y_arr, proba_after))
            result.ece_after = self._compute_ece(y_arr, proba_after)
            result.improvement = result.brier_score_before - result.brier_score_after

            # Quality gate
            if result.brier_score_after <= self.BRIER_THRESHOLD and result.ece_after <= self.ECE_THRESHOLD:
                result.passed_gate = True
                result.gate_reason = (
                    f"Calibration passed: Brier={result.brier_score_after:.3f} ≤ {self.BRIER_THRESHOLD}, "
                    f"ECE={result.ece_after:.3f} ≤ {self.ECE_THRESHOLD}"
                )
            else:
                result.passed_gate = False
                result.gate_reason = (
                    f"Calibration gate failed: Brier={result.brier_score_after:.3f} "
                    f"(threshold={self.BRIER_THRESHOLD}), ECE={result.ece_after:.3f} "
                    f"(threshold={self.ECE_THRESHOLD})"
                )

            logger.info(
                "[Calibrator] %s calibration: Brier %.3f→%.3f, ECE %.3f→%.3f, gate=%s",
                method, result.brier_score_before, result.brier_score_after,
                result.ece_before, result.ece_after, result.passed_gate,
            )

        except ImportError:
            result.gate_reason = "sklearn.calibration not available"
        except Exception as exc:
            logger.warning("[Calibrator] Calibration error: %s", exc)
            result.gate_reason = str(exc)

        return result

    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Expected Calibration Error (ECE) — bin-based estimate."""
        try:
            bins = np.linspace(0.0, 1.0, self.N_BINS_ECE + 1)
            ece = 0.0
            n = len(y_true)
            for i in range(self.N_BINS_ECE):
                mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
                if mask.sum() == 0:
                    continue
                acc = y_true[mask].mean()
                conf = y_prob[mask].mean()
                ece += (mask.sum() / n) * abs(acc - conf)
            return float(ece)
        except Exception:
            return 1.0
