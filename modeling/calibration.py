"""
modeling/calibration.py
-------------------------
Model probability calibration for classification models.

Calibration methods:
  - Platt scaling     (sigmoid / logistic regression on raw scores)
  - Isotonic regression (non-parametric, monotone, more flexible)

Diagnostics:
  - Brier score before/after calibration
  - Expected Calibration Error (ECE) using adaptive binning
  - Reliability diagram data (for dashboard rendering)
  - Calibration improvement summary

Usage::

    cal = ModelCalibrator(method="isotonic")
    result = cal.calibrate(model, X_cal, y_cal)
    print(result.brier_score_before, result.brier_score_after)
    print(result.ece_after, result.improved)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.modeling.calibration")


@dataclass
class CalibrationResult:
    method: str
    brier_score_before: float
    brier_score_after: float
    ece_before: float
    ece_after: float
    improved: bool
    improvement_pct: float
    reliability_diagram: List[Dict[str, float]]   # [{bin_center, fraction_pos, mean_pred, count}]
    calibrated_model: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "brier_score_before": round(self.brier_score_before, 6),
            "brier_score_after": round(self.brier_score_after, 6),
            "ece_before": round(self.ece_before, 6),
            "ece_after": round(self.ece_after, 6),
            "improved": self.improved,
            "improvement_pct": round(self.improvement_pct, 2),
            "reliability_diagram": self.reliability_diagram,
            "interpretation": self._interpret(),
        }

    def _interpret(self) -> str:
        if self.improved:
            return (f"Calibration improved by {abs(self.improvement_pct):.1f}% "
                    f"(ECE: {self.ece_before:.4f} → {self.ece_after:.4f}). "
                    f"Model probabilities are better calibrated using {self.method}.")
        return (f"Calibration did not improve significantly. "
                f"ECE: {self.ece_before:.4f} → {self.ece_after:.4f}.")


class ModelCalibrator:
    """
    Probability calibration for sklearn-compatible classifiers.

    Usage::

        mc = ModelCalibrator(method="isotonic", cv="prefit")
        result = mc.calibrate(trained_model, X_holdout, y_holdout)
        # Use result.calibrated_model for final predictions
    """

    def __init__(self, method: str = "isotonic", n_bins: int = 10) -> None:
        assert method in ("platt", "sigmoid", "isotonic"), \
            "method must be 'platt'/'sigmoid' or 'isotonic'"
        self.method = "sigmoid" if method == "platt" else method
        self.n_bins = n_bins

    def calibrate(
        self,
        model: Any,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> CalibrationResult:
        """
        Calibrate model probabilities using a held-out calibration set.

        Parameters
        ----------
        model     : fitted sklearn classifier with predict_proba
        X_cal     : calibration features (held out from training)
        y_cal     : calibration labels
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import brier_score_loss

        if not hasattr(model, "predict_proba"):
            logger.warning("Model does not support predict_proba — calibration skipped.")
            empty = CalibrationResult(
                method=self.method, brier_score_before=0.0, brier_score_after=0.0,
                ece_before=0.0, ece_after=0.0, improved=False, improvement_pct=0.0,
                reliability_diagram=[], calibrated_model=model,
            )
            return empty

        # Raw probabilities
        raw_proba = model.predict_proba(X_cal)[:, 1]
        brier_before = float(brier_score_loss(y_cal, raw_proba))
        ece_before = self._ece(y_cal, raw_proba)

        # Calibrate
        try:
            calibrated = CalibratedClassifierCV(model, cv="prefit", method=self.method)
            calibrated.fit(X_cal, y_cal)
            cal_proba = calibrated.predict_proba(X_cal)[:, 1]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Calibration fitting failed: %s. Using raw model.", exc)
            calibrated = model
            cal_proba = raw_proba

        brier_after = float(brier_score_loss(y_cal, cal_proba))
        ece_after = self._ece(y_cal, cal_proba)

        improvement = ((ece_before - ece_after) / max(ece_before, 1e-10)) * 100
        improved = brier_after <= brier_before and ece_after < ece_before

        reliability = self._reliability_diagram(y_cal, cal_proba)

        logger.info(
            "Calibration (%s): Brier %.4f→%.4f | ECE %.4f→%.4f | Improved=%s",
            self.method, brier_before, brier_after, ece_before, ece_after, improved,
        )

        return CalibrationResult(
            method=self.method,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
            improved=improved,
            improvement_pct=improvement,
            reliability_diagram=reliability,
            calibrated_model=calibrated if improved else model,
        )

    def _ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Expected Calibration Error (ECE) — adaptive binning."""
        n = len(y_true)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            bin_confidence = float(y_prob[mask].mean())
            bin_accuracy = float(y_true[mask].mean())
            ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
        return float(ece)

    def _reliability_diagram(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> List[Dict[str, float]]:
        """Generate reliability diagram data (bin-level)."""
        bins = np.linspace(0, 1, self.n_bins + 1)
        result = []
        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            result.append({
                "bin_center": round(float((lo + hi) / 2), 3),
                "fraction_positives": round(float(y_true[mask].mean()), 4),
                "mean_predicted_probability": round(float(y_prob[mask].mean()), 4),
                "count": int(mask.sum()),
            })
        return result
