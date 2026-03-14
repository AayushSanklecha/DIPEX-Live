"""
qa_control/calibrator.py
--------------------------
Model confidence calibration — runs AFTER modeling, BEFORE confidence scoring.

Problem: sklearn/XGBoost/LightGBM probability estimates are often miscalibrated
(e.g., the model says "85% confident" but is only correct 60% of the time).
This directly degrades the confidence scores shown in the Reports page and
causes the QA gate to make wrong decisions.

Solution: Platt Scaling (sigmoid calibration) or Isotonic Regression,
applied cross-validated to avoid overfitting the calibration itself.

What this module does:
  1. Detect if calibration is beneficial (Brier score + ECE before vs after)
  2. Apply calibration if ECE improves by > min_improvement threshold
  3. Return calibrated probability estimates + calibration metadata
  4. Fall back gracefully if insufficient data or sklearn not available

All thresholds are config-driven.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.qa_control.calibrator")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationReport:
    run_id: str
    applied: bool                    # whether calibration was applied
    method: str                      # "platt" | "isotonic" | "none"
    brier_before: Optional[float] = None
    brier_after: Optional[float]  = None
    ece_before: Optional[float]   = None
    ece_after: Optional[float]    = None
    improvement: Optional[float]  = None   # ECE reduction
    warnings: List[str]           = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "calibration_applied": self.applied,
            "method": self.method,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "ece_improvement": self.improvement,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceCalibrator:
    """
    Post-model probability calibration.

    Typical usage — called from the pipeline bridge after modeling::

        cal = ConfidenceCalibrator.from_config(config)
        proba_cal, cal_report = cal.calibrate(model, X_train, y_train, X_test)
        # proba_cal: calibrated probability array
        # cal_report: CalibrationReport with ECE metrics

    Config stanza (all optional)::

        qa_control:
          calibration:
            method: "platt"          # "platt" | "isotonic" | "auto" (pick best)
            cv_folds: 5              # cross-validation folds for cal fitting
            min_improvement: 0.01   # minimum ECE reduction to apply calibration
            ece_bins: 10             # number of reliability-diagram bins
            min_samples: 50          # minimum samples needed for calibration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("qa_control", {}).get("calibration", {})
        self.method: str             = cfg.get("method", "auto")
        self.cv_folds: int           = int(cfg.get("cv_folds", 5))
        self.min_improvement: float  = float(cfg.get("min_improvement", 0.01))
        self.ece_bins: int           = int(cfg.get("ece_bins", 10))
        self.min_samples: int        = int(cfg.get("min_samples", 50))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfidenceCalibrator":
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        run_id: str = "",
    ) -> Tuple[Optional[np.ndarray], CalibrationReport]:
        """
        Fit calibration on (X_train, y_train) and optionally predict on X_test.

        Parameters
        ----------
        model   : fitted sklearn-compatible classifier (must have predict_proba)
        X_train : training features
        y_train : training labels
        X_test  : (optional) test/inference features to get calibrated probabilities for
        run_id  : pipeline run ID

        Returns
        -------
        (calibrated_proba, CalibrationReport)
        calibrated_proba is None if calibration was not applied.
        """
        report = CalibrationReport(run_id=run_id, applied=False, method="none")

        if not hasattr(model, "predict_proba"):
            report.warnings.append("Model has no predict_proba — calibration skipped.")
            return None, report

        if len(X_train) < self.min_samples:
            report.warnings.append(
                f"Only {len(X_train)} samples — need {self.min_samples} for calibration."
            )
            return None, report

        if y_train.nunique() != 2:
            report.warnings.append(
                "Calibration currently supports binary classification only."
            )
            return None, report

        try:
            from sklearn.calibration import CalibratedClassifierCV, calibration_curve
            from sklearn.metrics import brier_score_loss
        except ImportError:
            report.warnings.append("sklearn.calibration not available — skipping.")
            return None, report

        try:
            # Raw model probabilities (class 1)
            raw_proba = model.predict_proba(X_train)[:, 1]
            y_arr = np.array(y_train)

            brier_before = float(brier_score_loss(y_arr, raw_proba))
            ece_before   = self._expected_calibration_error(raw_proba, y_arr)
            report.brier_before = round(brier_before, 6)
            report.ece_before   = round(ece_before, 6)

            # Determine method
            methods = ["sigmoid", "isotonic"] if self.method == "auto" else [
                "sigmoid" if self.method == "platt" else self.method
            ]

            best_ece_after = ece_before
            best_method    = "none"
            best_model     = None

            for m in methods:
                if m == "isotonic" and len(X_train) < 1000:
                    # Isotonic needs enough data to avoid overfitting
                    report.warnings.append(
                        f"Isotonic skipped — needs >= 1000 samples (have {len(X_train)})."
                    )
                    continue
                try:
                    n_cv = min(self.cv_folds, int(y_arr.sum()), int((1 - y_arr).sum()))
                    if n_cv < 2:
                        continue
                    cal_model = CalibratedClassifierCV(
                        model, method=m, cv=n_cv
                    )
                    cal_model.fit(X_train, y_train)
                    cal_proba = cal_model.predict_proba(X_train)[:, 1]
                    ece_after = self._expected_calibration_error(cal_proba, y_arr)
                    if ece_after < best_ece_after:
                        best_ece_after = ece_after
                        best_method    = m
                        best_model     = cal_model
                except Exception as exc:
                    report.warnings.append(f"Calibration with {m} failed: {exc}")

            improvement = ece_before - best_ece_after
            report.improvement = round(improvement, 6)

            if improvement >= self.min_improvement and best_model is not None:
                # Calibration is beneficial — apply it
                report.applied = True
                report.method  = best_method
                brier_after    = float(brier_score_loss(
                    y_arr, best_model.predict_proba(X_train)[:, 1]
                ))
                report.brier_after = round(brier_after, 6)
                report.ece_after   = round(best_ece_after, 6)

                cal_proba_test: Optional[np.ndarray] = None
                if X_test is not None:
                    cal_proba_test = best_model.predict_proba(X_test)[:, 1]

                logger.info(
                    "[Calibrator] run_id=%s Applied %s calibration. "
                    "ECE: %.4f → %.4f (Δ=%.4f) | Brier: %.4f → %.4f",
                    run_id[:8], best_method,
                    ece_before, best_ece_after, improvement,
                    brier_before, brier_after,
                )
                return cal_proba_test, report
            else:
                logger.info(
                    "[Calibrator] run_id=%s Calibration not applied "
                    "(ECE improvement=%.4f < threshold %.4f).",
                    run_id[:8], improvement, self.min_improvement,
                )
                return None, report

        except Exception as exc:
            report.warnings.append(f"Calibration pipeline failed: {exc}")
            logger.warning("[Calibrator] Failed: %s", exc)
            return None, report

    # ── Calibration metrics ───────────────────────────────────────────────────

    def _expected_calibration_error(
        self, proba: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Expected Calibration Error (ECE) — average gap between confidence and accuracy
        across equal-frequency probability bins.
        """
        try:
            bin_edges = np.linspace(0.0, 1.0, self.ece_bins + 1)
            ece = 0.0
            n   = len(proba)
            for i in range(self.ece_bins):
                mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
                if mask.sum() == 0:
                    continue
                bin_conf = proba[mask].mean()
                bin_acc  = labels[mask].mean()
                ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
            return float(ece)
        except Exception:
            return float("nan")


# ── Standalone utility ────────────────────────────────────────────────────────

def compute_ece(proba: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standalone ECE computation for use outside the calibrator class."""
    return ConfidenceCalibrator()._expected_calibration_error(proba, labels)
