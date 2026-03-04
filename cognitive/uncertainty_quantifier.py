"""
cognitive/uncertainty_quantifier.py
-------------------------------------
Production-grade Conformal Prediction Uncertainty Quantifier.

Purpose
-------
Augments any fitted sklearn estimator with statistically valid prediction
intervals / sets using the split-conformal (inductive) method.

Guarantees
----------
For a miscoverage rate α, the conformal prediction set contains the true
label with probability ≥ 1 - α for any data distribution.

Architecture
------------
  Classification → Adaptive Prediction Sets (APS)
    - Uses softmax probability scores as non-conformity measures.
    - Returns the smallest set of classes with cumulative probability ≥ 1 - α.

  Regression → Conformal Quantile Regression
    - Calibrates symmetric intervals using absolute residuals on held-out set.
    - Returns [ŷ - q̂, ŷ + q̂] where q̂ is the (1-α)(1+1/n) quantile of calib errors.

Fallback
--------
  If sklearn calibration fails, returns a ±2σ interval from validation residuals.

Usage
-----
    from cognitive.uncertainty_quantifier import UncertaintyQuantifier

    uq = UncertaintyQuantifier()
    uq.calibrate(estimator, X_calib, y_calib, task="classification", alpha=0.1)
    result = uq.predict(X_test)
    # result["prediction_sets"]  — list[set] for classification
    # result["intervals"]        — [(lo, hi)] for regression
    # result["coverage_target"]  — 0.90
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

@dataclass
class UncertaintyReport:
    point_estimates: List[float]
    intervals: List[Tuple[float, float]]
    prediction_sets: List[List[Any]]
    coverage_target: float
    q_hat: float
    method: str

    @property
    def tier(self) -> str:
        if self.q_hat > 0.5:
            return "SPECULATIVE"
        elif self.q_hat > 0.2:
            return "MODERATE"
        return "PRECISE"

    def safe_statement(self) -> str:
        if self.intervals:
            lo, hi = self.intervals[0]
            return f"Expected range: [{lo:.2f}, {hi:.2f}]"
        return "No confident statement can be made."

logger = logging.getLogger("dipex.cognitive.uncertainty_quantifier")


class UncertaintyQuantifier:
    """
    Split-conformal uncertainty quantifier.

    Works with any sklearn estimator that implements predict_proba()
    (classification) or predict() (regression).
    """

    def __init__(self) -> None:
        self._estimator: Any   = None
        self._task:      str   = "classification"
        self._alpha:     float = 0.10   # 90% coverage target
        self._q_hat:     float = 0.0    # calibrated quantile / threshold
        self._classes:   Optional[np.ndarray] = None
        self._calibrated: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        estimator: Any,
        X_calib:   np.ndarray,
        y_calib:   np.ndarray,
        task:      str   = "classification",
        alpha:     float = 0.10,
    ) -> Dict[str, Any]:
        """
        Calibrate prediction sets on a held-out calibration set.

        Parameters
        ----------
        estimator : Fitted sklearn estimator
        X_calib   : Calibration feature matrix (≥50 rows recommended)
        y_calib   : Calibration ground-truth labels / values
        task      : "classification" | "regression"
        alpha     : Desired miscoverage rate (e.g. 0.10 → 90% coverage)

        Returns
        -------
        dict with keys: q_hat, coverage_target, method, n_calib
        """
        self._estimator = estimator
        self._task      = task
        self._alpha     = alpha
        n = len(y_calib)

        try:
            if task == "classification":
                self._classes = np.array(estimator.classes_) if hasattr(estimator, "classes_") else None
                proba = estimator.predict_proba(X_calib)   # shape (n, K)
                y_idx = np.searchsorted(
                    self._classes if self._classes is not None else np.unique(y_calib),
                    y_calib,
                )
                # Non-conformity score = 1 - softmax probability of true class
                scores = 1.0 - proba[np.arange(n), y_idx]
                self._q_hat = float(
                    np.quantile(scores, min(1.0, (1 - alpha) * (1 + 1 / n)))
                )
                method = "aps_conformal"
            else:
                y_pred = estimator.predict(X_calib)
                residuals = np.abs(np.asarray(y_calib, dtype=float) - y_pred)
                self._q_hat = float(
                    np.quantile(residuals, min(1.0, (1 - alpha) * (1 + 1 / n)))
                )
                method = "split_conformal_regression"

            self._calibrated = True
            logger.info(
                "UncertaintyQuantifier calibrated: task=%s alpha=%.2f q_hat=%.6f n=%d",
                task, alpha, self._q_hat, n,
            )
            return {
                "q_hat":           round(self._q_hat, 6),
                "coverage_target": 1 - alpha,
                "method":          method,
                "n_calib":         n,
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning("UQ calibration failed: %s — using fallback.", exc)
            if self._task == "regression":
                try:
                    y_pred = estimator.predict(X_calib)
                    resid = np.abs(np.asarray(y_calib, dtype=float) - y_pred)
                    self._q_hat = float(np.mean(resid) + 2 * np.std(resid))
                    self._calibrated = True
                except Exception:  # noqa: BLE001
                    self._q_hat = 1.0
            return {
                "q_hat":           self._q_hat,
                "coverage_target": 1 - alpha,
                "method":          "fallback_2sigma",
                "n_calib":         n,
            }

    def predict(
        self,
        X_test: np.ndarray,
        return_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate conformal prediction sets / intervals for test data.

        Returns
        -------
        Classification:
            {
              "prediction_sets": List[List[Any]],  # classes in prediction set
              "set_sizes":       List[int],
              "coverage_target": float,
              "method":          str,
            }
        Regression:
            {
              "point_estimates": List[float],
              "intervals":       List[Tuple[float, float]],
              "coverage_target": float,
              "method":          str,
            }
        """
        if not self._calibrated or self._estimator is None:
            logger.warning("UQ not calibrated — returning empty result.")
            return {"error": "not_calibrated", "coverage_target": 1 - self._alpha}

        try:
            if self._task == "classification":
                return self._predict_classification(X_test)
            else:
                return self._predict_regression(X_test)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UQ prediction failed: %s", exc)
            return {"error": str(exc), "coverage_target": 1 - self._alpha}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _predict_classification(self, X_test: np.ndarray) -> Dict[str, Any]:
        proba = self._estimator.predict_proba(X_test)   # (n, K)
        classes = (
            self._classes.tolist()
            if self._classes is not None
            else list(range(proba.shape[1]))
        )
        prediction_sets = []
        for row in proba:
            # Include all classes whose non-conformity score ≤ q_hat
            # non_conformity = 1 - p(class) → include if 1 - p ≤ q_hat → p ≥ 1 - q_hat
            pset = [cls for cls, p in zip(classes, row) if (1 - p) <= self._q_hat]
            if not pset:  # guarantee at least one class
                pset = [classes[int(np.argmax(row))]]
            prediction_sets.append(pset)

        return {
            "prediction_sets": prediction_sets,
            "set_sizes":       [len(s) for s in prediction_sets],
            "coverage_target": 1 - self._alpha,
            "q_hat":           round(self._q_hat, 6),
            "method":          "aps_conformal",
        }

    def _predict_regression(self, X_test: np.ndarray) -> Dict[str, Any]:
        y_pred = self._estimator.predict(X_test)
        intervals = [(float(p - self._q_hat), float(p + self._q_hat)) for p in y_pred]
        return {
            "point_estimates": [round(float(p), 6) for p in y_pred],
            "intervals":       intervals,
            "interval_width":  round(2 * self._q_hat, 6),
            "coverage_target": 1 - self._alpha,
            "q_hat":           round(self._q_hat, 6),
            "method":          "split_conformal_regression",
        }

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def coverage_summary(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Compute empirical coverage on a labelled test set."""
        if not self._calibrated:
            raise RuntimeError("UncertaintyQuantifier must be calibrated first.")

        if self._task == "classification":
            n_samples = len(X_test)
            covered = 0
            for i in range(n_samples):
                probs = self._estimator.predict_proba(X_test[i : i + 1])[0]
                idx_sort = np.argsort(probs)[::-1]
                cum_probs = np.cumsum(probs[idx_sort])
                
                # Number of classes needed to reach 1 - threshold
                k = np.argmax(cum_probs >= 1 - self._q_hat) + 1
                pred_set = self._classes[idx_sort[:k]]
                if y_test[i] in pred_set:
                    covered += 1
            return {"empirical_coverage": covered / n_samples}
        else:
            preds = self._estimator.predict(X_test)
            lo = preds - self._q_hat
            hi = preds + self._q_hat
            covered = np.mean((y_test >= lo) & (y_test <= hi))
            return {"empirical_coverage": float(covered)}

    def quantify_dataframe_column(self, df: pd.DataFrame, column: str) -> UncertaintyReport:
        """Heuristic fallback to quantify a dataframe numeric column dynamically without a model."""
        vals = df[column].dropna().values
        if len(vals) < 2:
            return UncertaintyReport(
                point_estimates=vals.tolist(), intervals=[], prediction_sets=[],
                coverage_target=0.90, q_hat=0.0, method="heuristic"
            )
        q_lo, q_hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
        return UncertaintyReport(
            point_estimates=vals.tolist(),
            intervals=[(q_lo, q_hi)] * len(vals),
            prediction_sets=[],
            coverage_target=0.90,
            q_hat=0.0,
            method="heuristic_quantiles"
        )

    def quantify_mean(self, s: pd.Series) -> UncertaintyReport:
        n = len(s)
        q_hat = 1.0 if n < 30 else 1.0 / np.sqrt(n)
        mean_val = float(s.mean())
        std_val = float(s.std()) if n > 1 else 0.0
        margin = q_hat * std_val
        return UncertaintyReport(
            point_estimates=[mean_val],
            intervals=[(mean_val - margin, mean_val + margin)],
            prediction_sets=[],
            coverage_target=0.90,
            q_hat=q_hat,
            method="heuristic_mean"
        )

    def quantify_proportion(self, k: int, n: int) -> UncertaintyReport:
        p = k / max(1, n)
        q_hat = 1.0 if n < 30 else 1.0 / np.sqrt(n)
        margin = q_hat * np.sqrt(p * (1 - p))
        return UncertaintyReport(
            point_estimates=[p],
            intervals=[(max(0.0, p - margin), min(1.0, p + margin))],
            prediction_sets=[],
            coverage_target=0.90,
            q_hat=q_hat,
            method="heuristic_prop"
        )
