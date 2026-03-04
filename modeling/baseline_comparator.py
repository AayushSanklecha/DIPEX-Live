"""
modeling/baseline_comparator.py
---------------------------------
Enforce that trained models meaningfully outperform a naive baseline.

Rulebook (industry standard):
  - Classification: model AUC > DummyClassifier(strategy='most_frequent') AUC
                    model F1  > baseline F1 by at least `min_lift`
  - Regression:     model R²  > 0.0 (DummyRegressor(strategy='mean') gets R²=0)
                    model RMSE < baseline RMSE by at least `min_lift_pct` percent

If model fails to beat baseline → raises `BaselineNotBeatenError` (Hard Gate).
Can also run in WARN mode (warning only, doesn't block).

Usage::

    bc = BaselineComparator(min_lift=0.02, block_on_failure=True)
    result = bc.compare(model, X_train, y_train, X_test, y_test, task="classification")
    if not result.beats_baseline:
        raise result.gate_error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)

logger = logging.getLogger("dipex.modeling.baseline")


class BaselineNotBeatenError(ValueError):
    """Raised when model does not statistically outperform the dummy baseline."""


@dataclass
class BaselineComparisonResult:
    task: str
    model_name: str
    beats_baseline: bool
    block_reason: Optional[str]

    # Baseline metrics
    baseline_metric_name: str
    baseline_metric_value: float

    # Model metrics
    model_metric_value: float
    lift: float
    lift_pct: float

    details: Dict[str, Any] = field(default_factory=dict)
    gate_error: Optional[BaselineNotBeatenError] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model_name": self.model_name,
            "beats_baseline": self.beats_baseline,
            "block_reason": self.block_reason,
            "baseline_metric_name": self.baseline_metric_name,
            "baseline_metric_value": round(self.baseline_metric_value, 6),
            "model_metric_value": round(self.model_metric_value, 6),
            "lift": round(self.lift, 6),
            "lift_pct": round(self.lift_pct, 2),
            "verdict": "PASS" if self.beats_baseline else "FAIL",
            "interpretation": self._interpret(),
            "details": self.details,
        }

    def _interpret(self) -> str:
        if self.beats_baseline:
            return (f"Model '{self.model_name}' beats baseline on {self.baseline_metric_name}: "
                    f"{self.model_metric_value:.4f} vs {self.baseline_metric_value:.4f} "
                    f"(+{self.lift_pct:.1f}% lift). Training approved.")
        return (f"Model '{self.model_name}' FAILS to beat baseline on {self.baseline_metric_name}: "
                f"{self.model_metric_value:.4f} vs {self.baseline_metric_value:.4f}. "
                f"Model may have found spurious patterns or data is not predictive.")


class BaselineComparator:
    """
    Enforce model vs. naive baseline comparison.

    Parameters
    ----------
    min_lift : float
        Minimum absolute improvement over baseline metric required.
    block_on_failure : bool
        If True, raises BaselineNotBeatenError when model fails. If False, warns only.
    """

    def __init__(self, min_lift: float = 0.02, block_on_failure: bool = True) -> None:
        self.min_lift = min_lift
        self.block_on_failure = block_on_failure

    def compare(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task: str = "classification",
        model_name: str = "model",
    ) -> BaselineComparisonResult:
        """Compare model against a dummy baseline on test data."""
        if task == "classification":
            return self._compare_classification(model, X_train, y_train, X_test, y_test, model_name)
        else:
            return self._compare_regression(model, X_train, y_train, X_test, y_test, model_name)

    # ── Classification ────────────────────────────────────────────────────────

    def _compare_classification(self, model, X_train, y_train, X_test, y_test, model_name):
        # Fit dummy
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)

        # Baseline metrics
        classes = np.unique(y_train)
        is_binary = len(classes) == 2
        avg = "binary" if is_binary else "macro"

        base_f1 = float(f1_score(y_test, y_dummy, average=avg, zero_division=0))
        base_acc = float(accuracy_score(y_test, y_dummy))
        base_auc: Optional[float] = None
        try:
            if hasattr(dummy, "predict_proba"):
                dummy_proba = dummy.predict_proba(X_test)
                base_auc = float(roc_auc_score(y_test, dummy_proba[:, 1] if is_binary else dummy_proba, multi_class="ovr"))
        except Exception:  # noqa: BLE001
            pass

        # Model metrics
        y_pred = model.predict(X_test)
        model_f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
        model_acc = float(accuracy_score(y_test, y_pred))
        model_auc: Optional[float] = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                model_auc = float(roc_auc_score(y_test, proba[:, 1] if is_binary else proba, multi_class="ovr"))
        except Exception:  # noqa: BLE001
            pass

        # Primary metric: AUC if available, else F1
        if model_auc is not None and base_auc is not None:
            metric_name, base_val, model_val = "AUC", base_auc, model_auc
        else:
            metric_name, base_val, model_val = "F1", base_f1, model_f1

        lift = model_val - base_val
        lift_pct = (lift / max(base_val, 1e-6)) * 100
        beats = lift >= self.min_lift

        result = BaselineComparisonResult(
            task="classification", model_name=model_name, beats_baseline=beats,
            block_reason=None if beats else f"Lift={lift:.4f} < minimum required {self.min_lift}",
            baseline_metric_name=metric_name, baseline_metric_value=base_val,
            model_metric_value=model_val, lift=lift, lift_pct=lift_pct,
            details={
                "baseline_f1": round(base_f1, 4), "model_f1": round(model_f1, 4),
                "baseline_accuracy": round(base_acc, 4), "model_accuracy": round(model_acc, 4),
                "baseline_auc": round(base_auc, 4) if base_auc else None,
                "model_auc": round(model_auc, 4) if model_auc else None,
            },
        )
        self._handle_result(result)
        return result

    # ── Regression ────────────────────────────────────────────────────────────

    def _compare_regression(self, model, X_train, y_train, X_test, y_test, model_name):
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)

        base_r2   = float(r2_score(y_test, y_dummy))
        base_rmse = float(np.sqrt(mean_squared_error(y_test, y_dummy)))

        y_pred   = model.predict(X_test)
        model_r2 = float(r2_score(y_test, y_pred))
        model_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # For regression, lift = RMSE reduction (lower is better)
        lift      = base_rmse - model_rmse
        lift_pct  = (lift / max(base_rmse, 1e-6)) * 100
        beats     = model_r2 > 0 and lift >= 0

        result = BaselineComparisonResult(
            task="regression", model_name=model_name, beats_baseline=beats,
            block_reason=None if beats else (
                f"Model R²={model_r2:.4f} ≤ 0 (no better than mean prediction)" if model_r2 <= 0
                else f"RMSE lift {lift:.4f} insufficient"
            ),
            baseline_metric_name="RMSE", baseline_metric_value=base_rmse,
            model_metric_value=model_rmse, lift=lift, lift_pct=lift_pct,
            details={
                "baseline_r2": round(base_r2, 4), "model_r2": round(model_r2, 4),
                "baseline_rmse": round(base_rmse, 4), "model_rmse": round(model_rmse, 4),
                "model_mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            },
        )
        self._handle_result(result)
        return result

    def _handle_result(self, result: BaselineComparisonResult) -> None:
        if not result.beats_baseline:
            logger.error("Baseline gate FAILED for '%s': %s", result.model_name, result.block_reason)
            if self.block_on_failure:
                result.gate_error = BaselineNotBeatenError(result.block_reason)
        else:
            logger.info("Baseline gate PASSED for '%s': +%.2f%% lift",
                        result.model_name, result.lift_pct)
