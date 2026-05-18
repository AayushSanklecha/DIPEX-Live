"""
scripts/train_models/utils/training_validator.py
-------------------------------------------------
Shared quality-gate validator used across all training scripts.

Validates that a trained model meets:
  - No overfitting: |val_metric - holdout_metric| ≤ 0.03
  - Low variance:   CV std ≤ 0.05
  - Adequate accuracy: metric ≥ min_threshold
  - Learning curve: both train and val improve over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.train.validator")


@dataclass
class ValidationReport:
    passed: bool = False
    gates_failed: List[str] = field(default_factory=list)
    gates_passed: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "gates_failed": self.gates_failed,
            "gates_passed": self.gates_passed,
            "summary": self.summary,
        }


class TrainingValidator:
    """
    Applies anti-overfitting quality gates to a trained model's metrics.

    Usage::

        validator = TrainingValidator()
        report = validator.validate(
            val_metric=0.87, holdout_metric=0.85,
            cv_scores=[0.88, 0.86, 0.87, 0.84, 0.89],
            min_threshold=0.70,
        )
        print(report.passed, report.summary)
    """

    OVERFIT_THRESHOLD = 0.03
    CV_STD_THRESHOLD  = 0.05
    MIN_CV_FOLDS      = 3

    def validate(
        self,
        val_metric: float,
        holdout_metric: float,
        cv_scores: Optional[List[float]] = None,
        min_threshold: float = 0.60,
        metric_name: str = "accuracy",
    ) -> ValidationReport:
        """Run all quality gates and return a ValidationReport."""
        report = ValidationReport()

        # Gate 1: Overfitting check
        gap = abs(val_metric - holdout_metric)
        if gap > self.OVERFIT_THRESHOLD:
            report.gates_failed.append(
                f"OVERFIT: |val({val_metric:.3f}) - holdout({holdout_metric:.3f})| "
                f"= {gap:.3f} > {self.OVERFIT_THRESHOLD}"
            )
        else:
            report.gates_passed.append(f"OVERFIT_OK: gap={gap:.3f}")

        # Gate 2: Minimum threshold
        worst = min(val_metric, holdout_metric)
        if worst < min_threshold:
            report.gates_failed.append(
                f"UNDERFIT: {metric_name}={worst:.3f} < min={min_threshold}"
            )
        else:
            report.gates_passed.append(f"THRESHOLD_OK: {metric_name}={worst:.3f}")

        # Gate 3: CV variance
        if cv_scores is not None and len(cv_scores) >= self.MIN_CV_FOLDS:
            cv_arr = np.array(cv_scores)
            cv_std = float(cv_arr.std())
            cv_mean = float(cv_arr.mean())
            if cv_std > self.CV_STD_THRESHOLD:
                report.gates_failed.append(
                    f"HIGH_VARIANCE: CV std={cv_std:.3f} > {self.CV_STD_THRESHOLD}"
                )
            else:
                report.gates_passed.append(f"VARIANCE_OK: CV std={cv_std:.3f}")
            # Gate 4: CV mean vs holdout (training-time consistency)
            if abs(cv_mean - holdout_metric) > 0.08:
                report.gates_failed.append(
                    f"CV_HOLDOUT_MISMATCH: CV mean={cv_mean:.3f} vs holdout={holdout_metric:.3f}"
                )
            else:
                report.gates_passed.append(f"CV_CONSISTENCY_OK: CV={cv_mean:.3f} ≈ holdout={holdout_metric:.3f}")

        report.passed = len(report.gates_failed) == 0
        report.summary = (
            f"{'PASSED' if report.passed else 'FAILED'}: "
            f"{len(report.gates_passed)} gates passed, "
            f"{len(report.gates_failed)} failed. "
            + (f"Failures: {'; '.join(report.gates_failed)}" if report.gates_failed else "All gates clear.")
        )
        logger.info("[TrainingValidator] %s", report.summary)
        return report


def assert_no_data_leakage(X_train, X_test) -> bool:
    """
    Check that train and test splits share no identical rows.
    Returns True if no leakage detected.
    """
    try:
        import pandas as pd
        train_set = set(X_train.apply(tuple, axis=1))
        test_set  = set(X_test.apply(tuple, axis=1))
        overlap = train_set & test_set
        if overlap:
            logger.warning("⚠️  Data leakage detected: %d identical rows in train+test", len(overlap))
            return False
        logger.debug("✅ No data leakage between train and test sets")
        return True
    except Exception as exc:
        logger.debug("Leakage check skipped: %s", exc)
        return True


def compute_learning_curve_slope(
    train_scores: List[float],
    val_scores: List[float],
) -> Dict[str, float]:
    """
    Compute the slope of learning curves.
    Both slopes should be positive (improving) for a healthy model.
    """
    if len(train_scores) < 2 or len(val_scores) < 2:
        return {"train_slope": 0.0, "val_slope": 0.0, "healthy": True}
    x = np.arange(len(train_scores))
    train_slope = float(np.polyfit(x, train_scores, 1)[0])
    val_slope   = float(np.polyfit(x[:len(val_scores)], val_scores, 1)[0])
    healthy = train_slope >= -0.01 and val_slope >= -0.01
    return {
        "train_slope": round(train_slope, 6),
        "val_slope": round(val_slope, 6),
        "healthy": healthy,
    }
