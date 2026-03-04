"""
modeling/evaluator.py
----------------------
Comprehensive model evaluation engine.

Handles:
  - Classification: accuracy, F1, precision, recall, ROC-AUC, PR-AUC,
                    confusion matrix, classification report, calibration
  - Regression: RMSE, MAE, R², MAPE, explained variance
  - Cross-validation summary
  - Learning curve (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score,
)

logger = logging.getLogger("dipex.modeling.evaluator")


@dataclass
class EvalReport:
    model_name: str
    task: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]]
    classification_report: Optional[str]
    feature_importances: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "metrics": self.metrics,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.classification_report,
            "warnings": self.warnings,
        }

    def headline(self) -> str:
        if self.task == "classification":
            auc = self.metrics.get("roc_auc", "N/A")
            f1 = self.metrics.get("f1_weighted", "N/A")
            return f"[{self.model_name}] ROC-AUC={auc:.4f}  F1={f1:.4f}" if isinstance(auc, float) else f"[{self.model_name}]"
        else:
            r2 = self.metrics.get("r2", "N/A")
            rmse = self.metrics.get("rmse", "N/A")
            return f"[{self.model_name}] R²={r2:.4f}  RMSE={rmse:.4f}" if isinstance(r2, float) else f"[{self.model_name}]"


class ModelEvaluator:
    """
    Hold-out or training-set model evaluator.

    Usage::

        evaluator = ModelEvaluator()
        report = evaluator.evaluate(model, X_test, y_test, task="classification", model_name="RF")
        print(report.headline())
    """

    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        task: str,
        model_name: str = "",
        feature_names: Optional[List[str]] = None,
    ) -> EvalReport:
        """Evaluate a fitted model on X, y."""
        y_pred = model.predict(X)

        if task == "classification":
            return self._eval_classification(model, X, y, y_pred, model_name)
        else:
            return self._eval_regression(X, y, y_pred, model_name)

    def _eval_classification(
        self, model, X, y, y_pred, model_name: str
    ) -> EvalReport:
        metrics: Dict[str, float] = {
            "accuracy": round(float(accuracy_score(y, y_pred)), 6),
            "f1_weighted": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 6),
            "precision_weighted": round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 6),
            "recall_weighted": round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 6),
        }

        # ROC-AUC (binary or multi-class OvR)
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                n_classes = y_prob.shape[1]
                if n_classes == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y, y_prob[:, 1])), 6)
                    metrics["pr_auc"] = round(float(average_precision_score(y, y_prob[:, 1])), 6)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y, y_prob, multi_class="ovr", average="weighted")), 6)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ROC-AUC computation failed: %s", exc)

        cm = confusion_matrix(y, y_pred).tolist()
        cr = classification_report(y, y_pred, zero_division=0)

        return EvalReport(
            model_name=model_name, task="classification",
            metrics=metrics, confusion_matrix=cm, classification_report=cr,
        )

    def _eval_regression(self, X, y, y_pred, model_name: str) -> EvalReport:
        mse = mean_squared_error(y, y_pred)
        metrics = {
            "rmse": round(float(np.sqrt(mse)), 6),
            "mae": round(float(mean_absolute_error(y, y_pred)), 6),
            "r2": round(float(r2_score(y, y_pred)), 6),
            "explained_variance": round(float(explained_variance_score(y, y_pred)), 6),
        }
        # MAPE (skip zeros)
        y_arr = np.array(y, dtype=float)
        nonzero = y_arr != 0
        if nonzero.any():
            mape = float(np.mean(np.abs((y_arr[nonzero] - y_pred[nonzero]) / y_arr[nonzero])) * 100)
            metrics["mape_pct"] = round(mape, 4)

        return EvalReport(
            model_name=model_name, task="regression",
            metrics=metrics, confusion_matrix=None, classification_report=None,
        )

    def compare(self, reports: Dict[str, EvalReport]) -> pd.DataFrame:
        """Compare multiple EvalReports into a summary DataFrame."""
        rows = []
        for name, rep in reports.items():
            row = {"model": name, "task": rep.task}
            row.update(rep.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
