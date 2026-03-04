"""
modeling/explainability.py
---------------------------
Model explainability engine using SHAP.

Features:
  - SHAP TreeExplainer for tree-based models
  - SHAP LinearExplainer for linear models
  - SHAP KernelExplainer fallback for any model
  - Feature importance summary (mean |SHAP|)
  - Per-prediction explanations
  - SHAP value export for dashboard consumption
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.modeling.explainability")


@dataclass
class ExplainabilityReport:
    model_name: str
    method: str          # 'tree', 'linear', 'kernel'
    feature_names: List[str]
    mean_abs_shap: Dict[str, float]     # {feature: mean |SHAP|}
    shap_values: Optional[Any]           # raw SHAP array (not serializable)
    top_features: List[Dict[str, float]] # [{feature, shap_mean}] sorted desc
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "method": self.method,
            "feature_names": self.feature_names,
            "mean_abs_shap": self.mean_abs_shap,
            "top_features": self.top_features,
            "warnings": self.warnings,
        }


class ModelExplainer:
    """
    SHAP-based model explainability.

    Usage::

        explainer = ModelExplainer()
        report = explainer.explain(model, X_df, model_name="RF", task="classification")
        print(report.top_features[:5])
    """

    TOP_N = 20

    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        model_name: str = "",
        task: str = "classification",
        n_samples: int = 200,
    ) -> ExplainabilityReport:
        """Compute SHAP values and return ExplainabilityReport."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed — explainability skipped.")
            return ExplainabilityReport(
                model_name=model_name, method="unavailable",
                feature_names=list(X.columns), mean_abs_shap={},
                shap_values=None, top_features=[],
                warnings=["shap not installed — run: pip install shap"],
            )

        # Sample for speed if large
        X_sample = X.sample(min(n_samples, len(X)), random_state=42) if len(X) > n_samples else X
        feature_names = list(X.columns)

        # Try Tree → Linear → Kernel
        method = "kernel"
        shap_values = None

        try:
            exp = shap.TreeExplainer(model)
            shap_values = exp.shap_values(X_sample)
            method = "tree"
        except Exception:  # noqa: BLE001
            try:
                exp = shap.LinearExplainer(model, X_sample)
                shap_values = exp.shap_values(X_sample)
                method = "linear"
            except Exception:  # noqa: BLE001
                try:
                    background = shap.sample(X_sample, min(50, len(X_sample)))
                    exp = shap.KernelExplainer(model.predict, background)
                    shap_values = exp.shap_values(X_sample.iloc[:min(30, len(X_sample))], silent=True)
                    method = "kernel"
                except Exception as exc:  # noqa: BLE001
                    return ExplainabilityReport(
                        model_name=model_name, method="failed",
                        feature_names=feature_names, mean_abs_shap={},
                        shap_values=None, top_features=[],
                        warnings=[f"SHAP explainability failed: {exc}"],
                    )

        # Aggregate: if multi-class, shap_values is a list → use mean across classes
        if isinstance(shap_values, list):
            sv_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            sv_array = np.abs(shap_values)

        mean_abs = dict(zip(feature_names, sv_array.mean(axis=0).tolist()))
        top = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)[:self.TOP_N]

        return ExplainabilityReport(
            model_name=model_name,
            method=method,
            feature_names=feature_names,
            mean_abs_shap={k: round(v, 8) for k, v in mean_abs.items()},
            shap_values=shap_values,
            top_features=[{"feature": k, "shap_mean": round(v, 8)} for k, v in top],
        )

    def explain_prediction(
        self, model: Any, X_row: pd.DataFrame
    ) -> Dict[str, Any]:
        """SHAP explanation for a single prediction."""
        try:
            import shap
            exp = shap.TreeExplainer(model)
            sv = exp.shap_values(X_row)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            features = list(X_row.columns)
            return {
                "prediction": model.predict(X_row)[0],
                "shap_per_feature": {
                    f: round(float(v), 8) for f, v in zip(features, sv[0])
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
