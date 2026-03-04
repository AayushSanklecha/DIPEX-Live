"""
explanation/shap_explainer.py
------------------------------
Production-grade SHAP explainability engine for DIPEX models.

Purpose
-------
Generate feature-level explanations for any fitted sklearn estimator using
SHAP (SHapley Additive exPlanations). Provides per-prediction and global
summary explanations for transparency and regulatory compliance.

Architecture
------------
• Tries shap.TreeExplainer  → fastest, native for forest/boosting models.
• Falls back to shap.LinearExplainer → for linear models.
• Falls back to shap.KernelExplainer (100-row sample) → universal fallback.
• If SHAP is not installed at all → uses permutation importance as proxy.

Output
------
{
  "feature_names":  List[str],
  "mean_abs_shap":  List[float],   # global importance ranking
  "shap_values":    List[float],   # per-instance values for first prediction
  "top_features":   List[dict],    # [{feature, shap_value}, ...] top-20
  "method":         str,
}

Usage
-----
    from explanation.shap_explainer import SHAPExplainer

    explainer = SHAPExplainer()
    result = explainer.explain(model=fitted_model,
                               X=X_df,
                               feature_names=list(X_df.columns),
                               task="classification")
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.explanation.shap")


class SHAPExplainer:
    """
    SHAP-based model explainer with multi-tier fallback strategy.
    Designed to work with any sklearn-compatible estimator.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        model:         Any,
        X:             Any,        # pd.DataFrame or np.ndarray
        feature_names: Optional[List[str]] = None,
        task:          str = "classification",
        max_background: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a fitted model.

        Parameters
        ----------
        model         : Fitted sklearn estimator
        X             : Input features (DataFrame or array, n × p)
        feature_names : Column names; inferred from X if it's a DataFrame
        task          : "classification" | "regression"
        max_background: Max rows used for KernelExplainer background sample

        Returns
        -------
        See module docstring for output schema.
        """
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.array(X)
            feature_names = feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

        # ── Try SHAP ─────────────────────────────────────────────────────────
        try:
            import shap
            return self._shap_explain(model, X_arr, feature_names, task, max_background, shap)
        except ImportError:
            logger.warning("SHAPExplainer: shap not installed — using permutation importance fallback.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAPExplainer: shap failed (%s) — using permutation importance.", exc)

        # ── Fallback: sklearn permutation importance ──────────────────────────
        return self._permutation_explain(model, X_arr, feature_names, task)

    # ── SHAP tier ─────────────────────────────────────────────────────────────

    def _shap_explain(
        self, model, X_arr, feature_names, task, max_background, shap
    ) -> Dict[str, Any]:
        method = "unknown"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Tier 1: TreeExplainer (forests, boosting)
            if hasattr(model, "estimators_") or hasattr(model, "tree_"):
                try:
                    ex = shap.TreeExplainer(model)
                    values = ex.shap_values(X_arr)
                    method = "tree_explainer"
                except Exception:  # noqa: BLE001
                    values = None
            else:
                values = None

            # Tier 2: LinearExplainer
            if values is None and hasattr(model, "coef_"):
                try:
                    bg = shap.maskers.Independent(X_arr[:min(len(X_arr), max_background)])
                    ex = shap.LinearExplainer(model, masker=bg)
                    values = ex.shap_values(X_arr)
                    method = "linear_explainer"
                except Exception:  # noqa: BLE001
                    values = None

            # Tier 3: KernelExplainer (universal fallback)
            if values is None:
                bg = X_arr[:min(len(X_arr), max_background)]
                pred_fn = model.predict_proba if (task == "classification" and hasattr(model, "predict_proba")) \
                          else model.predict
                ex = shap.KernelExplainer(pred_fn, bg)
                sample  = X_arr[:min(50, len(X_arr))]
                values  = ex.shap_values(sample)
                X_arr   = sample
                method  = "kernel_explainer"

        # Handle multi-class (take class-1 shap values)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]

        values = np.array(values)
        mean_abs = np.abs(values).mean(axis=0) if values.ndim == 2 else np.abs(values)

        return self._format_result(mean_abs, values, feature_names, method)

    # ── Permutation fallback ──────────────────────────────────────────────────

    def _permutation_explain(self, model, X_arr, feature_names, task) -> Dict[str, Any]:
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import accuracy_score, r2_score
            metric  = accuracy_score if task == "classification" else r2_score
            scoring = "accuracy" if task == "classification" else "r2"
            perm    = permutation_importance(
                model, X_arr, model.predict(X_arr),
                n_repeats=5, random_state=42, scoring=scoring,
            )
            mean_abs = perm.importances_mean
            return self._format_result(mean_abs, mean_abs[np.newaxis, :],
                                       feature_names, "permutation_importance")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAPExplainer: permutation fallback failed: %s", exc)
            # Last resort: return zeros
            n = len(feature_names)
            mean_abs = np.zeros(n)
            return self._format_result(mean_abs, mean_abs[np.newaxis, :],
                                       feature_names, "no_explainability")

    # ── Format result ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_result(
        mean_abs, values, feature_names, method
    ) -> Dict[str, Any]:
        n = len(feature_names)
        mean_abs = np.array(mean_abs).flatten()[:n]
        sorted_idx = np.argsort(mean_abs)[::-1]
        top_features = [
            {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in sorted_idx[:20]
        ]
        return {
            "feature_names":  feature_names,
            "mean_abs_shap":  [round(float(v), 6) for v in mean_abs],
            "shap_values":    [round(float(v), 6) for v in (values[0] if values.ndim == 2 else values)[:n]],
            "top_features":   top_features,
            "method":         method,
        }
