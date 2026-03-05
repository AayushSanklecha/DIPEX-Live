"""
proposal/automl.py
------------------
True multi-model AutoML — races 4 candidates and returns the best.

Models tried (classification): LogisticRegression, RandomForest, XGBoost, LightGBM
Models tried (regression)    : Ridge, RandomForest, XGBoost, LightGBM

Selection:
  - 3-fold cross-validation per model
  - Classification: ROC-AUC (primary) + F1 (secondary)
  - Regression    : R² (primary) + RMSE (secondary)
  - Winner = highest CV score
  - Any model that fails to import or train is silently skipped
  - Falls back to single RandomForest if all others fail

Output includes `all_results` dict so the report can show a comparison table.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_MIN_SAMPLES: int = 20   # minimum rows for a meaningful CV


# ── SHAP feature importance helper ────────────────────────────────────

def _compute_shap_importances(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int = 500,
) -> Dict[str, Any]:
    """
    Fit the winning model on all data and compute SHAP feature importances.

    Returns
    -------
    {
      "shap_top_features": [{"feature": str, "importance": float}, ...],  # top 10
      "shap_method": "tree" | "linear" | "sklearn_fi" | "none"
    }
    """
    try:
        # Sample for speed on large datasets
        if len(X) > max_rows:
            idx = np.random.default_rng(42).choice(len(X), max_rows, replace=False)
            X_s = X.iloc[idx].reset_index(drop=True)
            y_s = y.iloc[idx].reset_index(drop=True)
        else:
            X_s, y_s = X, y

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_s, y_s)

        try:
            import shap  # type: ignore

            # Tree-based models
            if model_name in ("RandomForest", "XGBoost", "LightGBM"):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_s)
                # For classifiers, shap_values may be list[array] — take class 1
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                mean_abs = np.abs(shap_vals).mean(axis=0)
                method = "tree"

            # Linear models
            elif model_name in ("LogisticRegression", "Ridge"):
                explainer = shap.LinearExplainer(model, X_s)
                shap_vals = explainer.shap_values(X_s)
                mean_abs = np.abs(shap_vals).mean(axis=0)
                method = "linear"

            else:
                raise ValueError("Unknown model type for SHAP")

            ranked = sorted(
                zip(X.columns.tolist(), mean_abs.tolist()),
                key=lambda t: t[1], reverse=True
            )[:10]

            return {
                "shap_top_features": [
                    {"feature": f, "importance": round(v, 4)}
                    for f, v in ranked
                ],
                "shap_method": method,
            }

        except ImportError:
            pass  # fall through to sklearn fallback

        # Fallback: sklearn feature_importances_ (tree models only)
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            ranked = sorted(
                zip(X.columns.tolist(), fi.tolist()),
                key=lambda t: t[1], reverse=True
            )[:10]
            return {
                "shap_top_features": [
                    {"feature": f, "importance": round(v, 4)}
                    for f, v in ranked
                ],
                "shap_method": "sklearn_fi",
            }

    except Exception as exc:
        logger.warning("SHAP/feature importance failed for %s: %s", model_name, exc)

    return {"shap_top_features": [], "shap_method": "none"}


# ── Lazy model loaders (skip silently if package not installed) ───────────────

def _classifiers() -> List[Tuple[str, Any]]:
    models: List[Tuple[str, Any]] = []

    try:
        from sklearn.linear_model import LogisticRegression
        models.append(("LogisticRegression",
                        LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)))
    except Exception:
        pass

    try:
        from sklearn.ensemble import RandomForestClassifier
        models.append(("RandomForest",
                        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier
        models.append(("XGBoost",
                        XGBClassifier(n_estimators=100, random_state=42,
                                      use_label_encoder=False,
                                      eval_metric="logloss", verbosity=0, n_jobs=-1)))
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models.append(("LightGBM",
                        LGBMClassifier(n_estimators=100, random_state=42,
                                       verbose=-1, n_jobs=-1)))
    except Exception:
        pass

    return models


def _regressors() -> List[Tuple[str, Any]]:
    models: List[Tuple[str, Any]] = []

    try:
        from sklearn.linear_model import Ridge
        models.append(("Ridge", Ridge()))
    except Exception:
        pass

    try:
        from sklearn.ensemble import RandomForestRegressor
        models.append(("RandomForest",
                        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
    except Exception:
        pass

    try:
        from xgboost import XGBRegressor
        models.append(("XGBoost",
                        XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)))
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor
        models.append(("LightGBM",
                        LGBMRegressor(n_estimators=100, random_state=42,
                                      verbose=-1, n_jobs=-1)))
    except Exception:
        pass

    return models


# ── AutoML class ──────────────────────────────────────────────────────────────

class AutoMLProposal:
    """
    Multi-model AutoML — races candidates and returns the best performer.

    Usage::

        result = AutoMLProposal().propose(df, target_col="label")
        # result["model_type"]  → winning model name
        # result["metric_value"] → best CV score
        # result["all_results"] → {model: score, ...} comparison table
    """

    def propose(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Run a 3-fold CV race across all available models.
        Returns the best model's results with a full comparison table.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # ── Prepare features ──────────────────────────────────────────────────
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col].copy()

        if X.empty or len(X.columns) == 0:
            raise ValueError("No numeric feature columns available for AutoML.")

        if len(df) < _MIN_SAMPLES:
            raise ValueError(
                f"Dataset has only {len(df)} rows — minimum {_MIN_SAMPLES} required for CV."
            )

        # ── Detect task ───────────────────────────────────────────────────────
        is_classification = y.nunique() < 20 or pd.api.types.is_object_dtype(y)
        task = "classification" if is_classification else "regression"

        # Encode string labels
        if is_classification and pd.api.types.is_object_dtype(y):
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y), index=y.index)
            # Binary-only for AUC — multi-class falls back to accuracy
            if y.nunique() > 2:
                is_classification = True   # still classification
                primary_metric = "accuracy"
            else:
                primary_metric = "roc_auc"
        elif is_classification:
            primary_metric = "roc_auc" if y.nunique() == 2 else "accuracy"
        else:
            primary_metric = "r2"

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if is_classification \
             else KFold(n_splits=3, shuffle=True, random_state=42)

        candidates = _classifiers() if is_classification else _regressors()
        if not candidates:
            # Absolute last resort
            return self._fallback(X, y, task, list(X.columns))

        all_results: Dict[str, float] = {}
        best_name  = None
        best_score = -999.0

        for name, model in candidates:
            try:
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(model, X, y, cv=cv,
                                             scoring=primary_metric, n_jobs=1)
                elapsed = time.perf_counter() - t0
                cv_mean = float(np.clip(scores.mean(), 0.0, 1.0)) if is_classification \
                         else float(scores.mean())
                all_results[name] = round(cv_mean, 4)

                logger.info(
                    "AutoML [%s] task=%s %s=%.4f ± %.4f  (%.2fs)",
                    name, task, primary_metric, cv_mean, scores.std(), elapsed,
                )

                if cv_mean > best_score:
                    best_score = cv_mean
                    best_name  = name

            except Exception as exc:
                logger.warning("AutoML [%s] failed: %s", name, exc)
                all_results[name] = None  # type: ignore[assignment]

        if best_name is None:
            return self._fallback(X, y, task, list(X.columns))

        # Filter out invalid/None scores from comparison table
        all_results = {k: v for k, v in all_results.items() if v is not None}

        # ── Compute secondary metric on the winner ────────────────────────────
        secondary: Dict[str, Any] = {}
        try:
            if is_classification and primary_metric == "roc_auc":
                f1_scores = cross_val_score(
                    dict(candidates)[best_name], X, y,
                    cv=cv, scoring="f1", n_jobs=1
                )
                secondary["f1_score"] = round(float(f1_scores.mean()), 4)
            elif not is_classification:
                rmse_scores = cross_val_score(
                    dict(candidates)[best_name], X, y,
                    cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
                )
                secondary["rmse"] = round(float(-rmse_scores.mean()), 4)
        except Exception:
            pass

        logger.info(
            "AutoML winner: %s  %s=%.4f  features=%d",
            best_name, primary_metric, best_score, len(X.columns),
        )

        # ── SHAP feature importances on the winning model ─────────────────────
        shap_info: Dict[str, Any] = {"shap_top_features": [], "shap_method": "none"}
        try:
            winner_model = dict(candidates)[best_name]
            shap_info = _compute_shap_importances(
                best_name, winner_model, X, y, max_rows=500
            )
            if shap_info["shap_top_features"]:
                top3 = ", ".join(f['feature'] for f in shap_info["shap_top_features"][:3])
                logger.info("[SHAP] Top features for %s: %s", best_name, top3)
        except Exception as exc:
            logger.warning("[SHAP] automl shap failed: %s", exc)

        return {
            "model_type":         best_name,
            "task":               task,
            "metric_name":        primary_metric,
            "metric_value":       round(best_score, 4),
            "cv_folds":           3,
            "features":           list(X.columns),
            "features_used":      len(X.columns),
            "all_results":        {k: v for k, v in all_results.items() if v is not None},
            "shap_top_features":  shap_info["shap_top_features"],
            "shap_method":        shap_info["shap_method"],
            "status":             "PROPOSED",
            **secondary,
        }

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback(self, X: pd.DataFrame, y: pd.Series,
                  task: str, features: List[str]) -> Dict[str, Any]:
        """Single RandomForest fallback — used only if all CV candidates fail."""
        logger.warning("AutoML: all candidates failed — using RandomForest fallback")
        try:
            if task == "classification":
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                m = RandomForestClassifier(n_estimators=50, random_state=42)
                m.fit(X_tr, y_tr)
                score = float(accuracy_score(y_te, m.predict(X_te)))
                return {
                    "model_type": "RandomForest",
                    "task": task, "metric_name": "accuracy",
                    "metric_value": round(score, 4), "cv_folds": 0,
                    "features": features, "features_used": len(features),
                    "all_results": {"RandomForest": round(score, 4)},
                    "status": "PROPOSED_FALLBACK",
                }
            else:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                m = RandomForestRegressor(n_estimators=50, random_state=42)
                m.fit(X_tr, y_tr)
                score = float(r2_score(y_te, m.predict(X_te)))
                return {
                    "model_type": "RandomForest",
                    "task": task, "metric_name": "r2",
                    "metric_value": round(score, 4), "cv_folds": 0,
                    "features": features, "features_used": len(features),
                    "all_results": {"RandomForest": round(score, 4)},
                    "status": "PROPOSED_FALLBACK",
                }
        except Exception as exc:
            return {
                "model_type": "None", "task": task,
                "metric_name": "N/A", "metric_value": 0.0,
                "features": features, "features_used": len(features),
                "all_results": {}, "status": "ERROR",
                "error": str(exc),
            }
