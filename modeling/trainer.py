"""
modeling/trainer.py
--------------------
Enterprise ML model training engine.

Supports:
  - RandomForest (classification / regression)
  - XGBoost (classification / regression)
  - LightGBM (classification / regression)
  - LogisticRegression / Ridge (sklearn)
  - GradientBoostingClassifier / GradientBoostingRegressor (sklearn)

Features:
  - Auto task-type detection (classification / regression)
  - Stratified K-Fold cross-validation
  - Optuna hyperparameter search (optional)
  - Returns TrainResult: model artifact, metrics, feature importances
  - Full sklearn pipeline integration with preprocessing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
)

logger = logging.getLogger("dipex.modeling.trainer")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    run_id: str
    model_name: str
    task: str                         # 'classification' | 'regression'
    cv_metrics: Dict[str, float]
    feature_importances: List[Dict[str, Any]]
    best_params: Dict[str, Any]
    training_time_s: float
    model: Any = field(repr=False)    # fitted sklearn estimator
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "task": self.task,
            "cv_metrics": self.cv_metrics,
            "feature_importances": self.feature_importances[:20],  # top 20
            "best_params": self.best_params,
            "training_time_s": round(self.training_time_s, 2),
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Multi-algorithm cross-validated model trainer.

    Usage::

        trainer = ModelTrainer(config)
        result = trainer.train(df, target="churn", run_id="run-001")
        print(result.cv_metrics)
    """

    DEFAULT_ALGORITHMS = ["random_forest", "logistic", "mlp"]  # [ML] MLP added

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("modeling", {})
        self.algorithms: List[str] = cfg.get("algorithms", self.DEFAULT_ALGORITHMS)
        self.cv_folds: int = int(cfg.get("cv_folds", 5))
        self.use_optuna: bool = bool(cfg.get("hyperparameter_search", "none") == "optuna")
        self.n_trials: int = int(cfg.get("n_trials", 20))
        self.random_state: int = 42

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelTrainer":
        return cls(config)

    def train(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        run_id: str = "",
        time_col: Optional[str] = None,
        run_guards: bool = True,
    ) -> Dict[str, "TrainResult"]:
        """
        Train all configured algorithms and return a dict of TrainResult.

        Parameters
        ----------
        time_col   : if provided, enables temporal split validation (no cross-time leakage)
        run_guards : if True, runs leakage detection + baseline comparator (config-driven)
        """
        feats = features or [c for c in df.columns if c != target]
        sub = df[[target] + feats].dropna()

        if sub.empty:
            raise ValueError(f"No rows remain after dropping NaN for target='{target}'")

        X_df = sub[feats]
        X = X_df.values
        y = sub[target].values

        # Auto-detect task type
        n_unique = len(np.unique(y))
        task = "classification" if n_unique <= 20 and y.dtype in (int, np.int64, np.int32, object) else "regression"
        logger.info("Task type: %s (target='%s', unique_values=%d)", task, target, n_unique)

        # ── Leakage detection (pre-training guard) ────────────────────────────
        if run_guards:
            try:
                from modeling.leakage_detector import LeakageDetector, LeakageDetectedError
                ld = LeakageDetector()
                leakage_report = ld.detect(sub, target=target, feature_columns=feats, time_col=time_col)
                if leakage_report.high_severity_count > 0:
                    logger.error("Leakage detector blocked training: %s", leakage_report.summary)
                    raise LeakageDetectedError(leakage_report.summary)
                elif leakage_report.flags:
                    logger.warning("Leakage warnings (%d flags): %s",
                                   len(leakage_report.flags), leakage_report.summary)
            except ImportError:
                pass

        # ── Temporal split validation ─────────────────────────────────────────
        X_train_final, X_test_final, y_train_final, y_test_final = None, None, None, None
        if time_col and time_col in sub.columns:
            logger.info("Applying temporal train/test split on '%s'", time_col)
            try:
                dates = pd.to_datetime(sub[time_col], errors="coerce").sort_values()
                split_idx = int(len(dates) * 0.8)
                train_idx = dates.index[:split_idx]
                test_idx  = dates.index[split_idx:]
                X_train_final = sub.loc[train_idx, feats].values
                X_test_final  = sub.loc[test_idx, feats].values
                y_train_final = sub.loc[train_idx, target].values
                y_test_final  = sub.loc[test_idx, target].values
                logger.info("Temporal split: train=%d, test=%d", len(X_train_final), len(X_test_final))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temporal split failed (non-fatal): %s", exc)

        results: Dict[str, "TrainResult"] = {}
        for algo in self.algorithms:
            try:
                result = self._train_one(algo, X, y, feats, task, run_id)
                # ── Baseline comparator (post-training guard) ─────────────────
                if run_guards and X_test_final is not None:
                    try:
                        from modeling.baseline_comparator import BaselineComparator
                        bc = BaselineComparator(min_lift=0.02, block_on_failure=False)
                        bc_result = bc.compare(
                            result.model,
                            X_train_final, y_train_final,
                            X_test_final, y_test_final,
                            task=task, model_name=algo,
                        )
                        result.warnings.append(
                            f"Baseline comparator: {bc_result.verdict} "
                            f"({bc_result.baseline_metric_name} lift={bc_result.lift:.4f})"
                        )
                        result.cv_metrics["baseline_lift"] = round(bc_result.lift, 6)
                        result.cv_metrics["beats_baseline"] = float(bc_result.beats_baseline)
                    except Exception as exc:  # noqa: BLE001
                        result.warnings.append(f"Baseline comparator skipped: {exc}")

                results[algo] = result
                logger.info(
                    "Trained %s — CV metrics: %s",
                    algo,
                    {k: round(v, 4) for k, v in result.cv_metrics.items()},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Training failed for %s: %s", algo, exc)
                results[algo] = TrainResult(
                    run_id=run_id, model_name=algo, task=task,
                    cv_metrics={}, feature_importances=[], best_params={},
                    training_time_s=0.0, model=None,
                    warnings=[f"Training failed: {exc}"],
                )
        return results


    def _train_one(
        self,
        algo: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task: str,
        run_id: str,
    ) -> TrainResult:
        t0 = time.perf_counter()
        estimator, params = self._build_estimator(algo, task)

        # Optuna tuning (if enabled)
        best_params: Dict[str, Any] = params
        if self.use_optuna:
            estimator, best_params = self._optuna_tune(estimator, algo, task, X, y)

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state) \
            if task == "classification" else \
            KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        scoring = ["f1_weighted", "roc_auc_ovr_weighted", "accuracy"] if task == "classification" \
            else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, error_score="raise")

        # Fit final model on all data
        estimator.fit(X, y)
        elapsed = time.perf_counter() - t0

        # Feature importances
        importances = self._feature_importances(estimator, feature_names)

        # Format CV metrics
        cv_metrics = self._format_cv_metrics(cv_results, task)

        return TrainResult(
            run_id=run_id,
            model_name=algo,
            task=task,
            cv_metrics=cv_metrics,
            feature_importances=importances,
            best_params=best_params,
            training_time_s=elapsed,
            model=estimator,
        )

    def _build_estimator(self, algo: str, task: str):
        params: Dict[str, Any] = {}
        if algo == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            params = {"n_estimators": 200, "max_depth": 6, "random_state": self.random_state, "n_jobs": -1}
            cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        elif algo == "xgboost":
            try:
                from xgboost import XGBClassifier, XGBRegressor
                params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                          "subsample": 0.8, "random_state": self.random_state,
                          "eval_metric": "logloss" if task == "classification" else "rmse",
                          "verbosity": 0}
                cls = XGBClassifier if task == "classification" else XGBRegressor
            except ImportError:
                logger.warning("XGBoost not installed, falling back to GradientBoosting.")
                return self._build_estimator("gradient_boosting", task)
        elif algo == "lightgbm":
            try:
                import lightgbm as lgb
                params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                          "random_state": self.random_state, "verbosity": -1, "n_jobs": -1}
                cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
            except ImportError:
                logger.warning("LightGBM not installed, falling back to RandomForest.")
                return self._build_estimator("random_forest", task)
        elif algo == "logistic":
            from sklearn.linear_model import LogisticRegression, Ridge
            if task == "classification":
                params = {"max_iter": 1000, "random_state": self.random_state}
                cls = LogisticRegression
            else:
                params = {"alpha": 1.0}
                cls = Ridge
        elif algo == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            params = {"n_estimators": 100, "max_depth": 4, "random_state": self.random_state}
            cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        elif algo == "mlp":
            # [ML] Deep learning: Multi-Layer Perceptron
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            params = {
                "mlp__hidden_layer_sizes": (128, 64, 32),
                "mlp__activation": "relu",
                "mlp__solver": "adam",
                "mlp__max_iter": 500,
                "mlp__random_state": self.random_state,
                "mlp__early_stopping": True,
                "mlp__validation_fraction": 0.1,
            }
            if task == "classification":
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32), activation="relu",
                        solver="adam", max_iter=500, random_state=self.random_state,
                        early_stopping=True, validation_fraction=0.1,
                    )),
                ])
                return CalibratedClassifierCV(pipeline, cv=3, method="isotonic"), params
            else:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("mlp", MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32), activation="relu",
                        solver="adam", max_iter=500, random_state=self.random_state,
                        early_stopping=True, validation_fraction=0.1,
                    )),
                ])
                return pipeline, params
        else:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            params = {"n_estimators": 100, "random_state": self.random_state}
            cls = RandomForestClassifier if task == "classification" else RandomForestRegressor

        return cls(**params), params

    def _optuna_tune(self, base_estimator, algo: str, task: str, X, y):
        """Light Optuna hyperparameter search."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                if algo in ("random_forest", "gradient_boosting"):
                    n_est = trial.suggest_int("n_estimators", 50, 300)
                    max_d = trial.suggest_int("max_depth", 2, 8)
                    params_t = {"n_estimators": n_est, "max_depth": max_d, "random_state": self.random_state}
                    est, _ = self._build_estimator(algo, task)
                    est.set_params(**params_t)
                else:
                    est, _ = self._build_estimator(algo, task)
                scoring = "f1_weighted" if task == "classification" else "r2"
                cv = StratifiedKFold(3) if task == "classification" else KFold(3)
                s = cross_validate(est, X, y, cv=cv, scoring=scoring)
                return s[f"test_score"].mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best = study.best_params
            est, _ = self._build_estimator(algo, task)
            try:
                est.set_params(**best)
            except Exception:  # noqa: BLE001
                pass
            return est, best
        except ImportError:
            return base_estimator, {}

    def _feature_importances(self, model, feature_names: List[str]) -> List[Dict[str, Any]]:
        imp_arr = None
        if hasattr(model, "feature_importances_"):
            imp_arr = model.feature_importances_
        elif hasattr(model, "coef_"):
            arr = model.coef_
            imp_arr = np.abs(arr[0] if arr.ndim > 1 else arr)

        if imp_arr is None:
            return []

        sorted_idx = np.argsort(imp_arr)[::-1]
        return [
            {"feature": feature_names[i], "importance": round(float(imp_arr[i]), 6)}
            for i in sorted_idx
        ]

    def _format_cv_metrics(self, cv_results: Dict, task: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in cv_results.items():
            if k.startswith("test_"):
                name = k.replace("test_", "")
                val = float(np.mean(v))
                if name.startswith("neg_"):
                    name = name.replace("neg_", "")
                    val = -val
                out[name] = round(val, 6)
        return out
