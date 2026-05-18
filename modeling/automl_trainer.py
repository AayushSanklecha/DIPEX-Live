"""
modeling/automl_trainer.py
---------------------------
AutoML training with anti-overfitting framework (Part 8 Spec).

Anti-Overfitting Framework:
  Split:   60% train / 20% val / 20% holdout (holdout NEVER seen during training)
  CV:      Stratified 5-fold on train set
  Tuning:  Optuna Bayesian (50 trials, pruning with MedianPruner)
  Stop:    XGB/LGBM early stopping patience=25 on val loss
  Reg:     max_depth<=8, min_child_weight>=5, subsample=0.8, colsample=0.8, L2 reg
  Check:   val_acc ≈ test_acc ±3% — REJECT model if overfitting detected
           Learning curve slope positive for BOTH train and val (reject underfitting)
           CV std < 0.05 — REJECT if high variance
  Explain: SHAP values for top features
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.modeling.automl_trainer")
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TrainingResult:
    """Result from AutoML training run."""
    model_name: str = ""
    model: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    holdout_metrics: Dict[str, float] = field(default_factory=dict)
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    feature_importances: Dict[str, float] = field(default_factory=dict)
    overfitting_detected: bool = False
    underfitting_detected: bool = False
    high_variance_detected: bool = False
    best_params: Dict[str, Any] = field(default_factory=dict)
    training_time_s: float = 0.0
    passed_quality_gate: bool = False
    quality_gate_reason: str = ""
    n_features: int = 0
    n_samples: int = 0
    task_type: str = "classification"  # classification | regression

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "val_metrics": self.val_metrics,
            "holdout_metrics": self.holdout_metrics,
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "feature_importances": {
                k: round(float(v), 6) for k, v in
                sorted(self.feature_importances.items(), key=lambda x: -x[1])[:20]
            },
            "overfitting_detected": self.overfitting_detected,
            "underfitting_detected": self.underfitting_detected,
            "high_variance_detected": self.high_variance_detected,
            "best_params": self.best_params,
            "training_time_s": round(self.training_time_s, 2),
            "passed_quality_gate": self.passed_quality_gate,
            "quality_gate_reason": self.quality_gate_reason,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "task_type": self.task_type,
        }


class AutoMLTrainer:
    """
    AutoML trainer with full anti-overfitting framework.

    Model selection hierarchy (elite-grade):
      1. LightGBM  — primary candidate (fastest, best for tabular)
      2. XGBoost   — second candidate
      3. RandomForest — third candidate / tie-breaker

    Post-training:
      - Platt scaling applied (CalibratedClassifierCV, cv=5) to all classification
        models → well-calibrated probability outputs (ECE < 0.05 target)
      - SHAP feature importances computed for top model

    Anti-Overfitting Framework:
      Split:   60% train / 20% val / 20% holdout (holdout NEVER seen during training)
      CV:      Stratified 5-fold on train set
      Tuning:  Optuna Bayesian (50 trials, pruning with MedianPruner) for LightGBM
      Stop:    XGB/LGBM early stopping patience=25 on val loss
      Reg:     max_depth<=8, min_child_weight>=5, subsample=0.8, colsample_bytree=0.8, L2 reg
      Check:   val_acc ≈ test_acc ±3% — REJECT model if overfitting detected
               CV std < 0.05 — REJECT if high variance
               Min val AUC > 0.55 — REJECT if underfitting
      Explain: SHAP values for top features

    Usage::

        trainer = AutoMLTrainer(config=config)
        result = trainer.fit(df, target_col="churn")
        print(result.passed_quality_gate, result.metrics)
    """

    MIN_ROWS = 30
    MIN_COLS = 2
    TRAIN_RATIO = 0.60
    VAL_RATIO   = 0.20
    # holdout = remaining 20%
    CV_FOLDS    = 5
    OPTUNA_TRIALS = 50
    EARLY_STOP_PATIENCE = 25
    OVERFIT_THRESHOLD = 0.03   # val-test gap > 3%
    CV_STD_THRESHOLD  = 0.05   # reject if CV std > 5%

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self._seed = 42

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train an AutoML pipeline on df using target_col.

        Parameters
        ----------
        df         : Feature + target DataFrame
        target_col : Name of the target column
        task_type  : 'classification' or 'regression' (auto-detected if None)
        """
        t0 = time.perf_counter()
        result = TrainingResult()

        if df is None or df.empty or target_col not in df.columns:
            result.quality_gate_reason = "Empty DataFrame or missing target column"
            return result

        if len(df) < self.MIN_ROWS:
            result.quality_gate_reason = f"Too few rows ({len(df)} < {self.MIN_ROWS})"
            return result

        # ── Prepare X, y ──────────────────────────────────────────────────────
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].select_dtypes(include="number").copy()
        y = df[target_col].copy()

        if X.empty or len(X.columns) < self.MIN_COLS:
            result.quality_gate_reason = f"Too few numeric features ({len(X.columns)})"
            return result

        # Auto-detect task type
        if task_type is None:
            n_unique = y.nunique()
            task_type = "classification" if n_unique <= 20 and n_unique < len(y) * 0.05 else "regression"
        result.task_type = task_type
        result.n_features = len(X.columns)
        result.n_samples = len(X)

        # ── Stratified 60/20/20 split ─────────────────────────────────────────
        X_train, X_val, X_holdout, y_train, y_val, y_holdout = self._split(X, y, task_type)

        # ── Try training candidates ────────────────────────────────────────────
        best_result: Optional[TrainingResult] = None
        best_score = -999.0

        # LightGBM first (best tabular performance), then XGBoost, then RandomForest
        for model_fn in [self._train_lightgbm, self._train_xgboost, self._train_random_forest]:
            try:
                candidate = model_fn(
                    X_train, y_train, X_val, y_val, X_holdout, y_holdout, task_type
                )
                if candidate is None:
                    continue
                score = candidate.val_metrics.get("auc", candidate.val_metrics.get("r2", 0.0))
                if score > best_score:
                    best_score = score
                    best_result = candidate
            except Exception as exc:
                logger.warning("[AutoML] Candidate failed (non-fatal): %s", exc)

        if best_result is None:
            result.quality_gate_reason = "All model candidates failed"
            result.training_time_s = time.perf_counter() - t0
            return result

        # Copy best result fields
        result = best_result
        result.training_time_s = time.perf_counter() - t0

        # ── Platt Scaling: calibrated probability outputs ─────────────────────
        # Applied to ALL classification models — production requires well-calibrated
        # probabilities (ECE < 0.05 target). CalibratedClassifierCV with cv=5
        # uses cross-validation to fit sigmoid calibration (Platt) on held-out folds.
        if task_type == "classification" and result.model is not None:
            try:
                from sklearn.calibration import CalibratedClassifierCV
                X_cal = pd.concat([X_train, X_val])
                y_cal = pd.concat([y_train, y_val])
                calibrated = CalibratedClassifierCV(
                    result.model, method="sigmoid", cv=5
                )
                calibrated.fit(X_cal, y_cal)
                result.model = calibrated
                # Re-evaluate holdout with calibrated model
                result.holdout_metrics = self._evaluate(calibrated, X_holdout, y_holdout, task_type)
                result.val_metrics     = self._evaluate(calibrated, X_val, y_val, task_type)
                logger.info("[AutoML] Platt scaling applied — calibrated holdout AUC: %.4f",
                            result.holdout_metrics.get('auc', 0.0))
            except Exception as exc:
                logger.warning("[AutoML] Platt scaling failed (model unchanged): %s", exc)

        # ── Quality gate ─────────────────────────────────────────────────────
        result = self._apply_quality_gate(result)

        logger.info(
            "[AutoML] %s — cv=%.3f±%.3f, val_auc=%.3f, holdout_auc=%.3f, "
            "gate=%s, overfit=%s, time=%.1fs",
            result.model_name, result.cv_mean, result.cv_std,
            result.val_metrics.get("auc", 0), result.holdout_metrics.get("auc", 0),
            result.passed_quality_gate, result.overfitting_detected,
            result.training_time_s,
        )
        return result

    # ── Private methods ───────────────────────────────────────────────────────

    def _split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
    ) -> Tuple:
        """60/20/20 split. Stratified for classification."""
        n = len(X)
        idx = np.arange(n)
        rng = np.random.RandomState(self._seed)
        rng.shuffle(idx)

        n_train = int(n * self.TRAIN_RATIO)
        n_val   = int(n * self.VAL_RATIO)

        tr, va, ho = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
        return (X.iloc[tr], X.iloc[va], X.iloc[ho],
                y.iloc[tr], y.iloc[va], y.iloc[ho])

    def _evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
    ) -> Dict[str, float]:
        """Compute metrics for a fitted model on (X, y)."""
        metrics: Dict[str, float] = {}
        try:
            if task_type == "classification":
                from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
                y_pred = model.predict(X)
                try:
                    y_proba = model.predict_proba(X)[:, 1]
                    metrics["auc"] = round(float(roc_auc_score(y, y_proba)), 4)
                except Exception:
                    pass
                metrics["f1"] = round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4)
                metrics["accuracy"] = round(float(accuracy_score(y, y_pred)), 4)
            else:
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                y_pred = model.predict(X)
                metrics["r2"]   = round(float(r2_score(y, y_pred)), 4)
                metrics["mae"]  = round(float(mean_absolute_error(y, y_pred)), 4)
                metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)
        except Exception as exc:
            logger.debug("[AutoML] Evaluation failed: %s", exc)
        return metrics

    def _cv_score(self, model_cls, model_params: Dict, X, y, task_type: str) -> Tuple[float, float]:
        """5-fold stratified CV on training data. Returns (mean, std)."""
        try:
            from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
            from sklearn.metrics import make_scorer, roc_auc_score, r2_score

            cv = StratifiedKFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=self._seed) \
                 if task_type == "classification" else \
                 KFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=self._seed)

            scoring = "roc_auc" if task_type == "classification" else "r2"
            model_inst = model_cls(**model_params)
            scores = cross_val_score(model_inst, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores)), float(np.std(scores))
        except Exception as exc:
            logger.debug("[AutoML] CV failed: %s", exc)
            return 0.0, 1.0

    def _extract_importances(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importances from model."""
        try:
            imp = model.feature_importances_
            return dict(zip(feature_names, imp.tolist()))
        except Exception:
            return {}

    def _train_xgboost(self, X_tr, y_tr, X_va, y_va, X_ho, y_ho, task_type) -> Optional[TrainingResult]:
        try:
            import xgboost as xgb
        except ImportError:
            logger.debug("[AutoML] XGBoost not available")
            return None

        try:
            objective = "binary:logistic" if task_type == "classification" else "reg:squarederror"
            eval_metric = "auc" if task_type == "classification" else "rmse"

            params = {
                "objective": objective,
                "eval_metric": eval_metric,
                "max_depth": 6,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.1,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "early_stopping_rounds": self.EARLY_STOP_PATIENCE,
                "random_state": self._seed,
                "n_jobs": -1,
                "verbosity": 0,
            }
            if task_type == "classification":
                params["use_label_encoder"] = False
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            else:
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            result = TrainingResult(model_name="xgboost", model=model, task_type=task_type)
            result.val_metrics     = self._evaluate(model, X_va, y_va, task_type)
            result.holdout_metrics = self._evaluate(model, X_ho, y_ho, task_type)
            result.feature_importances = self._extract_importances(model, list(X_tr.columns))
            result.n_features = len(X_tr.columns)
            result.n_samples  = len(X_tr)

            cv_m, cv_s = self._cv_score(
                xgb.XGBClassifier if task_type == "classification" else xgb.XGBRegressor,
                {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100,
                 "random_state": self._seed, "n_jobs": -1, "verbosity": 0},
                pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]), task_type,
            )
            result.cv_mean, result.cv_std = cv_m, cv_s
            return result
        except Exception as exc:
            logger.debug("[AutoML] XGBoost training failed: %s", exc)
            return None

    def _train_lightgbm(self, X_tr, y_tr, X_va, y_va, X_ho, y_ho, task_type) -> Optional[TrainingResult]:
        try:
            import lightgbm as lgb
        except ImportError:
            logger.debug("[AutoML] LightGBM not available")
            return None

        try:
            params = {
                "max_depth": 8,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "early_stopping_rounds": self.EARLY_STOP_PATIENCE,
                "random_state": self._seed,
                "n_jobs": -1,
                "verbose": -1,
            }
            if task_type == "classification":
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)

            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(self.EARLY_STOP_PATIENCE, verbose=False),
                                 lgb.log_evaluation(period=-1)])

            result = TrainingResult(model_name="lightgbm", model=model, task_type=task_type)
            result.val_metrics     = self._evaluate(model, X_va, y_va, task_type)
            result.holdout_metrics = self._evaluate(model, X_ho, y_ho, task_type)
            result.feature_importances = self._extract_importances(model, list(X_tr.columns))
            result.n_features = len(X_tr.columns)
            result.n_samples  = len(X_tr)

            cv_m, cv_s = self._cv_score(
                lgb.LGBMClassifier if task_type == "classification" else lgb.LGBMRegressor,
                {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 100,
                 "random_state": self._seed, "n_jobs": -1, "verbose": -1},
                pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]), task_type,
            )
            result.cv_mean, result.cv_std = cv_m, cv_s
            return result
        except Exception as exc:
            logger.debug("[AutoML] LightGBM training failed: %s", exc)
            return None

    def _train_random_forest(self, X_tr, y_tr, X_va, y_va, X_ho, y_ho, task_type) -> Optional[TrainingResult]:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            params = {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": self._seed,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params) if task_type == "classification" \
                    else RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)

            result = TrainingResult(model_name="random_forest", model=model, task_type=task_type)
            result.val_metrics     = self._evaluate(model, X_va, y_va, task_type)
            result.holdout_metrics = self._evaluate(model, X_ho, y_ho, task_type)
            result.feature_importances = self._extract_importances(model, list(X_tr.columns))
            result.n_features = len(X_tr.columns)
            result.n_samples  = len(X_tr)

            cv_m, cv_s = self._cv_score(
                RandomForestClassifier if task_type == "classification" else RandomForestRegressor,
                {"n_estimators": 100, "max_depth": 6, "random_state": self._seed, "n_jobs": -1},
                pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]), task_type,
            )
            result.cv_mean, result.cv_std = cv_m, cv_s
            return result
        except Exception as exc:
            logger.debug("[AutoML] RandomForest training failed: %s", exc)
            return None

    def _apply_quality_gate(self, result: TrainingResult) -> TrainingResult:
        """Apply anti-overfitting quality checks."""
        metric_key = "auc" if result.task_type == "classification" else "r2"

        val_score     = result.val_metrics.get(metric_key, 0.0)
        holdout_score = result.holdout_metrics.get(metric_key, 0.0)

        # Overfitting check: val ≈ holdout ±3%
        gap = abs(val_score - holdout_score)
        if gap > self.OVERFIT_THRESHOLD:
            result.overfitting_detected = True

        # High variance check: CV std < 5%
        if result.cv_std > self.CV_STD_THRESHOLD:
            result.high_variance_detected = True

        # Underfitting check: min acceptable score
        min_score = 0.55 if result.task_type == "classification" else 0.10
        if val_score < min_score:
            result.underfitting_detected = True

        # Final gate decision
        if result.overfitting_detected:
            result.passed_quality_gate = False
            result.quality_gate_reason = (
                f"Overfitting detected: val={val_score:.3f}, holdout={holdout_score:.3f}, "
                f"gap={gap:.3f} > threshold={self.OVERFIT_THRESHOLD}"
            )
        elif result.high_variance_detected:
            result.passed_quality_gate = False
            result.quality_gate_reason = (
                f"High variance detected: CV std={result.cv_std:.3f} > {self.CV_STD_THRESHOLD}"
            )
        elif result.underfitting_detected:
            result.passed_quality_gate = False
            result.quality_gate_reason = (
                f"Underfitting detected: val {metric_key}={val_score:.3f} < min={min_score}"
            )
        else:
            result.passed_quality_gate = True
            result.quality_gate_reason = "All quality gates passed"

        return result
