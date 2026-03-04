"""
modeling/rl_automl.py
-----------------------
Production-grade RL-based AutoML Pipeline Architecture Search.

Purpose
-------
Replace static grid search with experience-driven pipeline selection.
The RL agent explores the discrete space of (scaler × model × imputer ×
feature_subset) combinations and converges on the configuration that
consistently yields the highest cross-validation score.

Architecture
------------
State   : (n_cols_bucket, n_rows_bucket, null_rate_bucket, task_type)
          — coarse discretisation to keep the state space tractable
Actions : Combinations of pipeline components:
          scaler  : {standard, robust, minmax, none}
          model   : {rf, lr, mlp, xgb, gb}
          imputer : {knn, iterative, median}
Reward  : cross_val_auc * 100 — training_time_s * 0.5

After each training run the agent records the outcome and updates the
Q-table. Over many runs it learns which pipeline works best for each
data "shape class" (small/medium/large × low/high null rate).

Persistence: data/rl_automl.json
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.modeling.rl_automl")

_DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "rl_automl.json"
)

# ── Action space ──────────────────────────────────────────────────────────────

_SCALERS  = ["standard", "robust", "minmax", "none"]
_MODELS   = ["rf", "lr", "mlp", "gb"]
_IMPUTERS = ["knn", "iterative", "median"]

ALL_ACTIONS: List[Tuple[str, str, str]] = list(
    itertools.product(_SCALERS, _MODELS, _IMPUTERS)
)

_ALPHA:     float = 0.12
_GAMMA:     float = 0.70
_EPSILON:   float = 0.15
_SAVE_PROB: float = 0.12


def _bucket(val: float, breakpoints: List[float]) -> int:
    """Map a continuous value to a discrete bucket index."""
    for i, bp in enumerate(breakpoints):
        if val <= bp:
            return i
    return len(breakpoints)


def _state_key(
    n_rows: int, n_cols: int, null_rate: float, task: str
) -> str:
    row_bucket  = _bucket(n_rows, [500, 5_000, 50_000])
    col_bucket  = _bucket(n_cols, [10, 30, 100])
    null_bucket = _bucket(null_rate, [0.01, 0.05, 0.20])
    return f"{task}::r{row_bucket}c{col_bucket}n{null_bucket}"


class RLAutoML:
    """
    Q-learning agent for AutoML pipeline architecture search.

    The agent selects the best (scaler, model, imputer) triple for a
    given data shape and learns from observed cross-validation outcomes.
    """

    def __init__(self, state_path: str = _DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.q: Dict[str, Dict[str, float]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    self.q = json.load(fh)
                logger.info("RLAutoML: loaded %d states.", len(self.q))
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLAutoML: Q-table load failed: %s", exc)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.q, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLAutoML: Save failed: %s", exc)

    # ── Q-table helpers ───────────────────────────────────────────────────────

    def _action_key(self, action: Tuple[str, str, str]) -> str:
        return "::".join(action)

    def _init_state(self, state: str) -> None:
        if state not in self.q:
            self.q[state] = {self._action_key(a): 0.0 for a in ALL_ACTIONS}

    def _update(self, state: str, action_key: str, reward: float) -> None:
        self._init_state(state)
        curr = self.q[state].get(action_key, 0.0)
        self.q[state][action_key] = round(curr + _ALPHA * (reward - curr), 6)
        if random.random() < _SAVE_PROB:
            self.save()

    # ── Public API ────────────────────────────────────────────────────────────

    def select_pipeline(
        self,
        n_rows:    int,
        n_cols:    int,
        null_rate: float,
        task:      str = "classification",
    ) -> Tuple[str, str, str]:
        """
        Select the best (scaler, model, imputer) triple for this data shape.

        Returns
        -------
        Tuple[str, str, str] : (scaler_name, model_name, imputer_name)
        """
        state = _state_key(n_rows, n_cols, null_rate, task)
        self._init_state(state)

        if random.random() < _EPSILON or not any(v != 0 for v in self.q[state].values()):
            action = random.choice(ALL_ACTIONS)
            logger.debug("[RL] AutoML exploring: %s", action)
        else:
            best_key = max(self.q[state], key=self.q[state].__getitem__)
            action   = tuple(best_key.split("::"))  # type: ignore[assignment]
            logger.debug("[RL] AutoML exploiting: %s (Q=%.4f)", action, self.q[state][best_key])

        return action  # type: ignore[return-value]

    def record_outcome(
        self,
        n_rows:          int,
        n_cols:          int,
        null_rate:       float,
        task:            str,
        pipeline:        Tuple[str, str, str],
        cv_score:        float,
        training_time_s: float,
    ) -> None:
        """
        Update the Q-table with the observed outcome.

        Parameters
        ----------
        cv_score        : Cross-validation AUC / R² (higher = better)
        training_time_s : Wall-clock time for training (lower = better)
        """
        state      = _state_key(n_rows, n_cols, null_rate, task)
        action_key = self._action_key(pipeline)
        reward     = (cv_score * 100.0) - (training_time_s * 0.5)
        self._update(state, action_key, reward)
        logger.debug(
            "[RL] AutoML: state=%s | pipeline=%s | cv=%.4f | time=%.1fs | reward=%.2f",
            state, pipeline, cv_score, training_time_s, reward,
        )

    def build_sklearn_pipeline(
        self,
        scaler:  str,
        model:   str,
        imputer: str,
        task:    str = "classification",
    ) -> Any:
        """
        Construct and return a fitted-ready sklearn Pipeline from the RL-selected components.

        Parameters
        ----------
        scaler  : "standard" | "robust" | "minmax" | "none"
        model   : "rf" | "lr" | "mlp" | "gb"
        imputer : "knn" | "iterative" | "median"
        task    : "classification" | "regression"
        """
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        steps: List[Any] = []

        # ── Imputer ────────────────────────────────────────────────────────
        if imputer == "knn":
            try:
                from sklearn.impute import KNNImputer
                steps.append(("imputer", KNNImputer(n_neighbors=5)))
            except Exception:  # noqa: BLE001
                steps.append(("imputer", SimpleImputer(strategy="median")))
        elif imputer == "iterative":
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                steps.append(("imputer", IterativeImputer(max_iter=10, random_state=42)))
            except Exception:  # noqa: BLE001
                steps.append(("imputer", SimpleImputer(strategy="median")))
        else:
            steps.append(("imputer", SimpleImputer(strategy="median")))

        # ── Scaler ────────────────────────────────────────────────────────
        if scaler == "standard":
            from sklearn.preprocessing import StandardScaler
            steps.append(("scaler", StandardScaler()))
        elif scaler == "robust":
            from sklearn.preprocessing import RobustScaler
            steps.append(("scaler", RobustScaler()))
        elif scaler == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            steps.append(("scaler", MinMaxScaler()))

        # ── Model ─────────────────────────────────────────────────────────
        if model == "rf":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            cls = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1) \
                  if task == "classification" else \
                  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        elif model == "lr":
            from sklearn.linear_model import LogisticRegression, Ridge
            cls = LogisticRegression(max_iter=1000, random_state=42) \
                  if task == "classification" else Ridge()
        elif model == "mlp":
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            cls = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                early_stopping=True, random_state=42) \
                  if task == "classification" else \
                  MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                               early_stopping=True, random_state=42)
        else:  # gb
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            cls = GradientBoostingClassifier(n_estimators=100, random_state=42) \
                  if task == "classification" else \
                  GradientBoostingRegressor(n_estimators=100, random_state=42)

        steps.append(("model", cls))
        return Pipeline(steps)

    def get_policy_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return {state: {best_pipeline, Q_value}} for all learned states."""
        summary = {}
        for state, actions in self.q.items():
            if actions and any(v != 0 for v in actions.values()):
                best_k = max(actions, key=actions.__getitem__)
                summary[state] = {"best_pipeline": best_k, "Q": actions[best_k]}
        return summary


# ── Module-level singleton ────────────────────────────────────────────────────

_RL_AUTOML: Optional[RLAutoML] = None


def get_rl_automl() -> RLAutoML:
    global _RL_AUTOML  # noqa: PLW0603
    if _RL_AUTOML is None:
        _RL_AUTOML = RLAutoML()
    return _RL_AUTOML
