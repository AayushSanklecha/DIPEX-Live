"""
proposal/proposers/automl_proposer.py
--------------------------------------
Generates model and metric candidates using a quick AutoML cycle.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import logging

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)

_MIN_SAMPLES: int = 10

class AutoMLProposer(BaseProposer):
    """
    Suggests model candidates based on a fast train/test evaluation.
    """

    def propose(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Runs a quick AutoML cycle to find a candidate model.
        Expects 'target_col' in kwargs.
        """
        target_col = kwargs.get("target_col")
        if not target_col:
            return {"error": "target_col not provided"}

        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found"}

        # Basic preprocessing for candidate evaluation
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]

        if X.empty or len(X.columns) == 0:
            return {"error": "No numeric feature columns available"}

        if len(df) < _MIN_SAMPLES:
            return {"error": f"Insufficient data: {len(df)} rows < {_MIN_SAMPLES}"}

        is_classification = y.nunique() < 10 or pd.api.types.is_object_dtype(y)
        task = "classification" if is_classification else "regression"

        try:
            # Optional time-aware splitting if a timestamp column is provided
            time_col = kwargs.get("time_column")
            if time_col is not None and time_col in df.columns:
                ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
                non_null_mask = ts.notna()
                X = X.loc[non_null_mask]
                y = y.loc[non_null_mask]
                ts = ts.loc[non_null_mask]
                # Sort by time and take last 20% as validation (forward chaining)
                order = ts.sort_values().index
                split = int(len(order) * 0.8)
                train_idx = order[:split]
                val_idx = order[split:]
                X_train, X_test = X.loc[train_idx], X.loc[val_idx]
                y_train, y_test = y.loc[train_idx], y.loc[val_idx]
                train_index = train_idx
                val_index = val_idx
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                train_index = X_train.index
                val_index = X_test.index

            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metric_value = float(accuracy_score(y_test, y_pred))
                metric_name = "accuracy"
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metric_value = float(mean_squared_error(y_test, y_pred))
                metric_name = "mse"

            candidate = {
                "model_type": "RandomForest",
                "task": task,
                "metric_name": metric_name,
                "metric_value": round(metric_value, 4),
                "features": list(X.columns),
                "confidence": 0.8,  # Static confidence for the proposal
                # Artifacts exposed for downstream verifiers (not serialised externally)
                "artifacts": {
                    "estimator": model,
                    "X_train": X_train,
                    "y_train": y_train,
                    "y_true_val": y_test,
                    "y_pred_val": y_pred,
                    "train_index": train_index,
                    "val_index": val_index,
                },
            }

            return {
                "candidates": [candidate],
                "primary_task": task,
            }
        except Exception as e:
            logger.error("AutoMLProposer failed: %s", e)
            return {"error": str(e)}
