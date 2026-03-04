"""
proposal/automl.py
------------------
Proposes candidate models based on dataset characteristics.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_MIN_SAMPLES: int = 10  # Minimum rows required to attempt a train/test split


class AutoMLProposal:
    """Proposes candidate models based on dataset characteristics."""

    def propose(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Runs a quick AutoML cycle to find a candidate model.

        Raises:
            ValueError: If the target column is missing, or not enough data exists.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]

        if X.empty or len(X.columns) == 0:
            raise ValueError("No numeric feature columns available for AutoML proposal.")

        if len(df) < _MIN_SAMPLES:
            raise ValueError(
                f"Dataset has only {len(df)} rows — a minimum of {_MIN_SAMPLES} is required "
                "to perform a train/test split."
            )

        is_classification = y.nunique() < 10 or pd.api.types.is_object_dtype(y)
        task = "classification" if is_classification else "regression"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            metric_value = float(accuracy_score(y_test, model.predict(X_test)))
            metric_name = "accuracy"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            metric_value = float(mean_squared_error(y_test, model.predict(X_test)))
            metric_name = "mse"

        logger.info(
            "AutoML proposal: task=%s  model=RandomForest  %s=%.4f  features=%d",
            task,
            metric_name,
            metric_value,
            len(X.columns),
        )

        return {
            "model_type": "RandomForest",
            "task": task,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "features": list(X.columns),
            "status": "PROPOSED",
        }
