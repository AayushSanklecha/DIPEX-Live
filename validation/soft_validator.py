"""
validation/soft_validator.py
------------------------------
Production-grade ML Soft Validator (Anomaly-based range validation).

Purpose
-------
Traditional range rules treat every out-of-bounds value as an ERROR.
This module uses an Isolation Forest to distinguish:

  - Hard Anomaly  : The value is truly a data entry error
                    (statistically isolated across all features).
    → Keep severity as ERROR.

  - Soft Anomaly  : The value is out-of-bounds but consistent with
                    the rest of the row's multivariate distribution
                    (e.g., a new market opening with unusually high values).
    → Downgrade severity to WARNING.

Design
------
• IsolationForest fits on the full current batch (in-memory, no Colab needed).
• Only numeric columns are used as context features.
• Graceful fallback: if sklearn is absent or < 2 numeric columns exist,
  all violations are treated as hard anomalies (conservative behaviour).

Usage
-----
    from validation.soft_validator import SoftValidator

    sv = SoftValidator()
    classification = sv.classify_violations(df, col="revenue", mask=breach_mask)
    # {"hard_count": 3, "soft_count": 2, "hard_mask": pd.Series, "soft_mask": pd.Series}
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger("dipex.validation.soft_validator")


class SoftValidator:
    """
    IsolationForest-based anomaly classifier for out-of-bounds validation violations.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in the dataset (default 1 %).
        Lower → fewer rows classified as hard anomalies.
    """

    def __init__(self, contamination: float = 0.01) -> None:
        self.contamination = contamination
        self._model: Any   = None
        self._fitted: bool = False
        self._available: bool = self._check_sklearn()

    @staticmethod
    def _check_sklearn() -> bool:
        try:
            from sklearn.ensemble import IsolationForest  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "SoftValidator: scikit-learn not installed — all violations treated as hard anomalies."
            )
            return False

    def _fit(self, X: np.ndarray) -> None:
        from sklearn.ensemble import IsolationForest
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._fitted = True

    # ── Public API ────────────────────────────────────────────────────────────

    def classify_violations(
        self,
        df:   pd.DataFrame,
        col:  str,
        mask: pd.Series,
    ) -> Dict[str, Any]:
        """
        Classify out-of-bounds rows as hard or soft anomalies.

        Parameters
        ----------
        df   : Full DataFrame (used for multivariate context)
        col  : Column that triggered the range violation
        mask : Boolean Series aligned to df.index; True = out-of-bounds row

        Returns
        -------
        {
          "hard_count": int,          # rows classified as true errors
          "soft_count": int,          # rows classified as valid novelties
          "hard_mask":  pd.Series,    # boolean mask of hard anomaly rows
          "soft_mask":  pd.Series,    # boolean mask of soft anomaly rows
          "method":     str,          # "isolation_forest" | "fallback"
        }
        """
        violation_idx = mask[mask].index
        total_violations = len(violation_idx)

        if total_violations == 0:
            return {
                "hard_count": 0, "soft_count": 0,
                "hard_mask": pd.Series(dtype=bool),
                "soft_mask": pd.Series(dtype=bool),
                "method": "no_violations",
            }

        # ── Fallback path ────────────────────────────────────────────────────
        if not self._available:
            return {
                "hard_count": total_violations, "soft_count": 0,
                "hard_mask": mask[mask],
                "soft_mask": pd.Series(False, index=violation_idx),
                "method": "fallback",
            }

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return {
                "hard_count": total_violations, "soft_count": 0,
                "hard_mask": mask[mask],
                "soft_mask": pd.Series(False, index=violation_idx),
                "method": "fallback_too_few_features",
            }

        try:
            # Fit on the full dataset (captures baseline distribution)
            X_all = df[num_cols].fillna(df[num_cols].median())
            self._fit(X_all.values)

            # Score only the violating rows
            X_viol   = X_all.loc[violation_idx]
            # predict() returns -1 (anomaly) or +1 (inlier)
            preds    = self._model.predict(X_viol.values)
            is_hard  = pd.Series(preds == -1, index=violation_idx)
            is_soft  = ~is_hard

            return {
                "hard_count": int(is_hard.sum()),
                "soft_count": int(is_soft.sum()),
                "hard_mask":  is_hard,
                "soft_mask":  is_soft,
                "method":     "isolation_forest",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("SoftValidator: IsolationForest failed (%s) — falling back.", exc)
            return {
                "hard_count": total_violations, "soft_count": 0,
                "hard_mask": mask[mask],
                "soft_mask": pd.Series(False, index=violation_idx),
                "method": "fallback_exception",
            }
