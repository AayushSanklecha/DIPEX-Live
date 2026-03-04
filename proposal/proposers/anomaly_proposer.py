"""
proposal/proposers/anomaly_proposer.py
---------------------------------------
Suggests anomaly thresholds using an unsupervised Isolation Forest.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)

class AnomalyProposer(BaseProposer):
    """
    Uses Isolation Forest to detect multivariate outliers and suggests contamination thresholds.
    """

    def propose(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Runs Isolation Forest to suggest potential anomaly flags.
        """
        num_df = df.select_dtypes(include=[np.number]).dropna()
        if num_df.empty or len(num_df) < 10:
            return {"error": "Insufficient numeric data for anomaly detection"}

        cfg = self.config.get("proposal", {}).get("anomaly", {})
        n_estimators = cfg.get("n_estimators", 100)
        contamination = cfg.get("contamination", "auto")

        try:
            iso = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
            # fit_predict returns -1 for outliers, 1 for inliers
            preds = iso.fit_predict(num_df)
            outlier_count = int((preds == -1).sum())
            outlier_pct = outlier_count / len(num_df)

            # Decision Score: lower is more abnormal
            scores = iso.decision_function(num_df)
            suggested_threshold = float(np.percentile(scores, 5)) # Suggest a threshold at bottom 5%

            return {
                "anomaly_candidates": {
                    "method": "IsolationForest",
                    "detected_outlier_count": outlier_count,
                    "detected_outlier_pct": round(outlier_pct, 4),
                    "suggested_contamination": contamination if contamination != "auto" else round(outlier_pct, 4),
                    "suggested_threshold": round(suggested_threshold, 4),
                    "status": "PROPOSED"
                }
            }
        except Exception as e:
            logger.error("AnomalyProposer failed: %s", e)
            return {"error": str(e)}
