"""
proposal/proposers/ranker_proposer.py
--------------------------------------
Prioritizes high-effect signals using mutual information or correlation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import logging
import json
from pathlib import Path

from .base_proposer import BaseProposer

logger = logging.getLogger(__name__)

class RankerProposer(BaseProposer):
    """
    Ranks features by their predictive power (mutual information) to guide explanation focus.
    """

    def propose(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Computes mutual information between features and target.
        Expects 'target_col' in kwargs.
        """
        target_col = kwargs.get("target_col")
        if not target_col or target_col not in df.columns:
            return {"error": "Valid target_col required for ranking"}

        # Minimal preprocessing
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]

        if X.empty:
            return {"error": "No numeric features to rank"}

        is_classification = y.nunique() < 10 or pd.api.types.is_object_dtype(y)
        
        try:
            priors = self._load_feature_priors()
            # Handle classification mapping for y if object
            if is_classification and pd.api.types.is_object_dtype(y):
                y_encoded = pd.factorize(y)[0]
            else:
                y_encoded = y.fillna(0)

            if is_classification:
                mi = mutual_info_classif(X, y_encoded, random_state=42)
            else:
                mi = mutual_info_regression(X, y_encoded, random_state=42)

            rankings = []
            for col, score in zip(X.columns, mi):
                prior = float(priors.get(col, 0.0))
                adjusted = float(score) * (1.0 + min(2.0, prior * 0.1))
                rankings.append({
                    "column": col,
                    "score": round(float(score), 4),
                    "adjusted_score": round(float(adjusted), 4),
                    "prior_count": int(prior),
                    "impact": "high" if adjusted > 0.1 else "medium" if adjusted > 0.01 else "low"
                })

            # Sort by score descending
            rankings.sort(key=lambda x: x["adjusted_score"], reverse=True)

            return {
                "feature_importance_candidates": rankings,
                "top_insight_candidate": rankings[0]["column"] if rankings else None
            }
        except Exception as e:
            logger.error("RankerProposer failed: %s", e)
            return {"error": str(e)}

    def _load_feature_priors(self) -> Dict[str, float]:
        """
        Loads learned feature priors produced by Step 10.
        Priors are counts (or weights) keyed by column name.
        """
        path_str = (
            (self.config.get("storage", {}) or {}).get("ranker_priors_state")
            or "data/state/ranker_feature_priors.json"
        )
        path = Path(path_str)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return {k: float(v) for k, v in obj.items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("RankerProposer: failed to load priors: %s", exc)
        return {}
