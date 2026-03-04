"""
cognitive/sanity_prioritizer.py
---------------------------------
Production-grade ML Cognitive Sanity Check Prioritizer.

Purpose
-------
The cognitive layer generates many sanity checks (business rule validations,
statistical sanity flags, cross-column consistency checks). This module ranks
those checks by severity and likelihood of being real analysts' blockers
rather than false alarms.

Architecture
------------
• Uses a RandomForestClassifier trained on historical check outcomes.
• Features: statistical signal strength, null rate, column cardinality,
  past flag rate, inter-column correlation, and violation count.
• Output: {check_id: priority_score} — sorted descending.
• Fallback: statistical severity heuristic.

Colab Training
--------------
See colab/train_sanity_scorer.ipynb
Exports: models/sanity_prioritizer.pkl

Usage
-----
    from cognitive.sanity_prioritizer import SanityPrioritizer

    p = SanityPrioritizer()
    ranked = p.rank_checks(checks_list, df)
    # ranked = [{"check": ..., "priority": 0.93}, ...]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.cognitive.sanity_prioritizer")

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "sanity_prioritizer.pkl"
)


def _extract_check_features(
    check:  Dict[str, Any],
    df:     Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Extract numeric features for a single sanity check."""
    col      = check.get("column", "")
    severity = check.get("severity", "INFO")
    sev_enc  = {"CRITICAL": 1.0, "ERROR": 0.8, "WARNING": 0.5, "INFO": 0.2}.get(severity.upper(), 0.2)
    violation_rate = float(check.get("violation_rate", check.get("offending_count", 0)) /
                           max(check.get("total_count", 1), 1))
    past_flag_rate = float(check.get("past_flag_rate", 0.0))
    confidence     = float(check.get("confidence", 0.5))
    cross_col      = float(bool(check.get("cross_column", False)))
    business_rule  = float(bool(check.get("business_rule", False)))

    # Column-level signals from df
    if df is not None and col and col in df.columns:
        null_rate   = float(df[col].isnull().mean())
        card        = float(df[col].nunique() / max(len(df), 1))
        is_num      = float(pd.api.types.is_numeric_dtype(df[col]))
        skew        = float(df[col].skew()) if is_num and len(df[col].dropna()) > 3 else 0.0
        skew        = np.clip(skew, -10, 10)
    else:
        null_rate = card = is_num = 0.0
        skew = 0.0

    return np.array([
        sev_enc, violation_rate, past_flag_rate, confidence,
        cross_col, business_rule, null_rate, card, is_num, skew,
    ], dtype=np.float32)


def _heuristic_priority(feats: np.ndarray) -> float:
    sev, viol, past, conf, cross, biz, null, card, is_num, skew = feats
    score = sev * 0.35 + viol * 0.25 + conf * 0.15 + biz * 0.10 + cross * 0.10
    score += abs(skew) * 0.01
    return float(np.clip(score, 0.01, 1.0))


class SanityPrioritizer:
    """ML-driven cognitive sanity check priority ranker."""

    def __init__(self) -> None:
        self._model: Any  = None
        self._method: str = "heuristic"
        self._load()

    def _load(self) -> None:
        try:
            import joblib
            if os.path.exists(_MODEL_PATH):
                self._model  = joblib.load(_MODEL_PATH)
                self._method = "ml"
                logger.info("SanityPrioritizer: ML model loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.info("SanityPrioritizer: heuristic mode (%s).", exc)

    def _score_one(self, check: Dict[str, Any], df: Optional[pd.DataFrame]) -> float:
        feats = _extract_check_features(check, df)
        if self._model is not None:
            try:
                proba = self._model.predict_proba(feats.reshape(1, -1))[0]
                return float(max(proba))
            except Exception:  # noqa: BLE001
                pass
        return _heuristic_priority(feats)

    def rank_checks(
        self,
        checks: List[Dict[str, Any]],
        df:     Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank a list of sanity checks by priority.

        Parameters
        ----------
        checks : List of sanity check dicts from the cognitive layer
        df     : Optional DataFrame for column-level feature enrichment

        Returns
        -------
        List of {check_dict, priority: float} sorted descending by priority.
        """
        result = []
        for check in checks:
            score = self._score_one(check, df)
            result.append({**check, "priority": round(score, 4), "_method": self._method})

        ranked = sorted(result, key=lambda x: -x["priority"])
        logger.info(
            "SanityPrioritizer [%s]: ranked %d checks. Top priority=%.3f.",
            self._method, len(ranked), ranked[0]["priority"] if ranked else 0,
        )
        return ranked
