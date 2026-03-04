"""
reporting_service/chart_relevance_scorer.py
-------------------------------------------
Production-grade ML Chart Relevance Scorer.

Purpose
-------
Given a DataFrame + analyst context, ranks available chart types
(bar, line, scatter, heatmap, histogram, box, pie) by their statistical
relevance to the data shape, ensuring the most informative chart is
always generated first.

Architecture
------------
Features per candidate chart type:
  - Data shape signals (n_rows, n_cols, numeric ratio, cardinality)
  - Column dtype profile
  - Statistical signals (skewness, correlation density, null rate)
  - Analyst query intent (if available)

Classifier: RandomForestClassifier trained on (features, best_chart)
            pairs from historical user interactions.

If the model artifact is absent, a rule-based heuristic ranks chart
types using statistical properties of the data.

Colab Training
--------------
See colab/train_chart_scorer.ipynb
Exports: models/chart_relevance_scorer.pkl

Usage
-----
    from reporting_service.chart_relevance_scorer import ChartRelevanceScorer

    scorer = ChartRelevanceScorer()
    ranking = scorer.rank(df, query_intent="trend")
    # ranking = [
    #   {"chart_type": "line",  "score": 0.92},
    #   {"chart_type": "bar",   "score": 0.61},
    #   ...
    # ]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.reporting.chart_relevance")

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "chart_relevance_scorer.pkl"
)

_CHART_TYPES = ["bar", "line", "scatter", "heatmap", "histogram", "box", "pie"]

# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(
    df:            pd.DataFrame,
    query_intent:  Optional[str] = None,
) -> np.ndarray:
    """Extract a numeric feature vector for chart type selection."""
    n_rows   = len(df)
    n_cols   = len(df.columns)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_ratio = len(num_cols) / max(n_cols, 1)
    cat_ratio = len(cat_cols) / max(n_cols, 1)

    # Cardinality of first categorical column
    first_cat_card = df[cat_cols[0]].nunique() / max(n_rows, 1) if cat_cols else 0.0

    # Skewness of first numeric column
    skew = float(df[num_cols[0]].skew()) if num_cols else 0.0
    skew = np.clip(skew, -10, 10)

    # Mean absolute correlation between numeric columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        mean_corr = float(corr.values.mean())
    else:
        mean_corr = 0.0

    # Null rate
    null_rate = float(df.isnull().mean().mean())

    # Has datetime
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()
    has_dt    = float(len(dt_cols) > 0 or any("date" in c.lower() or "time" in c.lower()
                                               for c in df.columns))

    # Query intent encoding
    intent_map = {
        "trend": 0, "time_series": 0, "distribute": 1, "distribution": 1,
        "compare": 2, "group_by": 2, "correlation": 3, "top_n": 4, "bottom_n": 4,
        "aggregate": 5, "general": 6,
    }
    intent_enc = float(intent_map.get(query_intent or "general", 6)) / 6.0

    return np.array([
        min(n_rows / 10_000, 1.0),   # row density
        min(n_cols / 50.0, 1.0),     # col density
        num_ratio, cat_ratio,
        first_cat_card, skew,
        mean_corr, null_rate,
        has_dt, intent_enc,
    ], dtype=np.float32)


# ── Heuristic ranking ─────────────────────────────────────────────────────────

_HEURISTIC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "line":      {"has_dt": 0.9, "num_ratio": 0.6, "intent_trend": 1.0},
    "bar":       {"cat_ratio": 0.8, "intent_compare": 0.9, "intent_top_n": 1.0},
    "scatter":   {"mean_corr": 0.7, "num_ratio": 0.8, "intent_correlation": 1.0},
    "heatmap":   {"mean_corr": 0.9, "n_cols": 0.5},
    "histogram": {"num_ratio": 0.9, "skew_abs": 0.7, "intent_distribution": 1.0},
    "box":       {"num_ratio": 0.7, "cat_ratio": 0.5, "intent_distribution": 0.8},
    "pie":       {"cat_ratio": 0.9, "first_cat_card_low": 0.8},
}


def _heuristic_rank(
    df:           pd.DataFrame,
    query_intent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rule-based chart ranking (fallback when model absent)."""
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    has_dt   = any("date" in c.lower() or "time" in c.lower() for c in df.columns)
    num_ratio = len(num_cols) / max(len(df.columns), 1)
    cat_ratio = len(cat_cols) / max(len(df.columns), 1)

    scores: Dict[str, float] = {c: 0.3 for c in _CHART_TYPES}

    intent = (query_intent or "general").lower()

    if has_dt or "trend" in intent or "time" in intent:
        scores["line"] += 0.6
    if cat_ratio > 0.2 or "compare" in intent or "top" in intent:
        scores["bar"]  += 0.5
    if num_ratio > 0.5 and "corr" in intent:
        scores["scatter"] += 0.5
    if num_ratio > 0.8 and len(num_cols) > 3:
        scores["heatmap"] += 0.4
    if "distribut" in intent or "histogram" in intent:
        scores["histogram"] += 0.6
        scores["box"]       += 0.4
    if cat_ratio > 0.3 and len(cat_cols) > 0:
        card = df[cat_cols[0]].nunique()
        if card <= 10:
            scores["pie"] += 0.5

    total = sum(scores.values()) or 1.0
    return sorted(
        [{"chart_type": ct, "score": round(v / total, 4)} for ct, v in scores.items()],
        key=lambda x: -x["score"],
    )


# ── Main class ────────────────────────────────────────────────────────────────

class ChartRelevanceScorer:
    """ML-based chart type ranker for the reporting layer."""

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
                logger.info("ChartRelevanceScorer: model loaded from %s", _MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.info("ChartRelevanceScorer: heuristic mode (%s).", exc)

    def rank(
        self,
        df:           pd.DataFrame,
        query_intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank chart types by relevance to the DataFrame and analyst intent.

        Returns
        -------
        List of {"chart_type": str, "score": float} sorted descending by score.
        """
        if self._model is not None:
            try:
                feats = _extract_features(df, query_intent).reshape(1, -1)
                probas = self._model.predict_proba(feats)[0]
                classes = list(self._model.classes_)
                result = sorted(
                    [{"chart_type": ct, "score": round(float(probas[i]), 4)}
                     for i, ct in enumerate(classes)],
                    key=lambda x: -x["score"],
                )
                logger.debug("[ML] ChartScorer: ranked %d chart types via ML.", len(result))
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChartRelevanceScorer ML failed: %s — using heuristic.", exc)

        return _heuristic_rank(df, query_intent)
