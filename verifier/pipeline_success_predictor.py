"""
verifier/pipeline_success_predictor.py
----------------------------------------
Production-grade ML Pipeline Success Predictor.

Purpose
-------
Before a full pipeline run, this module predicts whether the run is
likely to succeed (valid results) or fail (data quality issues,
model degradation, drift-induced failure). This enables early warnings
and proactive intervention.

Architecture
------------
Features derived from the ingestion metadata, profiling results, and drift report:
  - null_rate, drift_detected, quality_score, row_count_k,
    n_columns, anomaly_count, schema_match, known_dataset

Classifier: RandomForestClassifier (Colab-trained)
Fallback  : Weighted heuristic threshold-based rule

Colab Training
--------------
See colab/train_pipeline_predictor.ipynb
Exports: models/pipeline_success_predictor.pkl

Usage
-----
    from verifier.pipeline_success_predictor import PipelineSuccessPredictor

    predictor = PipelineSuccessPredictor()
    result = predictor.predict(run_context)
    # result = {"success_prob": 0.87, "prediction": "LIKELY_SUCCESS",
    #           "warnings": [], "method": "ml"}
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.verifier.success_predictor")

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "pipeline_success_predictor.pkl"
)

_FEATURE_KEYS = [
    "null_rate", "drift_detected", "quality_score", "row_count_k",
    "n_columns", "anomaly_count", "schema_match", "known_dataset",
    "cv_score", "columns_drifted",
]


def _extract_features(ctx: Dict[str, Any]) -> np.ndarray:
    """Extract fixed-length feature vector from a run context dict."""
    return np.array([
        float(ctx.get("null_rate",       0.0)),
        float(bool(ctx.get("drift_detected", False))),
        float(ctx.get("quality_score",   1.0)),
        float(ctx.get("row_count",       0)) / 1000.0,
        float(ctx.get("n_columns",       0)),
        float(ctx.get("anomaly_count",   0)),
        float(bool(ctx.get("schema_match", True))),
        float(bool(ctx.get("known_dataset", True))),
        float(ctx.get("cv_score",        0.5)),
        float(ctx.get("columns_drifted", 0)),
    ], dtype=np.float32)


def _heuristic_predict(features: np.ndarray) -> float:
    """Weighted heuristic success probability."""
    (null_rate, drift, quality, rows_k, n_cols, anomalies,
     schema_match, known, cv_score, cols_drifted) = features

    score = 1.0
    score -= null_rate * 0.4
    score -= drift * 0.2
    score += (quality - 0.5) * 0.3
    score += min(rows_k / 1000.0, 0.1)
    score -= (anomalies / max(n_cols, 1)) * 0.2
    score += (schema_match - 0.5) * 0.2
    score += (cv_score - 0.5) * 0.2
    score -= cols_drifted * 0.05
    return float(np.clip(score, 0.05, 0.99))


class PipelineSuccessPredictor:
    """ML-based pipeline run success predictor."""

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
                logger.info("PipelineSuccessPredictor: ML model loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.info("PipelineSuccessPredictor: heuristic mode (%s).", exc)

    def predict(
        self,
        run_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict whether a pipeline run will succeed.

        Parameters
        ----------
        run_context : Dict with any/all of _FEATURE_KEYS fields

        Returns
        -------
        {
          "success_prob":  float,    # [0, 1] — probability of success
          "prediction":    str,      # "LIKELY_SUCCESS" | "AT_RISK" | "LIKELY_FAILURE"
          "warnings":      List[str],
          "method":        str,
        }
        """
        feats = _extract_features(run_context)

        if self._model is not None:
            try:
                proba = self._model.predict_proba(feats.reshape(1, -1))[0]
                # Assumes class 1 = success
                prob = float(proba[1])
            except Exception as exc:  # noqa: BLE001
                logger.warning("PipelineSuccessPredictor ML predict failed: %s", exc)
                prob = _heuristic_predict(feats)
        else:
            prob = _heuristic_predict(feats)

        if prob >= 0.75:
            prediction = "LIKELY_SUCCESS"
        elif prob >= 0.45:
            prediction = "AT_RISK"
        else:
            prediction = "LIKELY_FAILURE"

        warnings: List[str] = []
        if feats[1] == 1.0:  # drift_detected
            warnings.append("Data drift detected — model predictions may degrade.")
        if feats[0] > 0.15:  # null_rate
            warnings.append(f"High null rate ({feats[0]:.1%}) — imputation quality may suffer.")
        if feats[5] > 10:    # anomaly_count
            warnings.append(f"High anomaly count ({int(feats[5])}) — review data quality.")

        logger.info(
            "PipelineSuccessPredictor [%s]: prediction=%s (prob=%.3f), %d warnings.",
            self._method, prediction, prob, len(warnings),
        )
        return {
            "success_prob": round(prob, 4),
            "prediction":   prediction,
            "warnings":     warnings,
            "method":       self._method,
        }
