"""
proposal/ml_confidence_scorer.py
-----------------------------------
Production-grade ML Proposal Confidence Scorer.

Purpose
-------
After proposal candidates are generated, this module assigns an ML-derived
confidence score to each candidate, helping analysts prioritize which proposals
to action first.

Architecture
------------
• Features engineered from the proposal's metadata (anomaly count, data quality
  score, column drift flag, sample size, null rate, etc.)
• RandomForestClassifier (trained on past run outcomes) predicts:
    → "high_confidence" (implement this) vs. "low_confidence" (needs review)
• Confidence probability replaces the heuristic confidence_score field.

Colab Training
--------------
See colab/train_proposal_confidence.ipynb
Exports: models/proposal_confidence.pkl

Fallback
--------
If the model artifact is absent, a weighted heuristic is used (same logic,
rule-based rather than learned).

Usage
-----
    from proposal.ml_confidence_scorer import ProposalConfidenceScorer

    scorer = ProposalConfidenceScorer()
    feature_vec = scorer.extract_features(proposal_dict, run_context)
    score = scorer.score(feature_vec)
    # score = {"confidence": 0.87, "label": "high_confidence", "method": "ml"}
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("dipex.proposal.ml_confidence_scorer")

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "proposal_confidence.pkl"
)

# ── Feature extraction ────────────────────────────────────────────────────────

# 9 features — order must match _extract_features() exactly.
_FEATURE_KEYS = [
    "drift_flag",
    "quality_score",
    "null_rate",
    "sample_size_k",
    "n_columns",
    "cv_score",
    "flag_severity_max",
    "columns_drifted",
    "proposer_type_enc",
]

_PROPOSER_TYPE_MAP = {
    "anomaly":      0,
    "feature":      1,
    "join":         2,
    "model":        3,
    "correlation":  4,
    "hypothesis":   5,
    "optimization": 6,
    "rag":          7,
    "unknown":      -1,
}


def _extract_features(
    proposal:    Dict[str, Any],
    run_context: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Convert a proposal dict into a fixed-length numeric feature vector (9 features).
    Matches the training feature set exactly — do NOT add or remove features.
    Safe to call even if most fields are absent (defaults to 0).
    """
    ctx = run_context or {}

    drift_flag         = float(bool(proposal.get("drifted",      ctx.get("drifted", False))))
    quality_score      = float(proposal.get("quality_score",     ctx.get("quality_score", 1.0)))
    null_rate          = float(proposal.get("null_rate",          ctx.get("null_rate", 0.0)))
    sample_size_k      = float(proposal.get("sample_size",        ctx.get("rows", 0))) / 1000.0
    n_columns          = float(proposal.get("n_columns",          ctx.get("n_cols", 0)))
    cv_score           = float(proposal.get("cv_score",           ctx.get("cv_score", 0.5)))
    flag_severity_max  = float(proposal.get("flag_severity_max",  ctx.get("flag_severity_max", 1)))
    columns_drifted    = float(proposal.get("columns_drifted",    ctx.get("columns_drifted", 0)))
    p_type             = proposal.get("proposer_type", "unknown")
    proposer_enc       = float(_PROPOSER_TYPE_MAP.get(str(p_type).lower(), -1))

    # NOTE: 9 features — must match Colab training exactly
    return np.array([
        drift_flag, quality_score, null_rate,
        sample_size_k, n_columns, cv_score, flag_severity_max,
        columns_drifted, proposer_enc,
    ], dtype=np.float32)


def _heuristic_score(features: np.ndarray) -> float:
    """
    Weighted heuristic confidence (fallback when model absent).
    Operates on the 9-element vector produced by _extract_features().
    Returns a [0, 1] confidence score.
    """
    # Unpack in the same order as _extract_features() / _FEATURE_KEYS
    (drift_flag, quality_score, null_rate,
     sample_size_k, n_columns, cv_score, flag_severity_max,
     columns_drifted, proposer_enc) = features

    score = 0.5
    score += drift_flag * 0.1
    score += (quality_score - 0.5) * 0.2
    score -= null_rate * 0.3
    score += min(sample_size_k / 1000.0, 0.1)
    score += (cv_score - 0.5) * 0.2
    score -= flag_severity_max * 0.05
    return float(np.clip(score, 0.05, 0.99))


# ── Scorer class ─────────────────────────────────────────────────────────────

class ProposalConfidenceScorer:
    """
    ML-powered proposal confidence scorer with graceful heuristic fallback.
    """

    def __init__(self) -> None:
        self._model: Any  = None
        self._method: str = "heuristic"
        self._load()

    def _load(self) -> None:
        try:
            import joblib  # type: ignore
            if os.path.exists(_MODEL_PATH):
                self._model  = joblib.load(_MODEL_PATH)
                self._method = "ml"
                logger.info("ProposalConfidenceScorer: ML model loaded from %s", _MODEL_PATH)
            else:
                logger.info("ProposalConfidenceScorer: model absent — using heuristic fallback.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProposalConfidenceScorer: model load failed (%s) — using heuristic.", exc)

    def extract_features(
        self,
        proposal:    Dict[str, Any],
        run_context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Extract fixed-length feature vector from a proposal dict."""
        return _extract_features(proposal, run_context)

    def score(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Assign an ML confidence score to a proposal.

        Parameters
        ----------
        features : np.ndarray from extract_features()

        Returns
        -------
        {
          "confidence": float,     # [0, 1] — higher = more confident
          "label":      str,       # "high_confidence" | "low_confidence"
          "method":     str,       # "ml" | "heuristic"
        }
        """
        if self._model is not None:
            try:
                X         = features.reshape(1, -1)
                proba     = self._model.predict_proba(X)[0]
                # Assumes class index 1 = "high_confidence"
                conf      = float(proba[1])
                label     = "high_confidence" if conf >= 0.5 else "low_confidence"
                return {"confidence": round(conf, 4), "label": label, "method": "ml"}
            except Exception as exc:  # noqa: BLE001
                logger.warning("ProposalConfidenceScorer ML predict failed: %s", exc)

        conf  = _heuristic_score(features)
        label = "high_confidence" if conf >= 0.5 else "low_confidence"
        return {"confidence": round(conf, 4), "label": label, "method": "heuristic"}

    def batch_score(
        self,
        proposals:   List[Dict[str, Any]],
        run_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score a list of proposals in batch.
        Enriches each proposal dict with a ``_ml_confidence`` key.
        """
        enriched = []
        for prop in proposals:
            feats  = self.extract_features(prop, run_context)
            result = self.score(feats)
            enriched.append({**prop, "_ml_confidence": result})
        logger.info(
            "ProposalConfidenceScorer: scored %d proposals via %s.",
            len(proposals), self._method,
        )
        return enriched
