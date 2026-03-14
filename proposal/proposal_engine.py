"""
proposal/proposal_engine.py
----------------------------
Step 4 — Proposal Layer: Master Orchestrator.

Gathers hypotheses and candidate technical paths from all sub-intelligence sources.

Simulates: "What are the most promising avenues for this specific dataset?"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from .proposers.automl_proposer import AutoMLProposer
from .proposers.anomaly_proposer import AnomalyProposer
from .proposers.ranker_proposer import RankerProposer
from .proposers.bandit_proposer import BanditProposer
from .proposers.aggregation_proposer import AggregationProposer
from .proposers.transformation_proposer import TransformationProposer
from .proposers.encoding_proposer import EncodingProposer
from .proposers.window_proposer import WindowProposer
from .rag.experience_recall import ExperienceRecall

logger = logging.getLogger(__name__)


class ProposalEngine:
    """
    Main entry point for Step 4. Orchestrates multiple technical proposers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        # Initialize sub-proposers
        self.proposers = {
            # Model and metric candidates
            "automl": AutoMLProposer(self.config),
            # Anomaly suggestions
            "anomaly": AnomalyProposer(self.config),
            # Insight ranking / high-effect signals
            "ranker": RankerProposer(self.config),
            # Contextual bandit strategies (e.g. retry strategy, encoding policy)
            "bandit": BanditProposer(self.config),
            # New: aggregations, transformations, encodings, streaming windows
            "aggregation": AggregationProposer(self.config),
            "transformation": TransformationProposer(self.config),
            "encoding": EncodingProposer(self.config),
            "window": WindowProposer(self.config),
        }
        self.recall_engine = ExperienceRecall(self.config)

    def generate_proposals(
        self,
        df: pd.DataFrame,
        run_id: str = "unknown",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Runs all proposers and aggregates results.
        """
        logger.info("ProposalEngine: generating hypotheses for run_id=%s", run_id)

        proposals: Dict[str, Any] = {
            "run_id": run_id,
            "candidates": {},
            "status": "CANDIDATES_COLLECTED",
        }

        # target_col is crucial for AutoML and Ranker
        target_col = kwargs.get("target_col") or self.config.get("pipeline", {}).get(
            "target_column"
        )
        kwargs["target_col"] = target_col

        for name, proposer in self.proposers.items():
            try:
                res = proposer.propose(df, **kwargs)
                if "error" in res:
                    logger.warning(
                        "Proposer '%s' returned error: %s", name, res.get("error")
                    )
                proposals["candidates"][name] = res
            except Exception as e:  # noqa: BLE001
                logger.error("Proposer '%s' failed unexpectedly: %s", name, e)
                proposals["candidates"][name] = {"error": str(e)}

        # Experience Recall (RAG)
        try:
            proposals["historical_precedents"] = self.recall_engine.recall(df)
            logger.info(
                "ProposalEngine: %d historical precedent(s) recalled.",
                len(proposals["historical_precedents"]),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("ExperienceRecall failed: %s", e)
            proposals["historical_precedents"] = []

        n_proposals = sum(
            1 for c in proposals["candidates"].values() if "error" not in c
        )
        
        # ── ML Confidence Scoring ──────────────────────────────────────────────
        try:
            from .ml_confidence_scorer import ProposalConfidenceScorer
            scorer = ProposalConfidenceScorer()
            for name, cand in proposals["candidates"].items():
                if "error" not in cand:
                    # Enrich candidate with basic stats for the scorer
                    cand["proposer_type"] = name
                    cand["sample_size"] = len(df)
                    cand["n_columns"] = df.shape[1]
                    
                    feat_vec = scorer.extract_features(cand, run_context=kwargs)
                    score_res = scorer.score(feat_vec)
                    cand["_ml_confidence"] = score_res
                    
            logger.info("ProposalEngine: Scored %d candidates with ML confidence.", n_proposals)
        except Exception as e:  # noqa: BLE001
            logger.warning("ProposalEngine: ML confidence scoring failed: %s", e)
        logger.info("ProposalEngine: %d proposer(s) yielded candidates.", n_proposals)

        return proposals
