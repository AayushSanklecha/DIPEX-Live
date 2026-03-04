"""
proposal/rag/experience_recall.py
----------------------------------
RAG-based retriever for past successful analysis patterns.

PRIMARY backend: RAGRetriever (ChromaDB + sentence-transformers)
  → Full semantic embedding similarity search via 'all-MiniLM-L6-v2'.
  → Stores approved run summaries as vector embeddings for high-fidelity recall.

The RAGRetriever IS the primary and only store. There is no fallback to a
lightweight NumPy store in production — this module requires chromadb and
sentence-transformers (both in requirements.txt). An import guard exists only
to prevent crashes during module load if those packages are broken; in that
case the class raises at instantiation time with a clear error message.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperienceRecall:
    """
    RAG-based retriever for past analytic experiences.

    Uses RAGRetriever (ChromaDB + SentenceTransformer) as the primary
    store for high-fidelity semantic similarity search.

    Workflow
    --------
    1. When a run is approved: call store_experience() to persist the
       run summary and metadata as a semantic vector embedding.
    2. Before/during the Proposal stage: call recall() to retrieve the
       top-k most semantically similar past approved runs. These are
       passed to proposers as historical precedents.

    Parameters
    ----------
    config : dict
        Pipeline config (loaded from config.yaml). Key paths:
          proposal.rag.db_path  → ChromaDB directory (default: data/chroma_db)
          proposal.rag.model    → Sentence-transformer model name
                                  (default: all-MiniLM-L6-v2)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        rag_cfg = self.config.get("proposal", {}).get("rag", {})

        db_path = rag_cfg.get("db_path", "data/chroma_db")
        model_name = rag_cfg.get("model", "all-MiniLM-L6-v2")

        # Import RAGRetriever (ChromaDB + sentence-transformers primary backend)
        from proposal.rag_retriever import RAGRetriever
        self._retriever = RAGRetriever(db_path=db_path, model_name=model_name)
        logger.info(
            "ExperienceRecall: RAGRetriever initialized "
            "(db=%s model=%s collection_size=%d)",
            db_path, model_name, self._get_collection_size(),
        )

    # ── Dataset profile encoding ──────────────────────────────────────────────

    @staticmethod
    def _build_query_text(df: pd.DataFrame) -> str:
        """
        Build a rich natural-language description of the DataFrame's
        characteristics for semantic embedding. This gives the sentence-
        transformer meaningful text to embed rather than raw numbers.

        Example output:
            "Dataset with 12000 rows and 18 columns. 12 numeric columns,
             6 categorical columns. Mean null rate 0.03. Schema hash: ab12cd34.
             Column types: int64, float64, object."
        """
        n_rows = len(df)
        n_cols = len(df.columns)
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df.select_dtypes(include=["object", "category"]).columns)
        null_avg = round(float(df.isnull().mean().mean()), 4)
        col_sig = hashlib.md5("".join(sorted(df.columns)).encode()).hexdigest()[:8]
        dtype_summary = ", ".join(sorted(set(str(d) for d in df.dtypes)))

        return (
            f"Dataset with {n_rows} rows and {n_cols} columns. "
            f"{num_cols} numeric columns, {cat_cols} categorical columns. "
            f"Mean null rate {null_avg}. Schema hash: {col_sig}. "
            f"Column types: {dtype_summary}."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def recall(self, df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query for semantically similar past approved runs.

        Parameters
        ----------
        df    : Current DataFrame being processed.
        top_k : Number of historical precedents to return.

        Returns
        -------
        List of dicts, each with keys:
          - metadata  : dict of stored metadata (dataset_id, confidence_score, …)
          - relevance_score : float ∈ [0, 1]  (1 = most similar)
        """
        query_text = self._build_query_text(df)
        results = self._retriever.retrieve(query_text, n_results=top_k)
        logger.debug(
            "ExperienceRecall.recall: %d precedent(s) retrieved for df shape=%s",
            len(results), df.shape,
        )
        return results

    def store_experience(
        self,
        run_id: str,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Persist an approved run to the RAG experience store.

        Called by PipelineBridge._stage_record_experience() after a run
        passes Gate 1, Gate 2, and meets the confidence threshold so that
        future runs can recall and learn from this experience.

        Parameters
        ----------
        run_id   : Unique run ID (UUID string).
        df       : DataFrame from the approved run (used to build summary).
        metadata : Flat dict with run metadata (confidence_score, dataset_id, …).
        """
        summary = self._build_query_text(df)
        # Enrich summary with confidence and dataset for better semantic retrieval
        conf = metadata.get("confidence_score", 0.0)
        ds = metadata.get("dataset_id", "unknown")
        summary = f"{summary} Run {run_id} approved. Dataset: {ds}. Confidence: {conf:.3f}."

        self._retriever.add_verified_run(run_id=run_id, summary=summary, metadata=metadata)
        logger.info(
            "ExperienceRecall: approved run stored run_id=%s dataset=%s conf=%.3f",
            run_id, ds, conf,
        )

    def _get_collection_size(self) -> int:
        """Return the number of stored experience records (best-effort)."""
        try:
            if self._retriever._collection is not None:
                return self._retriever._collection.count()
        except Exception:  # noqa: BLE001
            pass
        return 0
