"""
proposal/rag_retriever.py
--------------------------
Primary RAG store for the DIPEX Proposal Layer.

Uses ChromaDB + SentenceTransformer for high-fidelity semantic search.

Embedding strategy (in priority order)
---------------------------------------
1. SentenceTransformer ('all-MiniLM-L6-v2')   — if sentence-transformers installed
2. ChromaDB built-in default embedding          — if only chromadb installed
3. Returns empty list                           — if chromadb is also absent

Strategy 2 (ChromaDB default embedding via `query_texts`) is fully functional
and gives good semantic search quality using ChromaDB's built-in tokenizer.
It is NOT a fallback in terms of quality — it is an alternative embedding
backend. The full vector store is active in both cases.

Requires: pip install chromadb
Optional: pip install sentence-transformers   (for higher-fidelity embeddings)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves past verified pipeline-run contexts using ChromaDB vector search.

    Parameters
    ----------
    db_path    : ChromaDB persistent store directory.
    model_name : SentenceTransformer model name (used when package is installed).
    """

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = db_path
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._model = None          # SentenceTransformer (optional)
        self._use_st = False        # True = use SentenceTransformer embeddings
        self._init()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self) -> None:
        """
        Initialise ChromaDB client, then optionally load SentenceTransformer.
        ChromaDB client is always initialised (it is required).
        SentenceTransformer is loaded only when available.
        """
        # ── ChromaDB (required) ───────────────────────────────────────────────
        try:
            import chromadb
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(
                name="verified_runs",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "RAGRetriever: ChromaDB store ready (size=%d) at %s",
                self._collection.count(), self.db_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RAGRetriever: ChromaDB unavailable — vector search disabled. "
                "Install with: pip install chromadb. Error: %s", exc
            )
            return

        # ── SentenceTransformer (optional, upgrades embedding quality) ────────
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_st = True
            logger.info(
                "RAGRetriever: SentenceTransformer '%s' loaded — using high-fidelity embeddings.",
                self.model_name,
            )
        except Exception as exc:  # noqa: BLE001
            self._use_st = False
            logger.info(
                "RAGRetriever: sentence-transformers not available (%s). "
                "Using ChromaDB built-in document embeddings — full vector search still active.",
                exc,
            )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_verified_run(
        self,
        run_id: str,
        summary: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Persist an approved run to the ChromaDB experience store.

        When SentenceTransformer is available, encodes the summary into a dense
        vector embedding and stores it alongside the document text.
        When only ChromaDB is available, stores the raw document text and lets
        ChromaDB handle embedding at query time.

        Parameters
        ----------
        run_id   : Unique pipeline run ID (UUID string).
        summary  : Human-readable run summary (used for semantic embedding).
        metadata : Flat dict of run metadata (str/int/float/bool values only).
        """
        if self._collection is None:
            logger.warning("RAGRetriever.add_verified_run: ChromaDB unavailable, skipping.")
            return

        # Sanitise metadata: ChromaDB only accepts str | int | float | bool values
        safe_meta: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            else:
                safe_meta[k] = str(v)

        try:
            if self._use_st and self._model is not None:
                # High-fidelity: explicit embedding via SentenceTransformer
                embedding = self._model.encode(summary).tolist()
                self._collection.upsert(
                    ids=[run_id],
                    embeddings=[embedding],
                    metadatas=[safe_meta],
                    documents=[summary],
                )
            else:
                # ChromaDB default embedding: store document text, embed at query time
                self._collection.upsert(
                    ids=[run_id],
                    metadatas=[safe_meta],
                    documents=[summary],
                )
            logger.debug("RAGRetriever: stored run_id=%s", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RAGRetriever.add_verified_run failed for run_id=%s: %s", run_id, exc
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector store for semantically similar past approved runs.

        Returns a list of dicts, each with:
          - metadata        : dict of metadata stored with the run
          - relevance_score : float ∈ [-1, 1] (1 = identical, based on cosine distance)
          - id              : run_id string

        Parameters
        ----------
        query_text : Natural-language text or dataset profile to search against.
        n_results  : Max number of results to return.
        """
        if self._collection is None:
            return []

        count = self._collection.count()
        if count == 0:
            logger.debug("RAGRetriever.retrieve: collection is empty")
            return []

        effective_n = min(n_results, count)

        try:
            if self._use_st and self._model is not None:
                # Use pre-computed SentenceTransformer embedding
                query_embedding = self._model.encode(query_text).tolist()
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=effective_n,
                )
            else:
                # ChromaDB default embedding via query_texts
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=effective_n,
                )

            contexts: List[Dict[str, Any]] = []
            if results.get("metadatas") and results["metadatas"][0]:
                ids = results.get("ids", [[]])[0]
                metas = results["metadatas"][0]
                distances = results.get("distances", [[]] * effective_n)[0]
                for i, (meta, dist) in enumerate(zip(metas, distances)):
                    contexts.append({
                        "id": ids[i] if i < len(ids) else "",
                        "metadata": meta,
                        "relevance_score": round(1.0 - float(dist), 4),
                    })
            return contexts

        except Exception as exc:  # noqa: BLE001
            logger.warning("RAGRetriever.retrieve failed: %s", exc)
            return []
