"""
learning/experience_memory.py
------------------------------
Experience Memory v1 — ChromaDB-backed store for approved pipeline runs.

Stores and retrieves approved (verifier-passed) run outcomes using
ChromaDB persistent vector storage with semantic document search.

This is the lightweight v1 store; the full HMAC-signed append-only
journal lives in ExperienceMemoryV2 (experience_memory_v2.py).
PipelineBridge uses BOTH: v2 as system-of-record, v1 as queryable index.

Design note on the import guard
--------------------------------
`chromadb` is always required (it is in requirements.txt). The try/except
below is NOT a graceful-degradation path — it is a protective startup guard
that prevents the entire Python process from crashing at import time if the
chromadb installation is broken (e.g., corrupt wheel, version conflict).
In normal operation chromadb is always available and the full stack is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── ChromaDB import guard ────────────────────────────────────────────────────
# chromadb is a required dependency (requirements.txt: chromadb>=0.5.0).
# The try/except exists only so that a broken installation fails with a clear
# error message at call-time rather than a cryptic ImportError at startup.
try:
    import chromadb as _chromadb
    _CHROMADB_OK = True
except Exception as _e:  # noqa: BLE001
    _chromadb = None  # type: ignore[assignment]
    _CHROMADB_OK = False
    logger.critical(
        "ChromaDB failed to import (%s). "
        "Run: pip install 'chromadb>=0.5.0' to fix this. "
        "ExperienceMemory will raise on use until resolved.", _e
    )


class ExperienceMemory:
    """
    Stores and retrieves approved (passed verifier) run outcomes
    using ChromaDB persistent vector storage.

    Parameters
    ----------
    db_path : str
        Directory for the ChromaDB persistent store.
        Defaults to 'data/experience_memory'.
    """

    def __init__(self, db_path: str = "data/experience_memory") -> None:
        if not _CHROMADB_OK:
            raise RuntimeError(
                "ChromaDB is not available. "
                "Install it with: pip install 'chromadb>=0.5.0'"
            )
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = _chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self._client.get_or_create_collection(
            name="verified_outcomes",
            metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
        )
        logger.info("ExperienceMemory: ChromaDB store ready at %s", self.db_path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(self, run_id: str, summary: str, metadata: Dict[str, Any]) -> None:
        """
        Persist a verified run to the ChromaDB experience memory.

        Parameters
        ----------
        run_id   : Unique identifier for the pipeline run (UUID).
        summary  : Human-readable narrative / description of the run.
        metadata : Flat dict (str values only) — confidence, dataset_id, etc.
        """
        # ChromaDB metadata values must be str | int | float | bool
        safe_meta: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            else:
                safe_meta[k] = str(v)

        try:
            # Upsert so re-running the same run_id doesn't fail
            self.collection.upsert(
                ids=[run_id],
                metadatas=[safe_meta],
                documents=[summary],
            )
            logger.debug("ExperienceMemory: stored run_id=%s", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("ExperienceMemory.store failed for run_id=%s: %s", run_id, exc)
            raise

    # ── Read ──────────────────────────────────────────────────────────────────

    def search_similar(
        self,
        context_text: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar past approved runs using ChromaDB's
        built-in document embedding search.

        Parameters
        ----------
        context_text : Natural-language query (e.g. dataset narrative, run summary).
        n_results    : Maximum number of similar results to return.

        Returns
        -------
        List of ChromaDB result dicts, each containing:
          - ids, documents, metadatas, distances
        """
        count = self.collection.count()
        if count == 0:
            logger.debug("ExperienceMemory.search_similar: collection is empty")
            return []

        effective_n = min(n_results, count)
        try:
            results = self.collection.query(
                query_texts=[context_text],
                n_results=effective_n,
            )
            return results  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            logger.error("ExperienceMemory.search_similar failed: %s", exc)
            return []

    def get_by_run_id(self, run_id: str) -> Dict[str, Any]:
        """Retrieve a stored run by its exact run_id."""
        try:
            result = self.collection.get(ids=[run_id])
            return result  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExperienceMemory.get_by_run_id failed: %s", exc)
            return {}

    def count(self) -> int:
        """Return the number of stored experience records."""
        return self.collection.count()
