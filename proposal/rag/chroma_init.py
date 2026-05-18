# proposal/rag/chroma_init.py
"""
ChromaDB vector store initialiser for DIPEX RAG pipeline.

Issue 08: Operational documentation and initialisation for ChromaDB.

Persistence strategy:
  Development : ./data/chroma (local directory, gitignored)
  Production  : CHROMA_PERSIST_PATH env var (mount as Docker volume)

Fallback strategy:
  If ChromaDB is unavailable or empty, the proposal engine falls back
  to random-selection AutoML — no error is raised, but a warning is logged.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "dipex_transformations"


def get_chroma_client():
    """
    Returns a ChromaDB persistent client.
    Creates the persistence directory if it doesn't exist.
    """
    import chromadb

    persist_path = os.environ.get("CHROMA_PERSIST_PATH", "./data/chroma")
    Path(persist_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_path)
    logger.info("ChromaDB initialised at: %s", persist_path)
    return client


def get_or_create_collection():
    """
    Returns the DIPEX transformations collection.
    Creates it on first run (idempotent).
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count = collection.count()
    if count == 0:
        logger.warning(
            "ChromaDB collection '%s' is empty. "
            "The RAG proposal engine will use AutoML fallback until "
            "past transformation data is loaded. "
            "Run: python main.py seed-rag --source historical_runs/",
            _COLLECTION_NAME,
        )
    else:
        logger.info(
            "ChromaDB collection '%s' loaded with %d transformation records.",
            _COLLECTION_NAME, count,
        )
    return collection


def chroma_healthcheck() -> bool:
    """
    Returns True if ChromaDB is reachable and collection exists.
    Used in startup checks and /health API endpoint.
    """
    try:
        col = get_or_create_collection()
        return col is not None
    except Exception as exc:
        logger.error("ChromaDB healthcheck failed: %s", exc)
        return False
