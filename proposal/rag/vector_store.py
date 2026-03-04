"""
proposal/rag/vector_store.py
-----------------------------
Lightweight Vector Store using NumPy (No external DB dependencies).
Provides simple cosine similarity search for experience recall.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class NumpyVectorStore:
    """
    A lightweight vector store for small-scale experience memory.
    Uses NumPy for cosine similarity.
    """

    def __init__(self, storage_path: str = "data/experience_vectors.json") -> None:
        self.storage_path = Path(storage_path)
        self.ids: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.ids = data.get("ids", [])
                    self.metadatas = data.get("metadatas", [])
                    vectors = data.get("embeddings", [])
                    self.embeddings = [np.array(v) for v in vectors]
                logger.debug("Vector store loaded: %d entries.", len(self.ids))
            except Exception as e:
                logger.error("Failed to load vector store: %s", e)

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ids": self.ids,
            "metadatas": self.metadatas,
            "embeddings": [v.tolist() for v in self.embeddings]
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def add(self, id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        if id in self.ids:
            idx = self.ids.index(id)
            self.embeddings[idx] = embedding
            self.metadatas[idx] = metadata
        else:
            self.ids.append(id)
            self.embeddings.append(embedding)
            self.metadatas.append(metadata)
        self._save()

    def query(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []

        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        similarities = []
        for idx, emb in enumerate(self.embeddings):
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                score = 0.0
            else:
                score = np.dot(query_embedding, emb) / (query_norm * emb_norm)
            similarities.append((idx, float(score)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(top_k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "id": self.ids[idx],
                "metadata": self.metadatas[idx],
                "score": round(score, 4)
            })
        return results
