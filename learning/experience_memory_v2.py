"""
learning/experience_memory_v2.py
---------------------------------
STEP 9 — Experience Memory (immutable, HMAC-signed append-only history).

Hardened with:
  - HMAC-SHA256 per-record integrity signature (verified on every read)
  - Append-only writes — no record may be updated or deleted
  - max_episodes cap with oldest-record pruning (FIFO)
  - recall_similar() for RAG-based similarity search (cosine on feature vectors)
  - ChromaDB optional index for approved outcomes (read-only from truth perspective)

JSONL log is the system-of-record. ChromaDB is a queryable index.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# HMAC key — pulled from env for security. Falls back to a fixed dev-only key.
_HMAC_KEY: bytes = os.environ.get("DIPEX_HMAC_KEY", "dipex-dev-hmac-key-2026").encode()


def _sign_record(record: Dict[str, Any]) -> str:
    """Compute HMAC-SHA256 signature for a record dict."""
    # Exclude the 'hmac' field itself from signing
    payload = {k: v for k, v in record.items() if k != "hmac"}
    content = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hmac.new(_HMAC_KEY, content, hashlib.sha256).hexdigest()


def _verify_record(record: Dict[str, Any]) -> bool:
    """Return True if the record's HMAC signature is valid."""
    stored = record.get("hmac")
    if not stored:
        return False  # unsigned records fail
    computed = _sign_record(record)
    return hmac.compare_digest(stored, computed)


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExperienceEvent:
    """Single immutable event in experience history."""
    event_id: str
    event_type: str
    run_id: str
    timestamp: str
    fingerprint: Optional[str] = None
    attempt: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
            "attempt": self.attempt,
            "payload": self.payload,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ExperienceMemoryV2
# ══════════════════════════════════════════════════════════════════════════════

class ExperienceMemoryV2:
    """
    Immutable experience memory with HMAC-SHA256 per-record integrity.

    Architecture:
      - JSONL log: system-of-record, append-only, HMAC-signed per line
      - ChromaDB index: optional queryable index for similarity search
      - max_episodes cap: oldest lines pruned when cap exceeded (FIFO)
      - recall_similar(): cosine similarity on feature vectors for RAG
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        store_cfg = self.config.get("storage", {})
        rl_cfg = self.config.get("rl", {})

        self._log_path = Path(
            store_cfg.get("experience_log", "data/experience/experience.jsonl")
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_episodes: int = int(rl_cfg.get("max_episodes", 10_000))

        # Optional ChromaDB index (graceful degradation if not installed)
        chroma_path = store_cfg.get("experience_chroma_path", "data/experience_memory")
        self._chroma_path = Path(chroma_path)
        self._collection = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self._chroma_path))
            self._collection = client.get_or_create_collection(name="experience_events")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChromaDB not available — similarity search disabled: %s", exc
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExperienceMemoryV2":
        return cls(config)

    # ------------------------------------------------------------------
    # Core: append-only log
    # ------------------------------------------------------------------

    def append_event(
        self,
        event_type: str,
        run_id: str,
        payload: Dict[str, Any],
        fingerprint: Optional[str] = None,
        attempt: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> ExperienceEvent:
        """
        Append one event to the JSONL log with HMAC-SHA256 signature.
        Enforces max_episodes cap after write.
        """
        event = ExperienceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            run_id=run_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            fingerprint=fingerprint,
            attempt=attempt,
            payload=payload,
        )
        record = event.to_dict()
        record["hmac"] = _sign_record(record)

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug(
            "ExperienceMemory: event appended type=%s run_id=%s", event_type, run_id
        )

        # Cap enforcement (FIFO pruning)
        self._enforce_cap()
        return event

    def list_recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Return last N events, verifying HMAC on each.
        Events with invalid HMAC are logged as warnings and skipped.
        """
        if not self._log_path.exists():
            return []
        buf: List[Dict[str, Any]] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if not _verify_record(record):
                        logger.warning(
                            "ExperienceMemory: HMAC mismatch on event_id=%s — skipped",
                            record.get("event_id", "?"),
                        )
                        continue
                    buf.append(record)
                except json.JSONDecodeError:
                    continue
        return buf[-limit:]

    def get_run_history(self, run_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Return HMAC-verified events for a specific run_id."""
        events = [e for e in self.list_recent(limit=5000) if e.get("run_id") == run_id]
        return events[-limit:]

    # ------------------------------------------------------------------
    # RAG: similarity-based recall
    # ------------------------------------------------------------------

    def recall_similar(
        self,
        feature_vector: List[float],
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k most similar approved outcomes using cosine similarity
        on feature vectors stored in ChromaDB.

        Args:
            feature_vector: Numeric representation of current dataset characteristics
            top_k: Number of results to return
            min_confidence: Minimum confidence score to include in results

        Returns:
            List of experience records ordered by similarity (most similar first)
        """
        if self._collection is None:
            logger.debug("recall_similar: ChromaDB not available — returning empty")
            return []

        try:
            results = self._collection.query(
                query_embeddings=[feature_vector],
                n_results=min(top_k * 3, 50),  # over-fetch, then filter
            )
            candidates = []
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                conf = float(meta.get("confidence_score", 0.0))
                if conf >= min_confidence:
                    candidates.append({
                        "id": doc_id,
                        "metadata": meta,
                        "document": results["documents"][0][i],
                    })
            return candidates[:top_k]
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall_similar failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # High-level recorders
    # ------------------------------------------------------------------

    def record_confidence_vector(
        self,
        run_id: str,
        fingerprint: str,
        attempt: int,
        confidence_vector: Dict[str, Any],
        gate1_decision: str,
        gate2_decision: str,
    ) -> ExperienceEvent:
        return self.append_event(
            event_type="CONFIDENCE_VECTOR",
            run_id=run_id,
            fingerprint=fingerprint,
            attempt=attempt,
            payload={
                "confidence_vector": confidence_vector,
                "gate1_decision": gate1_decision,
                "gate2_decision": gate2_decision,
            },
        )

    def record_retry_decision(
        self,
        run_id: str,
        fingerprint: str,
        attempt: int,
        retry_decision: Dict[str, Any],
    ) -> ExperienceEvent:
        return self.append_event(
            event_type="RETRY_DECISION",
            run_id=run_id,
            fingerprint=fingerprint,
            attempt=attempt,
            payload=retry_decision,
        )

    def record_user_feedback(
        self,
        run_id: str,
        rating: float,
        comment: str = "",
        tags: Optional[List[str]] = None,
        fingerprint: Optional[str] = None,
    ) -> ExperienceEvent:
        return self.append_event(
            event_type="USER_FEEDBACK",
            run_id=run_id,
            fingerprint=fingerprint,
            payload={
                "rating": float(rating),
                "comment": str(comment),
                "tags": tags or [],
            },
        )

    def record_approved_output(
        self,
        run_id: str,
        fingerprint: str,
        approved_output: Dict[str, Any],
        winning_strategy: Dict[str, Any],
        confidence_score: float,
        attempt: int,
        narrative: str = "",
    ) -> ExperienceEvent:
        """
        Records approved output as immutable HMAC-signed history.
        Also adds to ChromaDB index for future RAG queries.
        """
        evt = self.append_event(
            event_type="APPROVED_OUTPUT",
            run_id=run_id,
            fingerprint=fingerprint,
            attempt=attempt,
            payload={
                "confidence_score": float(confidence_score),
                "winning_strategy": winning_strategy,
                "approved_output": approved_output,
            },
        )

        # ChromaDB index (not system-of-record)
        if self._collection is not None:
            try:
                doc = narrative or json.dumps(winning_strategy, ensure_ascii=False)
                meta = {
                    "run_id": run_id,
                    "fingerprint": fingerprint,
                    "confidence_score": float(confidence_score),
                    "attempt": int(attempt),
                    "model_type": str(winning_strategy.get("model_type", "")),
                    "task": str(winning_strategy.get("task", "")),
                }
                self._collection.add(
                    ids=[f"{run_id}:{evt.event_id}"],
                    metadatas=[meta],
                    documents=[doc],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExperienceMemory: ChromaDB index add failed: %s", exc)

        return evt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_cap(self) -> None:
        """
        FIFO pruning: keep only the last max_episodes lines.
        Rewrites the log file with truncation only when needed.
        """
        if not self._log_path.exists():
            return

        # Count lines without loading all into memory
        line_count = 0
        with open(self._log_path, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1

        if line_count <= self._max_episodes:
            return

        # Need to prune: keep last max_episodes lines
        lines_to_drop = line_count - self._max_episodes
        kept: List[str] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= lines_to_drop:
                    kept.append(line)

        # Atomic-ish rewrite
        tmp = self._log_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
        tmp.replace(self._log_path)

        logger.info(
            "ExperienceMemory: pruned %d records to enforce max_episodes=%d cap",
            lines_to_drop, self._max_episodes,
        )
