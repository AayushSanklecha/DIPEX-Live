"""
learning/transfer_learning.py
------------------------------
Enhancement 5: Cross-Dataset Transfer Learning.

Carries knowledge across different dataset runs so that strategies learned
on one dataset can benefit future runs on similar datasets.

How it works:
  1. After each run, a "domain fingerprint" (feature vector) is stored in
     data/state/domain_registry.json alongside the run's winning Q-values.
  2. On a new run, the fingerprint of the incoming data is compared against
     all stored fingerprints using cosine similarity.
  3. If a similar past run is found (similarity > threshold), its Q-values
     are blended into the new run's warm-start prior via a weighted average.
  4. The blend weight is proportional to the similarity score, so highly
     similar datasets get more knowledge transferred.

This means:
  - Run a banking dataset → learns banking strategies
  - Next run is also a banking dataset → immediately starts with those strategies
  - New run is a healthcare dataset → only partial transfer (different domain)

Safety:
  - Only efficiency weights (Q-values) are transferred, never safety rules
  - Transfer is bounded: source weights receive at most a 0.70 blend factor
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.transfer_learning")

_DEFAULT_REGISTRY_PATH = "data/state/domain_registry.json"
_SIMILARITY_THRESHOLD  = 0.75    # Minimum similarity for transfer to occur
_MAX_TRANSFER_WEIGHT   = 0.70    # Source knowledge contributes at most 70%
_MAX_REGISTRY_SIZE     = 100     # Max number of domain fingerprints to keep


class DomainFingerprint:
    """
    Feature vector representing the "signature" of a dataset run.
    Used for similarity-based matching between runs.
    """

    FEATURES = [
        "null_rate",
        "row_count_log",    # log10(row_count) for scale invariance
        "col_count",
        "drift_psi",
        "schema_complexity",  # number of columns / 10 (normalised)
        "confidence_score",
    ]

    def __init__(self, features: Dict[str, float]) -> None:
        self._vec: List[float] = [
            float(features.get(f, 0.0)) for f in self.FEATURES
        ]

    @classmethod
    def from_run_result(cls, metrics: Dict[str, Any]) -> "DomainFingerprint":
        """Build a fingerprint from a pipeline run's metrics dict."""
        row_count = max(1, int(metrics.get("rows_ingested", metrics.get("row_count", 1))))
        return cls({
            "null_rate":         float(metrics.get("null_rate",         0.0)),
            "row_count_log":     math.log10(row_count),
            "col_count":         float(metrics.get("col_count",         0.0)) / 100.0,
            "drift_psi":         float(metrics.get("drift_score",        metrics.get("drift_psi", 0.0))),
            "schema_complexity": float(metrics.get("schema_complexity", float(metrics.get("col_count", 10)) / 10)),
            "confidence_score":  float(metrics.get("confidence_score",  0.0)),
        })

    def to_list(self) -> List[float]:
        return list(self._vec)

    def similarity(self, other: "DomainFingerprint") -> float:
        """Cosine similarity between two fingerprint vectors."""
        a, b = self._vec, other._vec
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return float(max(0.0, min(1.0, dot / (mag_a * mag_b))))


class KnowledgeTransfer:
    """
    Cross-Dataset Transfer Learning Engine.

    Stores domain fingerprints + winning Q-values across runs and
    transfers prior knowledge to new similar datasets.

    Usage:
        kt = KnowledgeTransfer(config)
        # After a run completes:
        kt.store(run_id, domain, fingerprint, q_values)
        # Before a new run starts:
        transferred_prior = kt.transfer(new_fingerprint, target_domain)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("rl", {}).get("transfer_learning", {})
        self._path       = Path(cfg.get("registry_path", _DEFAULT_REGISTRY_PATH))
        self._threshold  = float(cfg.get("similarity_threshold", _SIMILARITY_THRESHOLD))
        self._max_weight = float(cfg.get("max_transfer_weight",  _MAX_TRANSFER_WEIGHT))
        self._max_size   = int(cfg.get("max_registry_size",      _MAX_REGISTRY_SIZE))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock       = threading.RLock()  # Thread-safe across pipeline workers

    def store(
        self,
        run_id: str,
        domain: str,
        fingerprint: DomainFingerprint,
        q_values: Dict[str, float],
    ) -> None:
        """Store a run's fingerprint and Q-values in the registry. Thread-safe."""
        entry = {
            "run_id":      run_id,
            "domain":      domain,
            "fingerprint": fingerprint.to_list(),
            "q_values":    {k: float(v) for k, v in q_values.items()},
        }
        with self._lock:
            registry = self._load()
            registry.append(entry)
            if len(registry) > self._max_size:
                registry = registry[-self._max_size:]
            self._save(registry)
        logger.info(
            "KnowledgeTransfer: stored fingerprint run_id=%s domain=%s",
            run_id, domain,
        )

    def transfer(
        self,
        target_fingerprint: DomainFingerprint,
        target_domain: Optional[str] = None,
        base_prior: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Find the most similar past run and blend its Q-values into base_prior.
        Thread-safe. Returns base_prior unchanged if no similar run found.
        """
        with self._lock:
            registry = self._load()

        if not registry:
            return base_prior or {}

        # Linear scan — O(n) — avoids redundant object construction in max()
        best_sim:   float = -1.0
        best_entry: Optional[Dict[str, Any]] = None
        for entry in registry:
            try:
                src_fp = DomainFingerprint(
                    dict(zip(DomainFingerprint.FEATURES, entry["fingerprint"]))
                )
                sim = target_fingerprint.similarity(src_fp)
                if sim > best_sim:
                    best_sim   = sim
                    best_entry = entry
            except Exception as exc:  # noqa: BLE001
                logger.debug("KnowledgeTransfer: skipping malformed entry: %s", exc)

        if best_entry is None or best_sim < self._threshold:
            logger.info(
                "KnowledgeTransfer: no match (best_sim=%.3f < threshold=%.2f)",
                max(best_sim, 0.0), self._threshold,
            )
            return base_prior or {}

        source_q      = best_entry.get("q_values", {})
        source_domain = best_entry.get("domain", "unknown")
        blend_weight  = min(self._max_weight, best_sim)
        base_weight   = 1.0 - blend_weight

        result: Dict[str, float] = {}
        for action in set(list((base_prior or {}).keys()) + list(source_q.keys())):
            base_val   = float((base_prior or {}).get(action, 0.50))
            source_val = float(source_q.get(action, base_val))
            result[action] = base_weight * base_val + blend_weight * source_val

        logger.info(
            "KnowledgeTransfer: transferred from run=%s domain=%s→%s sim=%.3f blend=%.2f",
            best_entry.get("run_id", "?"), source_domain, target_domain or "?",
            best_sim, blend_weight,
        )
        return result

    def find_most_similar(
        self,
        target_fingerprint: DomainFingerprint,
    ) -> Optional[Tuple[float, str, str]]:
        """Return (similarity, run_id, domain) for best match. Thread-safe."""
        with self._lock:
            registry = self._load()
        if not registry:
            return None

        best_sim   = -1.0
        best_entry = None
        for entry in registry:
            try:
                fp  = DomainFingerprint(dict(zip(DomainFingerprint.FEATURES, entry.get("fingerprint", []))))
                sim = target_fingerprint.similarity(fp)
                if sim > best_sim:
                    best_sim   = sim
                    best_entry = entry
            except Exception:  # noqa: BLE001
                pass

        if best_entry is None:
            return None
        return (max(0.0, best_sim), best_entry.get("run_id", ""), best_entry.get("domain", ""))

    def _load(self) -> List[Dict[str, Any]]:
        """Must be called inside self._lock."""
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KnowledgeTransfer: failed to load registry: %s", exc)
            return []

    def _save(self, registry: List[Dict[str, Any]]) -> None:
        """Atomic write via temp file + rename. Must be called inside self._lock."""
        try:
            dir_  = self._path.parent
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self._path)  # Atomic on POSIX; best-effort Windows
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as exc:  # noqa: BLE001
            logger.error("KnowledgeTransfer: failed to save registry: %s", exc)
