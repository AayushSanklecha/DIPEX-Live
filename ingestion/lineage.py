"""
ingestion/lineage.py
---------------------
Data lineage tracker for the DIPEX Bronze/Silver/Gold layer architecture.

Every derived dataset MUST carry a complete lineage record so that:
  - Any result can be traced back to its Silver (or Bronze) source
  - Every transformation step is logged with version and operator
  - Audit regulators can reconstruct any artefact from scratch
  - Corruption in Gold can be traced to its root cause

LineageRecord fields (all mandatory)
-------------------------------------
source_dataset_id     : ID of the originating ISSF dataset
source_snapshot_id    : Exact snapshot from which this was derived
source_layer          : 'bronze' | 'silver'
transformation_steps  : ordered list of TransformationStep
transformation_version: semver of the transformation pipeline
generating_component  : which system module created this (e.g. 'junior_analyst')
operator              : 'system' | <user_id>
timestamp             : ISO 8601 UTC
output_layer          : always 'gold'
output_checksum       : SHA-256 of the resulting Gold DataFrame/artefact
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.ingestion.lineage")

LINEAGE_STORE_PATH = "data/lineage"


# ── Transformation Step ───────────────────────────────────────────────────────

@dataclass
class TransformationStep:
    """One logical step in a transformation pipeline."""
    step_name: str                    # e.g. 'remove_duplicates', 'filter_nulls'
    component: str                    # e.g. 'junior_analyst', 'feature_engineering'
    input_layer: str                  # 'bronze' | 'silver' | 'gold'
    output_layer: str                 # always 'gold'
    parameters: Dict[str, Any]        = field(default_factory=dict)
    input_shape: Optional[tuple]      = None
    output_shape: Optional[tuple]     = None
    elapsed_ms: float                 = 0.0
    success: bool                     = True
    error: Optional[str]              = None
    timestamp: str                    = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("input_shape"):
            d["input_shape"] = list(d["input_shape"])
        if d.get("output_shape"):
            d["output_shape"] = list(d["output_shape"])
        return d


# ── Lineage Record ────────────────────────────────────────────────────────────

@dataclass
class LineageRecord:
    """
    Complete provenance for a single derived (Gold layer) artefact.
    Nothing may exist without a LineageRecord.
    """
    lineage_id: str                   = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    source_dataset_id: str            = ""
    source_snapshot_id: str           = ""
    source_layer: str                 = "silver"
    transformation_steps: List[TransformationStep] = field(default_factory=list)
    transformation_version: str       = "1.0.0"
    generating_component: str         = "system"
    operator: str                     = "system"
    timestamp: str                    = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    output_layer: str                 = "gold"
    output_dataset_id: str            = ""
    output_checksum: Optional[str]    = None
    output_shape: Optional[tuple]     = None
    tags: Dict[str, str]              = field(default_factory=dict)

    def add_step(self, step: TransformationStep) -> None:
        self.transformation_steps.append(step)
        logger.debug(
            "Lineage [%s]: step added '%s' by '%s' (%s→%s)",
            self.lineage_id[:8], step.step_name, step.component,
            step.input_layer, step.output_layer,
        )

    def seal(self, output_checksum: str, output_shape: tuple) -> None:
        """Finalise the lineage record once the Gold artefact is complete."""
        self.output_checksum = output_checksum
        self.output_shape    = output_shape
        logger.info(
            "Lineage [%s] sealed: %s → %s (%d steps), checksum=%s…",
            self.lineage_id[:8], self.source_dataset_id, self.output_dataset_id,
            len(self.transformation_steps), output_checksum[:12],
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("output_shape"):
            d["output_shape"] = list(d["output_shape"])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def validate(self) -> None:
        """Raise ValueError if any mandatory field is missing."""
        missing = [
            f for f in ("source_dataset_id", "source_snapshot_id",
                         "generating_component", "output_layer")
            if not getattr(self, f)
        ]
        if missing:
            raise ValueError(f"LineageRecord is incomplete — missing: {missing}")
        if any(s.output_layer in ("bronze", "silver")
               for s in self.transformation_steps):
            raise ValueError(
                "LineageRecord contains a step that writes BACK to Bronze or Silver — "
                "this violates immutability guarantees."
            )


# ── Lineage Store ─────────────────────────────────────────────────────────────

class LineageStore:
    """
    Persistent store for LineageRecords.

    Layout on disk::

        data/lineage/<dataset_id>/<lineage_id>.json

    Provides lookup by:
      - lineage_id
      - source_snapshot_id (find all Gold artefacts derived from a Silver snapshot)
      - generating_component (find all Gold artefacts produced by, e.g., 'junior_analyst')
    """

    def __init__(self, store_dir: str = LINEAGE_STORE_PATH) -> None:
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save(self, record: LineageRecord) -> str:
        """Persist a LineageRecord and return its path."""
        record.validate()
        dataset_dir = os.path.join(self.store_dir, _safe(record.output_dataset_id or record.source_dataset_id))
        os.makedirs(dataset_dir, exist_ok=True)
        path = os.path.join(dataset_dir, f"{record.lineage_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(record.to_json())
        logger.info("LineageRecord saved: %s", path)
        return path

    def get(self, lineage_id: str) -> Optional[LineageRecord]:
        """Load a LineageRecord by ID (searches recursively)."""
        for root, _, files in os.walk(self.store_dir):
            fname = f"{lineage_id}.json"
            if fname in files:
                with open(os.path.join(root, fname), encoding="utf-8") as f:
                    data = json.load(f)
                return self._from_dict(data)
        return None

    def list_for_snapshot(self, source_snapshot_id: str) -> List[LineageRecord]:
        """Find all Gold artefacts derived from a given Silver snapshot."""
        results = []
        for root, _, files in os.walk(self.store_dir):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(root, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("source_snapshot_id") == source_snapshot_id:
                        results.append(self._from_dict(data))
                except Exception:  # noqa: BLE001
                    pass
        return results

    def list_for_dataset(self, dataset_id: str) -> List[LineageRecord]:
        """Find all Gold artefacts for a given dataset."""
        results = []
        dataset_dir = os.path.join(self.store_dir, _safe(dataset_id))
        if not os.path.isdir(dataset_dir):
            return []
        for fname in os.listdir(dataset_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(dataset_dir, fname), encoding="utf-8") as f:
                        results.append(self._from_dict(json.load(f)))
                except Exception:  # noqa: BLE001
                    pass
        return sorted(results, key=lambda r: r.timestamp, reverse=True)

    @staticmethod
    def _from_dict(d: Dict) -> LineageRecord:
        steps = [TransformationStep(**s) for s in d.pop("transformation_steps", [])]
        d.pop("output_shape", None)  # will become None if not present
        rec = LineageRecord(**{k: v for k, v in d.items()
                               if k in LineageRecord.__dataclass_fields__})
        rec.transformation_steps = steps
        return rec


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", str(s))[:64]
