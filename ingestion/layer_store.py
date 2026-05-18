"""
ingestion/layer_store.py
--------------------------
Simple file-system-backed Layer Store.

Provides LayerStore — a lightweight registry for Silver/Gold artefacts
produced by the DIPEX pipeline. Used by api/routes/analyst_ops.py to
look up datasets by ID or lineage ID.

Storage layout (under data/layer_store/):
  {dataset_id}_{layer}_{timestamp}.parquet

If data_layers.LayerManager is available, delegate to it; otherwise
fall back to scanning the snapshot directory directly.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("dipex.ingestion.layer_store")

_DEFAULT_ROOT = os.path.join("data", "layer_store")
_SNAPSHOT_DIR = os.path.join("data", "snapshots")


class _Record:
    """Minimal artefact record returned by LayerStore.get / get_latest."""

    def __init__(self, path: str, dataset_id: str, layer: str, lineage_id: str) -> None:
        self.path       = path
        self.dataset_id = dataset_id
        self.layer      = layer
        self.lineage_id = lineage_id
        self.checksum   = ""
        self.created_at = ""

    def load(self) -> pd.DataFrame:
        if self.path.endswith(".parquet"):
            return pd.read_parquet(self.path)
        return pd.read_csv(self.path)


class LayerStore:
    """
    Lightweight file-system registry for Silver/Gold data artefacts.

    Usage::

        store  = LayerStore()
        record = store.get_latest(dataset_id="sales_q1", layer="silver")
        if record:
            df = record.load()
    """

    def __init__(self, root_dir: str = _DEFAULT_ROOT) -> None:
        self._root = root_dir
        os.makedirs(self._root, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_latest(
        self,
        dataset_id: str,
        layer: str = "silver",
    ) -> Optional[_Record]:
        """Return the most recent artefact for a dataset_id + layer, or None."""
        # 1. Check layer store directory
        matches = self._scan(dataset_id=dataset_id, layer=layer)
        if matches:
            path = sorted(matches)[-1]  # latest by filename timestamp
            return _Record(path, dataset_id, layer, lineage_id=os.path.basename(path))

        # 2. Fallback: snapshot dir (.parquet or .csv)
        for ext in (".parquet", ".csv"):
            snap = os.path.join(_SNAPSHOT_DIR, f"{dataset_id}_latest{ext}")
            if os.path.exists(snap):
                return _Record(snap, dataset_id, "snapshot", lineage_id=f"{dataset_id}_latest")

        logger.warning("LayerStore: no artefact found for dataset_id='%s' layer='%s'.", dataset_id, layer)
        return None

    def get(self, lineage_id: str) -> Optional[_Record]:
        """Return an artefact by its lineage_id (filename stem), or None."""
        for fname in os.listdir(self._root):
            if lineage_id in fname:
                path = os.path.join(self._root, fname)
                parts = fname.replace(".parquet", "").replace(".csv", "").split("_")
                dataset_id = parts[0] if parts else lineage_id
                layer      = parts[1] if len(parts) > 1 else "silver"
                return _Record(path, dataset_id, layer, lineage_id=lineage_id)
        return None

    def save(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        layer: str = "silver",
        lineage_id: Optional[str] = None,
    ) -> str:
        """Persist df as a parquet file and return the lineage_id."""
        import datetime as _dt
        ts  = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        lid = lineage_id or f"{dataset_id}_{layer}_{ts}"
        path = os.path.join(self._root, f"{lid}.parquet")
        df.to_parquet(path, index=False)
        logger.info("LayerStore: saved '%s' (%d rows) → %s", lid, len(df), path)
        return lid

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _scan(self, dataset_id: str, layer: str) -> list:
        results = []
        if not os.path.isdir(self._root):
            return results
        for fname in os.listdir(self._root):
            if fname.startswith(f"{dataset_id}_{layer}") and (
                fname.endswith(".parquet") or fname.endswith(".csv")
            ):
                results.append(os.path.join(self._root, fname))
        return results
