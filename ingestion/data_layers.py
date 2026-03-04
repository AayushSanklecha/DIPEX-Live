"""
ingestion/data_layers.py
--------------------------
Bronze / Silver / Gold data layer manager for DIPEX.

MANDATORY ARCHITECTURAL CONTRACT
---------------------------------
Bronze  — Raw ingested data. Immutable. Exact source copy. Hash-locked.
Silver  — Normalised ISSF snapshot. Immutable after creation.
Gold    — All analyst/ML/stats/reporting operations. Regenerable. Versioned.

Bronze and Silver layers are NEVER written to by anything except the ingestion
system. Gold is always regenerable from Silver. Silver is regenerable only by
a full re-ingestion run.

API
---
    lm = LayerManager()
    bronze = lm.store_bronze(df, dataset_id, snapshot_id)    # lock raw
    silver = lm.promote_to_silver(bronze)                    # after normalisation
    gold   = lm.derive_gold(silver, dataset_id="sales_agg", # analyst operation
                            component="junior_analyst",
                            transform_fn=my_aggregation)
    lm.verify_layer(silver.dataset_id, "silver")             # on-demand verify
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple, Any

import pandas as pd

from ingestion.immutability_guard import (
    ImmutableDataFrame,
    ImmutabilityViolationError,
    LayerAccessViolationError,
    LayerWriteGuard,
    MutationProbe,
    compute_dataframe_checksum,
    compute_file_checksum,
    DataFrameSignature,
)
from ingestion.lineage import LineageRecord, LineageStore, TransformationStep

logger = logging.getLogger("dipex.ingestion.data_layers")


# ── Layer Record (metadata) ───────────────────────────────────────────────────

@dataclass
class LayerRecord:
    """Persistent metadata entry for a stored layer."""
    layer_id: str
    layer: str                      # bronze | silver | gold
    dataset_id: str
    snapshot_id: str
    checksum: str
    shape: Tuple[int, int]
    file_path: str
    meta_path: str
    created_at: str
    component: str                  # who created this record
    lineage_id: Optional[str]       = None
    source_snapshot_id: Optional[str] = None
    tags: Dict[str, str]            = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "layer_id": self.layer_id,
            "layer": self.layer,
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "checksum": self.checksum,
            "shape": list(self.shape),
            "file_path": self.file_path,
            "meta_path": self.meta_path,
            "created_at": self.created_at,
            "component": self.component,
            "lineage_id": self.lineage_id,
            "source_snapshot_id": self.source_snapshot_id,
            "tags": self.tags,
        }


# ── Gold Artefact (returned to callers) ──────────────────────────────────────

@dataclass
class GoldArtefact:
    """
    Result of a Gold layer derivation.
    Contains the mutable DataFrame for use by analysts/ML/stats.
    Always carries a LineageRecord for full auditability.
    """
    dataset_id: str
    data: pd.DataFrame              # mutable — Gold layer
    lineage: LineageRecord
    checksum: str
    created_at: str
    component: str

    def to_dict(self) -> Dict:
        return {
            "dataset_id": self.dataset_id,
            "shape": list(self.data.shape),
            "checksum": self.checksum,
            "created_at": self.created_at,
            "component": self.component,
            "lineage_id": self.lineage.lineage_id,
            "source_snapshot_id": self.lineage.source_snapshot_id,
            "transformation_steps": len(self.lineage.transformation_steps),
        }


# ── Layer Manager ─────────────────────────────────────────────────────────────

class LayerManager:
    """
    Central authority for Bronze / Silver / Gold data layer operations.

    Enforces:
    - Immutability of Bronze and Silver after storage
    - Copy-on-write for all Gold derivations
    - Checksum lock/verify before every read
    - Lineage tracking for every Gold artefact
    - Component-level write access control

    Storage layout::

        data/
          bronze/<dataset_id>/<snapshot_id>_bronze.parquet  + .meta.json
          silver/<dataset_id>/<snapshot_id>_silver.parquet  + .meta.json
          gold/<dataset_id>/<layer_id>_gold.parquet         + .meta.json

    """

    CHECKSUM_LOCK_EXT = ".sha256"

    def __init__(
        self,
        base_dir: str = "data",
        lineage_store: Optional[LineageStore] = None,
    ) -> None:
        self.base_dir = base_dir
        self.lineage_store = lineage_store or LineageStore(
            store_dir=os.path.join(base_dir, "lineage")
        )
        for layer in ("bronze", "silver", "gold"):
            os.makedirs(os.path.join(base_dir, layer), exist_ok=True)

    # ── Bronze ────────────────────────────────────────────────────────────────

    def store_bronze(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        snapshot_id: str,
        component: str = "ingestion",
        tags: Optional[Dict] = None,
    ) -> ImmutableDataFrame:
        """
        Lock raw ingested data as Bronze.
        Only 'ingestion' component may call this.
        Returns an ImmutableDataFrame — callers get a copy via .copy().
        """
        LayerWriteGuard.assert_write_allowed("bronze", component)
        df_copy = df.copy(deep=True)
        checksum = compute_dataframe_checksum(df_copy)

        layer_dir = self._layer_dir("bronze", dataset_id)
        parquet_path = os.path.join(layer_dir, f"{snapshot_id}_bronze.parquet")
        df_copy.to_parquet(parquet_path, index=False)
        file_checksum = compute_file_checksum(parquet_path)

        # Write checksum lock file
        lock_path = parquet_path + self.CHECKSUM_LOCK_EXT
        with open(lock_path, "w") as f:
            f.write(file_checksum)

        record = LayerRecord(
            layer_id=str(uuid.uuid4()),
            layer="bronze",
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            checksum=checksum,
            shape=tuple(df_copy.shape),
            file_path=parquet_path,
            meta_path=parquet_path + ".meta.json",
            created_at=datetime.now(timezone.utc).isoformat(),
            component=component,
            tags=tags or {},
        )
        self._save_meta(record)
        logger.info(
            "Bronze locked: dataset=%s snapshot=%s shape=%s checksum=%s…",
            dataset_id, snapshot_id, df_copy.shape, checksum[:12],
        )
        return ImmutableDataFrame(df_copy, layer="bronze", dataset_id=dataset_id)

    def load_bronze(self, dataset_id: str, snapshot_id: str) -> ImmutableDataFrame:
        """Load Bronze layer with checksum verification."""
        return self._load_immutable("bronze", dataset_id, snapshot_id)

    # ── Silver ────────────────────────────────────────────────────────────────

    def promote_to_silver(
        self,
        bronze: ImmutableDataFrame,
        normalised_df: pd.DataFrame,
        dataset_id: str,
        snapshot_id: str,
        component: str = "normaliser",
        tags: Optional[Dict] = None,
    ) -> ImmutableDataFrame:
        """
        Promote normalised data to Silver layer.
        Only 'ingestion', 'normaliser', 'schema_registry' components may call this.
        Original Bronze is never modified — normalised_df must be a new DataFrame.
        """
        LayerWriteGuard.assert_write_allowed("silver", component)

        # Verify Bronze integrity hasn't changed since loading
        bronze._verify_integrity()

        df_copy = normalised_df.copy(deep=True)
        checksum = compute_dataframe_checksum(df_copy)

        layer_dir = self._layer_dir("silver", dataset_id)
        parquet_path = os.path.join(layer_dir, f"{snapshot_id}_silver.parquet")
        df_copy.to_parquet(parquet_path, index=False)
        file_checksum = compute_file_checksum(parquet_path)

        lock_path = parquet_path + self.CHECKSUM_LOCK_EXT
        with open(lock_path, "w") as f:
            f.write(file_checksum)

        record = LayerRecord(
            layer_id=str(uuid.uuid4()),
            layer="silver",
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            checksum=checksum,
            shape=tuple(df_copy.shape),
            file_path=parquet_path,
            meta_path=parquet_path + ".meta.json",
            created_at=datetime.now(timezone.utc).isoformat(),
            component=component,
            source_snapshot_id=snapshot_id,
            tags=tags or {},
        )
        self._save_meta(record)
        logger.info(
            "Silver promoted: dataset=%s snapshot=%s shape=%s checksum=%s…",
            dataset_id, snapshot_id, df_copy.shape, checksum[:12],
        )
        return ImmutableDataFrame(df_copy, layer="silver", dataset_id=dataset_id)

    def load_silver(self, dataset_id: str, snapshot_id: str) -> ImmutableDataFrame:
        """Load Silver layer with checksum verification."""
        return self._load_immutable("silver", dataset_id, snapshot_id)

    # ── Gold ──────────────────────────────────────────────────────────────────

    def derive_gold(
        self,
        silver: ImmutableDataFrame,
        dataset_id: str,
        component: str,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        step_name: str = "transform",
        operator: str = "system",
        parameters: Optional[Dict] = None,
        source_snapshot_id: str = "",
        tags: Optional[Dict] = None,
    ) -> GoldArtefact:
        """
        Derive a Gold layer artefact from a Silver ImmutableDataFrame.

        - Silver is NEVER touched: transform_fn receives a deep copy
        - MutationProbe wraps the Silver during transform to detect any leak
        - Result is stored with checksum and lineage record
        - Returns named GoldArtefact with mutable DataFrame

        Parameters
        ----------
        silver         : ImmutableDataFrame (Silver or Bronze source)
        dataset_id     : Name for the resulting Gold artefact
        component      : e.g. 'junior_analyst', 'senior_analyst', 'modeling'
        transform_fn   : Pure function (pd.DataFrame → pd.DataFrame)
        step_name      : Human-readable name for this transformation
        operator       : 'system' | user identifier
        parameters     : Optional metadata about transformation parameters
        source_snapshot_id : For lineage tracking
        """
        LayerWriteGuard.assert_write_allowed("gold", component)
        silver._verify_integrity()

        # Build lineage record
        lineage = LineageRecord(
            source_dataset_id=silver._dataset_id,
            source_snapshot_id=source_snapshot_id or silver._dataset_id,
            source_layer=silver.layer,
            generating_component=component,
            operator=operator,
            output_dataset_id=dataset_id,
            tags=tags or {},
        )

        # Copy-on-write: give transform_fn a mutable copy, never the silver ref
        t0 = time.perf_counter()
        input_df = silver.copy()  # deep copy — verified by ImmutableDataFrame
        input_shape = input_df.shape

        # MutationProbe: if transform_fn somehow escapes the copy and touches silver's
        # internal _df (e.g., via captured reference), we catch it here
        with MutationProbe(silver._df, context=f"derive_gold/{component}/{step_name}"):
            try:
                output_df = transform_fn(input_df)
                success = True
                error_msg = None
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                success = False
                logger.error("Gold derivation failed — %s: %s", step_name, error_msg)
                output_df = input_df  # return the input copy unchanged on failure

        elapsed = (time.perf_counter() - t0) * 1000

        step = TransformationStep(
            step_name=step_name,
            component=component,
            input_layer=silver.layer,
            output_layer="gold",
            parameters=parameters or {},
            input_shape=input_shape,
            output_shape=tuple(output_df.shape),
            elapsed_ms=round(elapsed, 2),
            success=success,
            error=error_msg,
        )
        lineage.add_step(step)

        if not success:
            # Failed transformation does NOT partially write to gold
            raise RuntimeError(
                f"Gold derivation '{step_name}' by '{component}' failed. "
                f"Silver layer untouched. Error: {error_msg}"
            )

        # Store gold
        output_df = output_df.copy(deep=True)
        gold_checksum = compute_dataframe_checksum(output_df)
        lineage.seal(gold_checksum, tuple(output_df.shape))

        layer_dir = self._layer_dir("gold", dataset_id)
        layer_id  = str(uuid.uuid4())
        parquet_path = os.path.join(layer_dir, f"{layer_id}_gold.parquet")
        output_df.to_parquet(parquet_path, index=False)
        file_checksum = compute_file_checksum(parquet_path)
        lock_path = parquet_path + self.CHECKSUM_LOCK_EXT
        with open(lock_path, "w") as f:
            f.write(file_checksum)

        record = LayerRecord(
            layer_id=layer_id,
            layer="gold",
            dataset_id=dataset_id,
            snapshot_id=layer_id,
            checksum=gold_checksum,
            shape=tuple(output_df.shape),
            file_path=parquet_path,
            meta_path=parquet_path + ".meta.json",
            created_at=datetime.now(timezone.utc).isoformat(),
            component=component,
            lineage_id=lineage.lineage_id,
            source_snapshot_id=source_snapshot_id,
            tags=tags or {},
        )
        self._save_meta(record)
        try:
            self.lineage_store.save(lineage)
        except Exception:  # noqa: BLE001
            pass

        logger.info(
            "Gold derived: %s → %s by '%s' | shape=%s checksum=%s… elapsed=%.0fms",
            silver._dataset_id, dataset_id, component,
            output_df.shape, gold_checksum[:12], elapsed,
        )
        return GoldArtefact(
            dataset_id=dataset_id,
            data=output_df,
            lineage=lineage,
            checksum=gold_checksum,
            created_at=record.created_at,
            component=component,
        )

    # ── Verification ──────────────────────────────────────────────────────────

    def verify_layer(
        self, dataset_id: str, layer: str, snapshot_id: str
    ) -> Dict[str, Any]:
        """
        Verify a stored layer's file checksum against its lock file.
        Returns a verification report dict.
        Raises ChecksumMismatchError if tampering detected.
        """
        from ingestion.immutability_guard import verify_file_checksum
        layer_dir = self._layer_dir(layer, dataset_id)
        meta_files = [f for f in os.listdir(layer_dir) if f.startswith(snapshot_id) and f.endswith(".meta.json")]
        if not meta_files:
            return {"verified": False, "error": f"No meta file found for {snapshot_id}"}
        meta_path = os.path.join(layer_dir, meta_files[0])
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        parquet_path = meta["file_path"]
        lock_path    = parquet_path + self.CHECKSUM_LOCK_EXT
        if not os.path.exists(lock_path):
            return {"verified": False, "error": "Checksum lock file missing — possible tampering"}
        with open(lock_path) as f:
            expected = f.read().strip()
        verify_file_checksum(parquet_path, expected)
        return {
            "verified": True,
            "layer": layer,
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "checksum": expected,
            "shape": meta.get("shape"),
        }

    def list_layer(self, layer: str, dataset_id: Optional[str] = None) -> List[Dict]:
        """List all records in a layer, optionally filtered by dataset_id."""
        results = []
        base = os.path.join(self.base_dir, layer)
        if not os.path.isdir(base):
            return []
        datasets = [dataset_id] if dataset_id else os.listdir(base)
        for ds in datasets:
            ds_dir = os.path.join(base, ds)
            if not os.path.isdir(ds_dir):
                continue
            for f in os.listdir(ds_dir):
                if f.endswith(".meta.json"):
                    try:
                        with open(os.path.join(ds_dir, f), encoding="utf-8") as fh:
                            results.append(json.load(fh))
                    except Exception:  # noqa: BLE001
                        pass
        return results

    # ── Internals ─────────────────────────────────────────────────────────────

    def _layer_dir(self, layer: str, dataset_id: str) -> str:
        import re
        safe_id = re.sub(r"[^\w\-]", "_", dataset_id)[:64]
        d = os.path.join(self.base_dir, layer, safe_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _load_immutable(
        self, layer: str, dataset_id: str, snapshot_id: str
    ) -> ImmutableDataFrame:
        from ingestion.immutability_guard import verify_file_checksum
        layer_dir = self._layer_dir(layer, dataset_id)
        parquet_path = os.path.join(layer_dir, f"{snapshot_id}_{layer}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"{layer.capitalize()} layer not found: {parquet_path}")
        lock_path = parquet_path + self.CHECKSUM_LOCK_EXT
        if os.path.exists(lock_path):
            with open(lock_path) as f:
                expected = f.read().strip()
            verify_file_checksum(parquet_path, expected)
        df = pd.read_parquet(parquet_path)
        return ImmutableDataFrame(df, layer=layer, dataset_id=dataset_id)

    def _save_meta(self, record: LayerRecord) -> None:
        with open(record.meta_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
