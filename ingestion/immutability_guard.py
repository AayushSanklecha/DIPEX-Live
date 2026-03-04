"""
ingestion/immutability_guard.py
---------------------------------
Copy-on-write semantics and mutation detection for the DIPEX data layer system.

This module enforces the fundamental invariant:
  Bronze and Silver layer data are NEVER mutated after storage.

Mechanisms
----------
1. SHA-256 fingerprinting of every stored DataFrame / JSON file
2. Checksum validation before any read — mismatch halts pipeline immediately
3. Defensive copy-on-write: any consumer receives a deep copy, never the original
4. Mutation probe: captures a structural signature before/after operations
5. Immutable wrapper that raises ImmutabilityViolationError on write-through attempts

Raised Errors
-------------
ImmutabilityViolationError  — any attempt to mutate a protected layer
ChecksumMismatchError       — stored checksum doesn't match on read (corruption or tampering)
LayerAccessViolationError   — wrong layer referenced for a given operation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger("dipex.ingestion.immutability_guard")


# ── Exceptions ────────────────────────────────────────────────────────────────

class ImmutabilityViolationError(RuntimeError):
    """Raised when any component attempts to mutate a Bronze or Silver layer object."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a stored checksum does not match the data on read."""


class LayerAccessViolationError(RuntimeError):
    """Raised when a component accesses a layer it has no right to write to."""


# ── Checksum utilities ────────────────────────────────────────────────────────

def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 checksum of a DataFrame.

    The checksum captures:
    - Column names (sorted for stability)
    - Column dtypes
    - Shape (rows × cols)
    - Content hash (row-by-row string digest, truncated for performance)
    """
    h = hashlib.sha256()

    # Structural fingerprint
    cols_sorted = sorted(df.columns.tolist())
    h.update(json.dumps(cols_sorted).encode())
    h.update(json.dumps({c: str(df[c].dtype) for c in cols_sorted}).encode())
    h.update(struct.pack(">QQ", *df.shape))

    # Content fingerprint (sample-based for large datasets)
    sample_size = min(10_000, len(df))
    sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    for col in cols_sorted:
        if col in sample.columns:
            col_str = sample[col].astype(str).str.cat(sep="|")
            h.update(col_str.encode(errors="replace"))

    return h.hexdigest()


def compute_file_checksum(path: str) -> str:
    """SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_json_checksum(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_file_checksum(path: str, expected: str) -> None:
    """Verify file checksum. Raises ChecksumMismatchError on mismatch."""
    actual = compute_file_checksum(path)
    if actual != expected:
        raise ChecksumMismatchError(
            f"Checksum mismatch for '{path}': "
            f"expected={expected[:16]}… actual={actual[:16]}… — "
            f"possible corruption or tampering detected"
        )


def verify_dataframe_checksum(df: pd.DataFrame, expected: str, label: str = "") -> None:
    """Verify DataFrame checksum. Raises ChecksumMismatchError on mismatch."""
    actual = compute_dataframe_checksum(df)
    if actual != expected:
        raise ChecksumMismatchError(
            f"DataFrame checksum mismatch{' for ' + label if label else ''}: "
            f"expected={expected[:16]}… actual={actual[:16]}… — "
            f"Silver/Bronze layer integrity violation"
        )


# ── Structural signature ──────────────────────────────────────────────────────

@dataclass
class DataFrameSignature:
    """Captures the structural identity of a DataFrame for before/after comparison."""
    shape: Tuple[int, int]
    columns: Tuple[str, ...]
    dtypes: Dict[str, str]
    checksum: str

    @classmethod
    def capture(cls, df: pd.DataFrame) -> "DataFrameSignature":
        return cls(
            shape=tuple(df.shape),
            columns=tuple(df.columns.tolist()),
            dtypes={c: str(df[c].dtype) for c in df.columns},
            checksum=compute_dataframe_checksum(df),
        )

    def assert_unchanged(self, after: "DataFrameSignature", context: str = "") -> None:
        """Raise ImmutabilityViolationError if anything changed."""
        if self.shape != after.shape:
            raise ImmutabilityViolationError(
                f"Layer mutation detected {context}: "
                f"shape changed {self.shape} → {after.shape}"
            )
        if self.columns != after.columns:
            raise ImmutabilityViolationError(
                f"Layer mutation detected {context}: "
                f"columns changed {set(self.columns) ^ set(after.columns)}"
            )
        if self.dtypes != after.dtypes:
            raise ImmutabilityViolationError(
                f"Layer mutation detected {context}: dtype changed — "
                f"{[(k, self.dtypes[k], after.dtypes[k]) for k in self.dtypes if self.dtypes.get(k) != after.dtypes.get(k)]}"
            )
        if self.checksum != after.checksum:
            raise ImmutabilityViolationError(
                f"Layer mutation detected {context}: "
                f"content checksum changed ({self.checksum[:16]}… → {after.checksum[:16]}…)"
            )


# ── Immutable DataFrame wrapper ───────────────────────────────────────────────

class ImmutableDataFrame:
    """
    Thin wrapper around a pandas DataFrame that:
    - Stores a reference to the underlying data (read-only access)
    - Provides .copy() for safe consumer access
    - Raises ImmutabilityViolationError on any write-through attempt
    - Verifies checksum on every .copy() call
    """

    def __init__(self, df: pd.DataFrame, layer: str, dataset_id: str) -> None:
        self._df       = df.copy(deep=True)  # store a private deep copy
        self._layer    = layer
        self._dataset_id = dataset_id
        self._checksum = compute_dataframe_checksum(self._df)
        self._signature = DataFrameSignature.capture(self._df)
        logger.debug(
            "ImmutableDataFrame locked: dataset=%s layer=%s shape=%s checksum=%s…",
            dataset_id, layer, df.shape, self._checksum[:12],
        )

    @property
    def checksum(self) -> str:
        return self._checksum

    @property
    def layer(self) -> str:
        return self._layer

    @property
    def shape(self) -> Tuple[int, int]:
        return self._df.shape

    @property
    def columns(self):
        return self._df.columns

    def copy(self) -> pd.DataFrame:
        """
        Return a deep copy for consumer use.
        Verifies integrity before returning — raises ChecksumMismatchError
        if internal state was corrupted.
        """
        self._verify_integrity()
        return self._df.copy(deep=True)

    def schema(self) -> Dict[str, str]:
        """Return column → dtype mapping (safe, non-mutating)."""
        return {c: str(self._df[c].dtype) for c in self._df.columns}

    def to_dict(self) -> Dict:
        """Return metadata dict (no data)."""
        return {
            "layer": self._layer,
            "dataset_id": self._dataset_id,
            "shape": self._df.shape,
            "columns": list(self._df.columns),
            "checksum": self._checksum,
        }

    def _verify_integrity(self) -> None:
        actual_sig = DataFrameSignature.capture(self._df)
        self._signature.assert_unchanged(actual_sig, context=f"({self._layer}/{self._dataset_id})")

    # Prevent accidental mutation via standard pandas indexing
    def __setitem__(self, key, value):
        raise ImmutabilityViolationError(
            f"Direct assignment to ImmutableDataFrame ({self._layer}/{self._dataset_id}) "
            f"is not allowed. Use .copy() to get a mutable Gold layer copy."
        )

    def __repr__(self) -> str:
        return (
            f"ImmutableDataFrame(layer={self._layer!r}, dataset_id={self._dataset_id!r}, "
            f"shape={self._df.shape}, checksum={self._checksum[:12]}…)"
        )


# ── Mutation probe context manager ────────────────────────────────────────────

class MutationProbe:
    """
    Context manager that captures a DataFrame signature before an operation
    and verifies it is unchanged after.

    Usage::

        with MutationProbe(silver_df, "preprocessing stage") as probe:
            result = some_transform(gold_df)   # operates on a copy
        # If silver_df was touched (bug/attack), raises ImmutabilityViolationError
    """

    def __init__(self, protected_df: pd.DataFrame, context: str = "") -> None:
        self._protected = protected_df
        self._context   = context
        self._before: Optional[DataFrameSignature] = None

    def __enter__(self) -> "MutationProbe":
        self._before = DataFrameSignature.capture(self._protected)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        after = DataFrameSignature.capture(self._protected)
        self._before.assert_unchanged(after, context=self._context)
        # We do not suppress exceptions
        return False


# ── Layer write guard ─────────────────────────────────────────────────────────

class LayerWriteGuard:
    """
    Enforces which components may write to which layers.

    Allowed writes:
      Bronze : ingestion system only
      Silver : normalisation/schema-registry only
      Gold   : any analyst operation, ML, stats, preprocessing, etc.
    """

    ALLOWED_WRITERS: Dict[str, set] = {
        "bronze": {"ingestion"},
        "silver": {"ingestion", "normaliser", "schema_registry"},
        "gold":   {
            "junior_analyst", "mid_analyst", "senior_analyst", "preprocessing",
            "feature_engineering", "statistics", "modeling",
            "sql_transform", "excel_transform", "streaming",
            "reporting", "dashboard", "profiling", "cognitive",
            "analyst_orchestrator",
        },
    }

    @classmethod
    def assert_write_allowed(cls, layer: str, component: str) -> None:
        allowed = cls.ALLOWED_WRITERS.get(layer.lower(), set())
        if component.lower() not in allowed:
            raise LayerAccessViolationError(
                f"Component '{component}' is NOT allowed to write to the {layer.upper()} layer. "
                f"Allowed writers: {sorted(allowed)}"
            )
