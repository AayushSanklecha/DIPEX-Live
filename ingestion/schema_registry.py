"""
ingestion/schema_registry.py
------------------------------
Persistent, versioned schema registry with drift detection.

Every successful ingestion a schema snapshot is stored under:
  data/schema_registry/<dataset_id>/v<version>.json

Drift categories
----------------
ADDITIVE     : New column added           → MINOR version bump, pipeline continues
MISSING      : Expected column removed    → MAJOR version bump, pipeline STOPS
TYPE_CHANGE  : Column dtype changed       → MAJOR version bump, pipeline STOPS
RENAME       : Column appears renamed     → BREAKING, pipeline STOPS
NO_CHANGE    : Schema identical           → no action

Version format: MAJOR.MINOR.PATCH (semver-lite)
  MAJOR : Breaking changes (missing col, type change)
  MINOR : Additive changes (new column)
  PATCH : Row-count only change (no schema change)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.ingestion.schema_registry")


# ── Drift Result ──────────────────────────────────────────────────────────────

@dataclass
class ColumnDrift:
    column: str
    change_type: str           # ADDITIVE | MISSING | TYPE_CHANGE | NO_CHANGE
    old_dtype: Optional[str]
    new_dtype: Optional[str]
    severity: str              # MINOR | BREAKING


@dataclass
class SchemaDriftReport:
    dataset_id: str
    old_version: Optional[str]
    new_version: str
    changes: List[ColumnDrift]
    is_breaking: bool
    summary: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        return {
            "dataset_id": self.dataset_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "is_breaking": self.is_breaking,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "changes": [
                {
                    "column": c.column,
                    "change_type": c.change_type,
                    "old_dtype": c.old_dtype,
                    "new_dtype": c.new_dtype,
                    "severity": c.severity,
                }
                for c in self.changes
            ],
        }


# ── Schema Version ────────────────────────────────────────────────────────────

class SchemaVersion:
    """Semver-lite version manager (MAJOR.MINOR.PATCH)."""

    def __init__(self, version_str: str = "1.0.0") -> None:
        parts = version_str.split(".")
        try:
            self.major = int(parts[0])
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            self.major, self.minor, self.patch = 1, 0, 0

    def bump_major(self) -> "SchemaVersion":
        sv = SchemaVersion(str(self))
        sv.major += 1; sv.minor = 0; sv.patch = 0
        return sv

    def bump_minor(self) -> "SchemaVersion":
        sv = SchemaVersion(str(self))
        sv.minor += 1; sv.patch = 0
        return sv

    def bump_patch(self) -> "SchemaVersion":
        sv = SchemaVersion(str(self))
        sv.patch += 1
        return sv

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)


# ── Schema Registry ───────────────────────────────────────────────────────────

class SchemaRegistry:
    """
    Persistent schema store. Compares incoming schema with previous version,
    categorises drift, bumps version, and writes new schema file.

    Usage::

        registry = SchemaRegistry()
        schema   = {"user_id": "int64", "revenue": "float64"}
        report   = registry.register(dataset_id="sales", schema=schema)
        if report.is_breaking:
            raise SchemaError(report.summary)
    """

    INITIAL_VERSION = "1.0.0"

    def __init__(self, registry_dir: str = "data/schema_registry") -> None:
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        dataset_id: str,
        schema: Dict[str, str],
        row_count: int = 0,
        source_uri: str = "",
    ) -> SchemaDriftReport:
        """
        Register a schema for `dataset_id`.
        - If first time: store as v1.0.0, return NO_CHANGE report.
        - Otherwise: compare, determine drift, bump version, store.
        """
        prev_record = self._load_latest(dataset_id)

        if prev_record is None:
            # First ingestion → store initial version
            self._store(dataset_id, schema, self.INITIAL_VERSION, row_count, source_uri)
            return SchemaDriftReport(
                dataset_id=dataset_id,
                old_version=None,
                new_version=self.INITIAL_VERSION,
                changes=[],
                is_breaking=False,
                summary=f"Initial schema registered ({len(schema)} columns) as v{self.INITIAL_VERSION}.",
            )

        old_schema  = prev_record["schema"]
        old_version = prev_record["version"]

        # Compute drift
        changes = self._compute_drift(old_schema, schema)
        is_breaking = any(c.severity == "BREAKING" for c in changes)

        # Bump version
        sv = SchemaVersion(old_version)
        if is_breaking:
            new_sv = sv.bump_major()
        elif any(c.change_type == "ADDITIVE" for c in changes):
            new_sv = sv.bump_minor()
        else:
            new_sv = sv.bump_patch()

        new_version = str(new_sv)
        self._store(dataset_id, schema, new_version, row_count, source_uri)

        summary = self._build_summary(changes, old_version, new_version)
        report = SchemaDriftReport(
            dataset_id=dataset_id,
            old_version=old_version,
            new_version=new_version,
            changes=changes,
            is_breaking=is_breaking,
            summary=summary,
        )

        if is_breaking:
            logger.error(
                "BREAKING schema drift in dataset '%s': %s (v%s → v%s)",
                dataset_id, summary, old_version, new_version,
            )
        elif changes:
            logger.warning(
                "Non-breaking schema drift in dataset '%s': %s (v%s → v%s)",
                dataset_id, summary, old_version, new_version,
            )
        else:
            logger.debug("Schema unchanged for dataset '%s' (v%s)", dataset_id, new_version)

        return report

    def get_history(self, dataset_id: str) -> List[Dict]:
        """Return all stored schema versions for a dataset, oldest first."""
        dir_path = os.path.join(self.registry_dir, _safe_id(dataset_id))
        if not os.path.isdir(dir_path):
            return []
        files = sorted(
            f for f in os.listdir(dir_path) if f.endswith(".json")
        )
        history = []
        for f in files:
            with open(os.path.join(dir_path, f), encoding="utf-8") as fh:
                history.append(json.load(fh))
        return history

    def get_latest_version(self, dataset_id: str) -> Optional[str]:
        rec = self._load_latest(dataset_id)
        return rec["version"] if rec else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _dataset_dir(self, dataset_id: str) -> str:
        d = os.path.join(self.registry_dir, _safe_id(dataset_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _load_latest(self, dataset_id: str) -> Optional[Dict]:
        """Load the most recent schema record for the dataset."""
        d = os.path.join(self.registry_dir, _safe_id(dataset_id))
        if not os.path.isdir(d):
            return None
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        if not files:
            return None
        with open(os.path.join(d, files[-1]), encoding="utf-8") as f:
            return json.load(f)

    def _store(
        self,
        dataset_id: str,
        schema: Dict[str, str],
        version: str,
        row_count: int,
        source_uri: str,
    ) -> None:
        d = self._dataset_dir(dataset_id)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"v{version.replace('.', '_')}_{ts}.json"
        record = {
            "dataset_id": dataset_id,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "row_count": row_count,
            "source_uri": source_uri,
            "schema": schema,
            "column_count": len(schema),
        }
        with open(os.path.join(d, filename), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    @staticmethod
    def _compute_drift(
        old: Dict[str, str], new: Dict[str, str]
    ) -> List[ColumnDrift]:
        changes: List[ColumnDrift] = []
        old_cols = set(old.keys())
        new_cols = set(new.keys())

        # Missing columns (BREAKING)
        for col in old_cols - new_cols:
            changes.append(ColumnDrift(
                column=col, change_type="MISSING",
                old_dtype=old[col], new_dtype=None, severity="BREAKING",
            ))

        # New columns (ADDITIVE — minor)
        for col in new_cols - old_cols:
            changes.append(ColumnDrift(
                column=col, change_type="ADDITIVE",
                old_dtype=None, new_dtype=new[col], severity="MINOR",
            ))

        # Type changes (BREAKING)
        for col in old_cols & new_cols:
            o, n = _normalise_dtype(old[col]), _normalise_dtype(new[col])
            if o != n:
                changes.append(ColumnDrift(
                    column=col, change_type="TYPE_CHANGE",
                    old_dtype=old[col], new_dtype=new[col], severity="BREAKING",
                ))

        return changes

    @staticmethod
    def _build_summary(changes: List[ColumnDrift], old_ver: str, new_ver: str) -> str:
        if not changes:
            return f"No schema changes (v{old_ver} → v{new_ver})."
        parts = []
        missing  = [c for c in changes if c.change_type == "MISSING"]
        added    = [c for c in changes if c.change_type == "ADDITIVE"]
        type_chg = [c for c in changes if c.change_type == "TYPE_CHANGE"]
        if missing:
            parts.append(f"{len(missing)} column(s) MISSING: {[c.column for c in missing]}")
        if added:
            parts.append(f"{len(added)} column(s) ADDED: {[c.column for c in added]}")
        if type_chg:
            parts.append(
                f"{len(type_chg)} type change(s): "
                + ", ".join(f"{c.column} ({c.old_dtype}→{c.new_dtype})" for c in type_chg)
            )
        return " | ".join(parts) + f" (v{old_ver} → v{new_ver})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_id(dataset_id: str) -> str:
    """Sanitise dataset_id for use as a directory name."""
    return re.sub(r"[^\w\-]", "_", dataset_id)[:64]


def _normalise_dtype(dtype: str) -> str:
    """Normalise dtype strings for comparison (int64 ≈ int32, float64 ≈ float32)."""
    dtype = dtype.lower()
    if re.match(r"int\d*", dtype):
        return "int"
    if re.match(r"float\d*", dtype):
        return "float"
    if dtype in ("object", "str", "string"):
        return "string"
    if dtype.startswith("datetime"):
        return "datetime"
    if dtype.startswith("bool"):
        return "bool"
    return dtype
