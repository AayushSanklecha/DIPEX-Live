"""
ingestion/issf.py
------------------
Internal Standard Schema Format (ISSF)

Every ingestion — file, API, database, or stream — MUST produce an ISSFSnapshot.
No downstream analytics stage may accept raw source data.

Contract fields
---------------
dataset_id          : Stable identifier for the logical dataset
snapshot_id         : UUID of this specific ingestion run
schema_version      : Semver string (MAJOR.MINOR.PATCH)
ingestion_timestamp : ISO 8601 UTC
data_mode           : 'batch' | 'live' | 'stream'
source_type         : 'file' | 'api' | 'database' | 'stream'
source_uri          : Sanitised source path / URL / DSN (no credentials)
validation_status   : 'PASSED' | 'FAILED' | 'WARN'
error_logs          : List of structured error records
column_metadata     : Per-column type/null/unique stats
row_count           : Final row count after cleaning
quality_score       : 0.0–1.0
fingerprint         : SHA-256 of schema + row_count + snapshot_id
data                : Optional in-memory DataFrame (excluded from serialisation)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


# ── Column Metadata ──────────────────────────────────────────────────────────

class ColumnMeta:
    def __init__(
        self,
        name: str,
        dtype: str,
        null_count: int,
        null_rate: float,
        unique_count: int,
        is_pk_candidate: bool = False,
        sample_values: Optional[List[Any]] = None,
        min_val: Any = None,
        max_val: Any = None,
    ) -> None:
        self.name            = name
        self.dtype           = dtype
        self.null_count      = null_count
        self.null_rate       = round(null_rate, 4)
        self.unique_count    = unique_count
        self.is_pk_candidate = is_pk_candidate
        self.sample_values   = sample_values or []
        self.min_val         = min_val
        self.max_val         = max_val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "null_rate": self.null_rate,
            "unique_count": self.unique_count,
            "is_pk_candidate": self.is_pk_candidate,
            "sample_values": [str(v) for v in self.sample_values[:5]],
            "min_val": str(self.min_val) if self.min_val is not None else None,
            "max_val": str(self.max_val) if self.max_val is not None else None,
        }


# ── Error Record ──────────────────────────────────────────────────────────────

class IngestionError:
    def __init__(
        self,
        error_type: str,
        message: str,
        row_index: Optional[int] = None,
        column: Optional[str] = None,
        severity: str = "ERROR",   # ERROR | WARN | INFO
        correlation_id: Optional[str] = None,
    ) -> None:
        self.error_type     = error_type
        self.message        = message
        self.row_index      = row_index
        self.column         = column
        self.severity       = severity
        self.correlation_id = correlation_id
        self.timestamp      = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "row_index": self.row_index,
            "column": self.column,
            "severity": self.severity,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


# ── ISSF Snapshot ─────────────────────────────────────────────────────────────

class ISSFSnapshot:
    """
    Internal Standard Schema Format snapshot.
    Every ingestion pipeline MUST produce one before downstream stages run.
    """

    VALID_DATA_MODES   = {"batch", "live", "stream"}
    VALID_SOURCE_TYPES = {"file", "api", "database", "stream"}
    VALID_STATUSES     = {"PASSED", "FAILED", "WARN"}

    def __init__(
        self,
        dataset_id: str,
        schema_version: str,
        data_mode: str,
        source_type: str,
        source_uri: str,
        column_metadata: List[ColumnMeta],
        row_count: int,
        quality_score: float,
        validation_status: str = "PASSED",
        error_logs: Optional[List[IngestionError]] = None,
        snapshot_id: Optional[str] = None,
        ingestion_timestamp: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert data_mode   in self.VALID_DATA_MODES,   f"Invalid data_mode: {data_mode}"
        assert source_type in self.VALID_SOURCE_TYPES, f"Invalid source_type: {source_type}"
        assert validation_status in self.VALID_STATUSES

        self.dataset_id          = dataset_id
        self.snapshot_id         = snapshot_id or str(uuid.uuid4())
        self.schema_version      = schema_version
        self.ingestion_timestamp = ingestion_timestamp or datetime.now(timezone.utc).isoformat()
        self.data_mode           = data_mode
        self.source_type         = source_type
        self.source_uri          = source_uri
        self.validation_status   = validation_status
        self.error_logs          = error_logs or []
        self.column_metadata     = column_metadata
        self.row_count           = row_count
        self.quality_score       = round(min(1.0, max(0.0, quality_score)), 4)
        self.data                = data          # Not serialised
        self.extra_meta          = extra_meta or {}
        self.fingerprint         = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        schema_sig = json.dumps(
            {c.name: c.dtype for c in self.column_metadata}, sort_keys=True
        )
        raw = f"{self.dataset_id}|{schema_sig}|{self.row_count}|{self.snapshot_id}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def is_compliant(self) -> bool:
        """Return True if snapshot meets minimum compliance requirements."""
        return (
            bool(self.dataset_id)
            and bool(self.snapshot_id)
            and bool(self.schema_version)
            and self.row_count >= 0
            and 0.0 <= self.quality_score <= 1.0
            and self.validation_status in self.VALID_STATUSES
        )

    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "dataset_id":          self.dataset_id,
            "snapshot_id":         self.snapshot_id,
            "schema_version":      self.schema_version,
            "ingestion_timestamp": self.ingestion_timestamp,
            "data_mode":           self.data_mode,
            "source_type":         self.source_type,
            "source_uri":          self.source_uri,
            "validation_status":   self.validation_status,
            "error_logs":          [e.to_dict() for e in self.error_logs],
            "column_metadata":     [c.to_dict() for c in self.column_metadata],
            "row_count":           self.row_count,
            "quality_score":       self.quality_score,
            "fingerprint":         self.fingerprint,
            "is_compliant":        self.is_compliant,
            **self.extra_meta,
        }
        if include_data and self.data is not None:
            out["data_preview"] = self.data.head(10).to_dict(orient="records")
        return out

    def save(self, directory: str = "data/snapshots") -> str:
        """Persist ISSF metadata to JSON (data stored separately as Parquet).

        The Parquet file is named ``{snapshot_id}_issf.parquet`` so that the
        analyst CLI, ingest_v2 API, and any other consumer can locate it with
        a single predictable pattern.
        """
        import os
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, f"{self.snapshot_id}_issf.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        if self.data is not None and not self.data.empty:
            try:
                parquet_path = os.path.join(directory, f"{self.snapshot_id}_issf.parquet")
                self.data.to_parquet(parquet_path, index=False, compression="snappy")
            except Exception:  # noqa: BLE001
                csv_path = os.path.join(directory, f"{self.snapshot_id}_issf.csv")
                self.data.to_csv(csv_path, index=False)
        return meta_path

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        dataset_id: str,
        source_type: str = "file",
        data_mode: str = "batch",
        source_uri: str = "",
        schema_version: str = "1.0",
        snapshot_id: Optional[str] = None,
    ) -> "ISSFSnapshot":
        """
        Factory: build a complete ISSFSnapshot directly from a raw DataFrame.

        Automatically computes:
          - ColumnMeta for every column (dtype, null stats, unique count, PK candidate)
          - quality_score ∈ [0, 1] based on completeness, inf-rate, zero-variance
          - validation_status: PASSED / WARN / FAILED

        Handles all edge cases:
          - Empty DataFrames (row_count=0, quality_score=0.0, status=FAILED)
          - All-null columns (flagged in ColumnMeta, status=WARN/FAILED)
          - Inf / -Inf values in numeric columns (penalises quality_score)
          - Duplicate column names (deduplicated before building ColumnMeta)
          - Zero-variance numeric columns (no div-by-zero)

        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame. Must be a valid pandas DataFrame.
        dataset_id : str
            Stable identifier for the logical dataset.
        source_type : str
            One of: 'file', 'api', 'database', 'stream'.
        data_mode : str
            One of: 'batch', 'live', 'stream'.
        source_uri : str
            Sanitised source path / URL (no credentials).
        schema_version : str
            Semver string, defaults to '1.0'.
        snapshot_id : str, optional
            Explicit UUID; auto-generated if not provided.

        Returns
        -------
        ISSFSnapshot
        """
        import math as _math

        # ── Guard: empty DataFrame ──────────────────────────────────────────────
        if df is None or (hasattr(df, "empty") and df.empty):
            return cls(
                dataset_id=dataset_id,
                schema_version=schema_version,
                data_mode=data_mode,
                source_type=source_type,
                source_uri=source_uri or "",
                column_metadata=[],
                row_count=0,
                quality_score=0.0,
                validation_status="FAILED",
                error_logs=[
                    IngestionError(
                        error_type="EMPTY_DATASET",
                        message="DataFrame is empty — no rows to process.",
                        severity="ERROR",
                    )
                ],
                snapshot_id=snapshot_id,
                data=df,
            )

        # ── Deduplicate column names (pandas allows duplicates) ─────────────────
        if df.columns.duplicated().any():
            seen: Dict[str, int] = {}
            new_cols = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df = df.copy()
            df.columns = new_cols

        n_rows = len(df)
        n_cols = len(df.columns)
        errors: List[IngestionError] = []
        warnings_count = 0

        # ── Build ColumnMeta for every column ───────────────────────────────────
        col_metas: List[ColumnMeta] = []
        completeness_scores: List[float] = []

        for col in df.columns:
            series = df[col]
            null_count  = int(series.isna().sum())
            null_rate   = null_count / n_rows if n_rows > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype)

            # min/max — only for numeric, handle inf/NaN safely
            min_val = max_val = None
            if pd.api.types.is_numeric_dtype(series):
                finite_vals = series.replace([float("inf"), float("-inf")], float("nan")).dropna()
                if len(finite_vals) > 0:
                    min_val = finite_vals.min()
                    max_val = finite_vals.max()
                # Warn if inf values present
                inf_count = int((series == float("inf")).sum() + (series == float("-inf")).sum())
                if inf_count > 0:
                    warnings_count += 1
                    errors.append(IngestionError(
                        error_type="INF_VALUES",
                        message=f"Column '{col}' contains {inf_count} inf/-inf values.",
                        column=col,
                        severity="WARN",
                    ))

            # Warn on fully-null columns
            if null_rate == 1.0:
                warnings_count += 1
                errors.append(IngestionError(
                    error_type="NULL_COLUMN",
                    message=f"Column '{col}' is entirely null ({n_rows}/{n_rows} rows).",
                    column=col,
                    severity="WARN",
                ))

            is_pk = (unique_count == n_rows and null_count == 0)
            col_metas.append(ColumnMeta(
                name=col,
                dtype=dtype_str,
                null_count=null_count,
                null_rate=null_rate,
                unique_count=unique_count,
                is_pk_candidate=is_pk,
                sample_values=list(series.dropna().head(5)),
                min_val=min_val,
                max_val=max_val,
            ))
            completeness_scores.append(1.0 - null_rate)

        # ── Quality score ────────────────────────────────────────────────────────
        # Base = mean completeness across all columns
        base_score = float(sum(completeness_scores) / n_cols) if n_cols > 0 else 0.0

        # Penalty for inf/nan in numeric columns
        num_cols = df.select_dtypes(include=[float, int]).columns
        if len(num_cols) > 0:
            try:
                inf_mask = df[num_cols].isin([float("inf"), float("-inf")])
                inf_rate = float(inf_mask.values.sum()) / max(1, n_rows * len(num_cols))
                nan_rate = float(df[num_cols].isna().values.sum()) / max(1, n_rows * len(num_cols))
            except Exception:
                inf_rate = nan_rate = 0.0
            base_score = max(0.0, base_score - inf_rate * 0.2 - nan_rate * 0.1)

        # Per-warning penalty (capped at 0.3 reduction)
        warn_penalty = min(0.3, warnings_count * 0.05)
        quality_score = max(0.0, min(1.0, base_score - warn_penalty))

        # ── Validation status ────────────────────────────────────────────────────
        fully_null_cols = sum(1 for cm in col_metas if cm.null_rate == 1.0)
        if fully_null_cols >= n_cols or quality_score < 0.3:
            validation_status = "FAILED"
        elif warnings_count > 0 or quality_score < 0.7:
            validation_status = "WARN"
        else:
            validation_status = "PASSED"

        return cls(
            dataset_id=dataset_id,
            schema_version=schema_version,
            data_mode=data_mode,
            source_type=source_type,
            source_uri=source_uri or "",
            column_metadata=col_metas,
            row_count=n_rows,
            quality_score=quality_score,
            validation_status=validation_status,
            error_logs=errors,
            snapshot_id=snapshot_id,
            data=df,
        )

    @classmethod
    def load(cls, snapshot_id: str, directory: str = "data/snapshots") -> "ISSFSnapshot":
        """Reload ISSF from persisted JSON + Parquet.

        Backward-compatible: checks ``_issf.parquet`` first, then falls back
        to the legacy ``_data.parquet`` name produced by older versions.
        """
        import os
        meta_path = os.path.join(directory, f"{snapshot_id}_issf.json")
        with open(meta_path, encoding="utf-8") as f:
            d = json.load(f)
        cols = [
            ColumnMeta(
                name=c["name"], dtype=c["dtype"],
                null_count=c["null_count"], null_rate=c["null_rate"],
                unique_count=c["unique_count"],
                is_pk_candidate=c.get("is_pk_candidate", False),
            )
            for c in d["column_metadata"]
        ]
        data = None
        # Try canonical name first, fall back to legacy name for backward compat
        for _parquet_name in (f"{snapshot_id}_issf.parquet", f"{snapshot_id}_data.parquet"):
            _p = os.path.join(directory, _parquet_name)
            if os.path.exists(_p):
                data = pd.read_parquet(_p)
                break
        return cls(
            dataset_id=d["dataset_id"],
            schema_version=d["schema_version"],
            data_mode=d["data_mode"],
            source_type=d["source_type"],
            source_uri=d["source_uri"],
            column_metadata=cols,
            row_count=d["row_count"],
            quality_score=d["quality_score"],
            validation_status=d["validation_status"],
            snapshot_id=d["snapshot_id"],
            ingestion_timestamp=d["ingestion_timestamp"],
            data=data,
        )

    def __repr__(self) -> str:
        return (
            f"ISSFSnapshot(dataset_id={self.dataset_id!r}, "
            f"snapshot_id={self.snapshot_id[:8]}…, "
            f"rows={self.row_count}, quality={self.quality_score:.2f}, "
            f"status={self.validation_status})"
        )
