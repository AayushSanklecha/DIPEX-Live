"""
ingestion/snapshot.py
---------------------
Manages immutable, content-addressed snapshots of datasets.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages immutable snapshots and fingerprints of datasets."""

    def __init__(self, snapshot_dir: str = "data/snapshots") -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.snapshot_dir / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    self.registry: dict = json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning(
                    "Registry file is corrupted or unreadable (%s). Starting fresh.", exc
                )
                self.registry = {}
        else:
            self.registry = {}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def calculate_hash(self, df: pd.DataFrame) -> str:
        """
        Computes a stable SHA-256 fingerprint of the dataframe content.

        Columns are sorted alphabetically before hashing to ensure that
        column-reordering does not produce a different fingerprint for the
        same logical dataset.
        """
        sorted_df = df.reindex(sorted(df.columns), axis=1)
        raw_bytes = pd.util.hash_pandas_object(sorted_df, index=True).values.tobytes()
        return hashlib.sha256(raw_bytes).hexdigest()

    def create_snapshot(self, df: pd.DataFrame, source_name: str) -> str:
        """Saves a snapshot of the dataframe and returns its content hash."""
        fingerprint = self.calculate_hash(df)
        timestamp = datetime.now(timezone.utc).isoformat()

        snapshot_filename = f"{source_name}_{fingerprint}.parquet"
        snapshot_path = self.snapshot_dir / snapshot_filename

        if not snapshot_path.exists():
            df.to_parquet(snapshot_path, engine="pyarrow")
            logger.info("Snapshot saved: %s", snapshot_path)
        else:
            logger.debug("Snapshot already exists (fingerprint=%s), skipping write.", fingerprint)

        self.registry[fingerprint] = {
            "source": source_name,
            "path": str(snapshot_path),
            "timestamp": timestamp,
            "fingerprint": fingerprint,
            "rows": len(df),
            "cols": len(df.columns),
        }
        self._save_registry()
        return fingerprint

    def get_snapshot(self, fingerprint: str) -> pd.DataFrame:
        """Retrieves a snapshot by its hash fingerprint."""
        if fingerprint not in self.registry:
            raise KeyError(f"Snapshot with fingerprint '{fingerprint}' not found.")
        path = Path(self.registry[fingerprint]["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Snapshot file missing on disk: {path}. Registry may be stale."
            )
        return pd.read_parquet(path)
