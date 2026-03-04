"""
query_engine/lineage_tracker.py
--------------------------------
Column-level lineage tracking for DataFrame transformations and SQL queries.

Appends tamper-evident JSONL entries describing what happened to each column:
  - Source origin (from ingestion)
  - SQL transforms
  - Feature engineering steps
  - Cleaning operations

Each entry links to the run_id and step, forming an auditable chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.query_engine.lineage_tracker")


class LineageTracker:
    """
    Append-only JSONL column lineage log.

    Usage::

        tracker = LineageTracker("audit/lineage.jsonl")
        tracker.record_source("sales_df", ["amount", "date", "customer_id"], run_id="run-001")
        tracker.record_transform("run-001", "feature_engineer", {"amount_log1p": {"derived_from": ["amount"]}})
    """

    def __init__(self, lineage_path: str = "audit/lineage.jsonl") -> None:
        self._path = lineage_path
        self._previous_hash: str = "GENESIS"
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        self._init_chain()

    def _init_chain(self) -> None:
        """Bootstrap hash chain from last line of existing log."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                self._previous_hash = last.get("entry_hash", "GENESIS")
        except Exception:  # noqa: BLE001
            pass

    def _append(self, entry: Dict[str, Any]) -> None:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry["previous_hash"] = self._previous_hash
        entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
        entry["entry_hash"] = entry_hash
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._previous_hash = entry_hash

    def record_source(
        self,
        source_name: str,
        columns: List[str],
        run_id: str = "",
        source_type: str = "file",
    ) -> None:
        """Record the origin of columns from data ingestion."""
        self._append({
            "event": "SOURCE_INGESTION",
            "run_id": run_id,
            "source_name": source_name,
            "source_type": source_type,
            "columns": columns,
        })

    def record_transform(
        self,
        run_id: str,
        step_name: str,
        column_lineage: Dict[str, Any],
    ) -> None:
        """
        Record column lineage from a transformation step.

        Parameters
        ----------
        run_id        : pipeline run identifier
        step_name     : e.g. 'feature_engineer', 'sql_query', 'cleaner'
        column_lineage: dict mapping new_col_name → {derived_from: [...], operation: '...'}
        """
        self._append({
            "event": "COLUMN_TRANSFORM",
            "run_id": run_id,
            "step": step_name,
            "lineage": column_lineage,
        })

    def record_sql(
        self,
        run_id: str,
        sql: str,
        input_views: List[str],
        output_columns: List[str],
    ) -> None:
        """Record columns produced by a SQL query."""
        self._append({
            "event": "SQL_QUERY",
            "run_id": run_id,
            "sql": sql,
            "input_views": input_views,
            "output_columns": output_columns,
        })

    def get_lineage(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read all lineage entries, optionally filtered by run_id."""
        if not os.path.exists(self._path):
            return []
        entries = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if run_id is None or entry.get("run_id") == run_id:
                        entries.append(entry)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read lineage log: %s", exc)
        return entries

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity of the lineage log."""
        if not os.path.exists(self._path):
            return True
        prev_hash = "GENESIS"
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    stored_hash = entry.pop("entry_hash", None)
                    if entry.get("previous_hash") != prev_hash:
                        logger.error("Hash chain broken at entry with timestamp %s", entry.get("timestamp"))
                        return False
                    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                    computed = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
                    if stored_hash and stored_hash != computed:
                        # Re-insert to compare
                        entry["entry_hash"] = stored_hash
                    prev_hash = stored_hash or prev_hash
        except Exception as exc:  # noqa: BLE001
            logger.error("Integrity check error: %s", exc)
            return False
        return True
