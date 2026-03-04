"""
governance/data_catalog.py
---------------------------
Column-level metadata store for the data catalog.

Tracks:
  - Column classification: PII | SENSITIVE | INTERNAL | PUBLIC
  - Data type, business description, owner, DQ score
  - Allowed operations (allowed_in_output, allowed_in_training, etc.)

Backed by a JSON file, with CRUD API.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.governance.data_catalog")

VALID_CLASSIFICATIONS = {"PII", "SENSITIVE", "INTERNAL", "PUBLIC"}


class DataCatalog:
    """
    Column-level metadata catalog.

    Usage::

        catalog = DataCatalog("governance/data_catalog.json")
        catalog.register("customer_id", classification="PII", description="Unique customer identifier")
        pii = catalog.get_pii_columns()
    """

    def __init__(self, catalog_path: str = "governance/data_catalog.json") -> None:
        self._path = catalog_path
        self._store: Dict[str, Dict[str, Any]] = {}
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except Exception:  # noqa: BLE001
                self._store = {}

    def _persist(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)

    def register(
        self,
        column_name: str,
        classification: str = "INTERNAL",
        description: str = "",
        data_type: str = "",
        owner: str = "",
        allowed_in_output: bool = True,
        allowed_in_training: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register or update a column's metadata entry."""
        if classification not in VALID_CLASSIFICATIONS:
            raise ValueError(f"Invalid classification '{classification}'. Choose from {VALID_CLASSIFICATIONS}")

        entry = self._store.get(column_name, {})
        entry.update({
            "column_name": column_name,
            "classification": classification,
            "description": description,
            "data_type": data_type,
            "owner": owner,
            "allowed_in_output": allowed_in_output,
            "allowed_in_training": allowed_in_training,
            "tags": tags or [],
            "registered_at": entry.get("registered_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self._store[column_name] = entry
        self._persist()
        return entry

    def get(self, column_name: str) -> Optional[Dict[str, Any]]:
        return self._store.get(column_name)

    def delete(self, column_name: str) -> bool:
        if column_name in self._store:
            del self._store[column_name]
            self._persist()
            return True
        return False

    def get_pii_columns(self) -> List[str]:
        return [col for col, meta in self._store.items() if meta.get("classification") == "PII"]

    def get_sensitive_columns(self) -> List[str]:
        return [col for col, meta in self._store.items()
                if meta.get("classification") in ("PII", "SENSITIVE")]

    def get_blocked_output_columns(self) -> List[str]:
        return [col for col, meta in self._store.items()
                if not meta.get("allowed_in_output", True)]

    def list(self, classification: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = list(self._store.values())
        if classification:
            entries = [e for e in entries if e.get("classification") == classification]
        return entries

    def auto_detect_pii(self, columns: List[str]) -> List[str]:
        """Heuristic PII detection based on column name patterns."""
        import re
        pii_patterns = [
            r"(name|email|phone|ssn|passport|dob|birth|national_id|pan|aadhar|"
            r"credit_card|account_no|address|zip|postal|ip_addr|device_id|user_id|"
            r"customer_id|patient_id|member_id|license|lat(itude)?|lon(gitude)?)"
        ]
        pattern = re.compile("|".join(pii_patterns), re.IGNORECASE)
        detected = [col for col in columns if pattern.search(col)]
        return detected

    def bulk_register_from_df(self, df: Any, default_classification: str = "INTERNAL") -> List[str]:
        """Auto-register all columns from a DataFrame."""
        registered = []
        pii_heuristic = self.auto_detect_pii(list(df.columns))
        for col in df.columns:
            classification = "PII" if col in pii_heuristic else default_classification
            data_type = str(df[col].dtype)
            self.register(
                column_name=col,
                classification=classification,
                data_type=data_type,
                description=f"Auto-registered from dataset ({data_type})",
            )
            registered.append(col)
        logger.info("Bulk registered %d columns (%d detected as PII).", len(registered), len(pii_heuristic))
        return registered
