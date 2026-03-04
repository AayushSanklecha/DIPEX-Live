"""
query_engine/query_registry.py
------------------------------
Named SQL query store with versioning and parameterization.

Stores queries in a JSON file:
  {
    "query_name": {
      "sql": "SELECT ...",
      "description": "...",
      "version": 1,
      "created_at": "...",
      "params": ["param_name"]
    }
  }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dipex.query_engine.query_registry")


class QueryRegistry:
    """
    Persistent named query store.

    Usage::

        registry = QueryRegistry("data/named_queries.json")
        registry.save("daily_totals", "SELECT date, SUM(amount) FROM df GROUP BY date", "Daily sales totals")
        sql = registry.get("daily_totals")
    """

    def __init__(self, registry_path: str = "data/named_queries.json") -> None:
        self._path = registry_path
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
                logger.info("Loaded %d named queries from %s.", len(self._store), self._path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load query registry: %s", exc)
                self._store = {}
        else:
            self._store = {}

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)

    def save(
        self,
        name: str,
        sql: str,
        description: str = "",
        params: Optional[List[str]] = None,
    ) -> None:
        """Save or update a named query."""
        existing = self._store.get(name, {})
        version = existing.get("version", 0) + 1
        self._store[name] = {
            "sql": sql.strip(),
            "description": description,
            "version": version,
            "params": params or [],
            "created_at": existing.get("created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._persist()
        logger.info("Saved named query '%s' (version %d).", name, version)

    def get(self, name: str) -> Optional[str]:
        """Return SQL string for a named query, or None."""
        entry = self._store.get(name)
        return entry["sql"] if entry else None

    def get_entry(self, name: str) -> Optional[Dict[str, Any]]:
        """Return full metadata entry."""
        return self._store.get(name)

    def delete(self, name: str) -> bool:
        """Remove a named query."""
        if name in self._store:
            del self._store[name]
            self._persist()
            return True
        return False

    def list(self) -> List[Dict[str, Any]]:
        """List all named queries with metadata."""
        return [
            {"name": n, **{k: v for k, v in entry.items() if k != "sql"}}
            for n, entry in self._store.items()
        ]

    def search(self, keyword: str) -> List[str]:
        """Search query names and descriptions."""
        kw = keyword.lower()
        return [
            name for name, entry in self._store.items()
            if kw in name.lower() or kw in entry.get("description", "").lower()
        ]

    def __len__(self) -> int:
        return len(self._store)
