"""
datasource/router.py
---------------------
DATA SOURCE LAYER — Unified DataSourceRouter

Single entry point that resolves source type strings to the appropriate
reader or connector from ingestion/readers/ and ingestion/connectors/.
Eliminates the need for callers to know which specific reader class to import.

Supported source types
-----------------------
  file sources  : "csv", "excel", "parquet", "json", "tsv"
  db sources    : "sql", "postgres", "mysql", "sqlite",
                  "mongo", "mongodb",
                  "duckdb", "neo4j", "redis",
                  "clickhouse", "elasticsearch"
  api source    : "api", "rest", "http"
  kafka source  : "kafka", "stream"
  fallback      : "auto" — tries file → api → db in order
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("dipex.datasource.router")

# ── Source type registry ──────────────────────────────────────────────────────

_FILE_TYPES   = {"csv", "excel", "xlsx", "xls", "parquet", "json", "tsv", "feather"}
_DB_TYPES     = {"sql", "postgres", "postgresql", "mysql", "sqlite",
                 "mongo", "mongodb", "duckdb", "neo4j", "redis",
                 "clickhouse", "elasticsearch"}
_API_TYPES    = {"api", "rest", "http", "https", "graphql"}
_STREAM_TYPES = {"kafka", "stream", "streaming"}


class DataSourceRouter:
    """
    Routes a source specification to the correct DIPEX reader/connector.

    Usage::

        router = DataSourceRouter(config)

        # Route by explicit type
        df = router.load(source="s3://bucket/data.csv", source_type="csv")

        # Auto-detect from path/URL
        df = router.load(source="postgres://host/db", source_type="auto")

        # Get the raw reader object for advanced use
        reader = router.get_reader("kafka")
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        source: str,
        source_type: str = "auto",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from `source` using the resolved reader strategy.

        Parameters
        ----------
        source      : path, URL, or connection string
        source_type : explicit type or "auto" for heuristic detection
        **kwargs    : forwarded to the underlying reader

        Returns
        -------
        pd.DataFrame
        """
        resolved = self._resolve_type(source, source_type)
        logger.info("DataSourceRouter: source_type=%s resolved=%s path=%s",
                    source_type, resolved, source)

        if resolved in _FILE_TYPES:
            return self._load_file(source, resolved, **kwargs)
        elif resolved in _DB_TYPES:
            return self._load_db(source, resolved, **kwargs)
        elif resolved in _API_TYPES:
            return self._load_api(source, **kwargs)
        elif resolved in _STREAM_TYPES:
            return self._load_stream(source, **kwargs)
        else:
            logger.warning("Unknown source type '%s' — attempting universal fallback", resolved)
            return self._load_fallback(source, **kwargs)

    def get_reader(self, source_type: str) -> Any:
        """
        Return the underlying reader/connector object for a given source type.
        Useful when callers need low-level access (e.g. for streaming).
        """
        resolved = self._resolve_type("", source_type)
        if resolved in _FILE_TYPES:
            from ingestion.readers.file_reader import FileReader
            return FileReader(self.config)
        elif resolved in _API_TYPES:
            from ingestion.readers.api_reader import APIReader
            return APIReader(self.config)
        elif resolved in _DB_TYPES:
            return self._get_db_connector(resolved)
        elif resolved in _STREAM_TYPES:
            from ingestion.readers.stream_reader import StreamReader
            return StreamReader(self.config)
        else:
            from ingestion.readers.universal_fallback import UniversalFallbackReader
            return UniversalFallbackReader(self.config)

    def supported_sources(self) -> Dict[str, List[str]]:
        """Return a dict of source category → list of supported type strings."""
        return {
            "file":   sorted(_FILE_TYPES),
            "database": sorted(_DB_TYPES),
            "api":    sorted(_API_TYPES),
            "stream": sorted(_STREAM_TYPES),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_type(self, source: str, source_type: str) -> str:
        """Resolve 'auto' or normalise explicit type to canonical form."""
        if source_type and source_type.lower() != "auto":
            return source_type.lower()

        # Heuristic detection from source path / URL
        src = source.lower()
        if any(src.endswith(f".{ext}") for ext in {"csv", "tsv", "xlsx", "xls", "parquet", "json", "feather"}):
            ext = src.rsplit(".", 1)[-1]
            return ext if ext in _FILE_TYPES else "csv"
        if src.startswith(("postgres", "mysql", "sqlite", "mongodb", "mongodb+srv")):
            return "db"
        if src.startswith(("http://", "https://", "graphql")):
            return "api"
        if "kafka" in src or src.startswith("kafka://"):
            return "kafka"
        # Default: try file reader
        return "csv"

    def _load_file(self, source: str, file_type: str, **kwargs) -> pd.DataFrame:
        try:
            from ingestion.readers.file_reader import FileReader
            reader = FileReader(self.config)
            return reader.read(source, file_type=file_type, **kwargs)
        except Exception as exc:
            logger.error("FileReader failed for '%s': %s", source, exc)
            raise

    def _load_db(self, source: str, db_type: str, **kwargs) -> pd.DataFrame:
        try:
            connector = self._get_db_connector(db_type)
            # Most connectors expose .query() or .read_table()
            if hasattr(connector, "query"):
                return connector.query(kwargs.pop("query", "SELECT 1"), **kwargs)
            elif hasattr(connector, "read_table"):
                return connector.read_table(kwargs.pop("table", "data"), **kwargs)
            else:
                from ingestion.readers.db_reader import DBReader
                return DBReader(self.config).read(source, **kwargs)
        except Exception as exc:
            logger.error("DB connector failed for '%s': %s", source, exc)
            raise

    def _load_api(self, source: str, **kwargs) -> pd.DataFrame:
        try:
            from ingestion.readers.api_reader import APIReader
            reader = APIReader(self.config)
            return reader.read(source, **kwargs)
        except Exception as exc:
            logger.error("APIReader failed for '%s': %s", source, exc)
            raise

    def _load_stream(self, source: str, **kwargs) -> pd.DataFrame:
        try:
            from ingestion.readers.stream_reader import StreamReader
            reader = StreamReader(self.config)
            return reader.read(source, **kwargs)
        except Exception as exc:
            logger.error("StreamReader failed for '%s': %s", source, exc)
            raise

    def _load_fallback(self, source: str, **kwargs) -> pd.DataFrame:
        from ingestion.readers.universal_fallback import UniversalFallbackReader
        return UniversalFallbackReader(self.config).read(source, **kwargs)

    def _get_db_connector(self, db_type: str) -> Any:
        """Return the correct DB connector object for db_type."""
        mapping = {
            "mongo": "ingestion.connectors.mongo_connector.MongoConnector",
            "mongodb": "ingestion.connectors.mongo_connector.MongoConnector",
            "duckdb": "ingestion.connectors.duckdb_connector.DuckDBConnector",
            "neo4j": "ingestion.connectors.neo4j_connector.Neo4jConnector",
            "redis": "ingestion.connectors.redis_connector.RedisConnector",
            "clickhouse": "ingestion.connectors.clickhouse_connector.ClickhouseConnector",
            "elasticsearch": "ingestion.connectors.elasticsearch_connector.ElasticsearchConnector",
        }
        sql_types = {"sql", "postgres", "postgresql", "mysql", "sqlite"}
        if db_type in sql_types:
            from ingestion.connectors.sql_connector import SQLConnector
            return SQLConnector(self.config)

        fqcn = mapping.get(db_type)
        if fqcn:
            module_path, class_name = fqcn.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(self.config)

        from ingestion.readers.db_reader import DBReader
        return DBReader(self.config)
