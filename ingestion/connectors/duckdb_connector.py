"""
ingestion/connectors/duckdb_connector.py
------------------------------------------
Production DuckDB connector for DIPEX.

DuckDB is an in-process analytical SQL engine — no server required.
Ideal for local analytical workloads, Parquet/Iceberg scanning, and
columnar data processing with near-zero infrastructure overhead.

Features:
- In-memory or file-backed database
- Native Parquet / CSV / JSON glob scanning (scan_parquet, scan_csv)
- SQL queries → DataFrame via DuckDB's native pandas integration
- Chunked streaming via LIMIT/OFFSET
- Connection is lazily created and reused
- Credentials: duckdb_path from config or env DUCKDB_PATH
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.duckdb")


class DuckDBConnector(BaseConnector):
    """
    DuckDB in-process analytical SQL connector.

    Config keys:
        duckdb_path     : File path for persistent DB, or ':memory:' (default)
                          Can be set via env DUCKDB_PATH.
        query           : Default SQL query to run on extract()
        table           : Table name (used if no query given)
        parquet_glob    : Glob path for parquet scanning, e.g. 'data/*.parquet'
        chunk_size      : Rows per chunk for stream() (default: 50_000)
        read_only       : Open in read-only mode (default: False)
        threads         : Number of DuckDB threads (default: auto)
        memory_limit    : e.g. '4GB' — DuckDB memory cap
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._conn = None
        self._chunk_size: int = int(config.get("chunk_size", 50_000))

    def _get_path(self) -> str:
        return os.environ.get(
            "DUCKDB_PATH", self.config.get("duckdb_path", ":memory:")
        )

    def _get_conn(self):
        """Lazy-create DuckDB connection. Compatible with DuckDB >= 0.9."""
        if self._conn is not None:
            return self._conn
        try:
            import duckdb  # type: ignore

            path      = self._get_path()
            read_only = bool(self.config.get("read_only", False))

            # DuckDB 1.x: connect(path, config={...})
            # DuckDB <1.x: connect(database=path, read_only=bool)
            # We support both via try/except.
            try:
                connect_cfg: dict = {}
                if read_only:
                    connect_cfg["access_mode"] = "read_only"
                threads = self.config.get("threads")
                mem_lim = self.config.get("memory_limit")
                if threads:
                    connect_cfg["threads"] = int(threads)
                conn = duckdb.connect(path, config=connect_cfg)
            except TypeError:
                # Fallback for older DuckDB API
                conn = duckdb.connect(database=path, read_only=read_only)
                threads = self.config.get("threads")
                mem_lim = self.config.get("memory_limit")
                if threads:
                    conn.execute(f"SET threads TO {int(threads)}")

            if mem_lim := self.config.get("memory_limit"):
                conn.execute(f"SET memory_limit='{mem_lim}'")

            self._conn = conn
            logger.info("DuckDBConnector: connected to '%s'", path)
            return conn
        except ImportError as exc:
            raise ConnectorError("duckdb is required: pip install duckdb") from exc
        except Exception as exc:
            raise ConnectorError(f"DuckDBConnector: connect failed — {exc}") from exc


    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1").fetchone()
            logger.info("DuckDBConnector: connection test PASSED")
            return True
        except Exception as exc:
            logger.error("DuckDBConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        try:
            conn  = self._get_conn()
            table = self.config.get("table")
            parquet_glob = self.config.get("parquet_glob")

            if parquet_glob:
                df_sample = conn.execute(
                    f"SELECT * FROM read_parquet('{parquet_glob}') LIMIT 5"
                ).df()
                return {
                    "source": parquet_glob,
                    "columns": list(df_sample.columns),
                    "dtypes": {c: str(df_sample[c].dtype) for c in df_sample.columns},
                    "description": f"Parquet glob schema: {parquet_glob}",
                }

            if table:
                rows = conn.execute(f"DESCRIBE {table}").fetchall()
                col_names   = [r[0] for r in rows]
                col_types   = [r[1] for r in rows]
                dtypes      = dict(zip(col_names, col_types))
                est_count   = -1
                try:
                    est_count = conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                except Exception:
                    pass
                return {
                    "table": table,
                    "columns": col_names,
                    "dtypes": dtypes,
                    "estimated_row_count": est_count,
                    "description": f"DuckDB DESCRIBE for table '{table}'",
                }

            # Fallback: list tables
            tables = conn.execute("SHOW TABLES").fetchall()
            return {
                "tables": [t[0] for t in tables],
                "description": "All tables in DuckDB database",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """Execute SQL or scan Parquet/CSV and return DataFrame."""
        sql = query or self._build_query()
        try:
            conn = self._get_conn()
            df   = conn.execute(sql).df()
            logger.info("DuckDBConnector: extracted %d rows", len(df))
            return df
        except Exception as exc:
            raise ConnectorError(f"DuckDBConnector: extract failed — {exc}") from exc

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Yield DataFrame chunks via LIMIT/OFFSET pagination."""
        size   = chunk_size or self._chunk_size
        base   = kwargs.get("query") or self._build_query()
        # Wrap base in CTE for clean pagination
        cte_sql = f"WITH _dipex_base AS ({base}) SELECT * FROM _dipex_base"
        offset  = 0
        conn    = self._get_conn()
        while True:
            try:
                chunk = conn.execute(f"{cte_sql} LIMIT {size} OFFSET {offset}").df()
                if chunk.empty:
                    break
                yield chunk
                offset += size
                if len(chunk) < size:
                    break
            except Exception as exc:
                raise ConnectorError(f"DuckDBConnector: stream failed — {exc}") from exc

    def scan_parquet(self, glob: str, query: Optional[str] = None) -> pd.DataFrame:
        """
        Convenience helper: scan a Parquet glob with an optional SQL WHERE/LIMIT.

        Example:
            df = connector.scan_parquet("data/sales/*.parquet",
                                        query="WHERE year = 2024")
        """
        sql = f"SELECT * FROM read_parquet('{glob}') {query or ''}"
        return self.extract(sql)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_query(self) -> str:
        parquet_glob = self.config.get("parquet_glob")
        if parquet_glob:
            return f"SELECT * FROM read_parquet('{parquet_glob}')"

        table = self.config.get("table")
        query = self.config.get("query")
        if query:
            return query
        if table:
            return f"SELECT * FROM {table}"
        raise ConnectorError(
            "DuckDBConnector: one of 'table', 'query', or 'parquet_glob' must be set in config"
        )
