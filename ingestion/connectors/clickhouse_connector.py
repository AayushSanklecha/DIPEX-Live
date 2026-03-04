"""
ingestion/connectors/clickhouse_connector.py
----------------------------------------------
Production ClickHouse columnar database connector for DIPEX.

Uses clickhouse-connect (HTTP interface) — no C driver dependency.
ClickHouse is purpose-built for OLAP workloads: petabyte-scale columnar
storage with sub-second query latency on billions of rows.

Features:
- HTTP-based connection via clickhouse-connect (pip install clickhouse-connect)
- Native DataFrame output via client.query_df()
- Chunked streaming via LIMIT/OFFSET
- DESCRIBE TABLE schema extraction
- Full credential isolation via env vars
- Connection pool via clickhouse-connect's built-in session management
- SASL/TLS support for production ClickHouse Cloud
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.clickhouse")

_DEFAULT_PORT: int = 8123
_DEFAULT_CHUNK: int = 100_000


class ClickHouseConnector(BaseConnector):
    """
    ClickHouse columnar OLAP connector using clickhouse-connect.

    Config keys:
        host        : ClickHouse host (env: CH_HOST, default: localhost)
        port        : HTTP port (env: CH_PORT, default: 8123)
        database    : Database name (env: CH_DB, default: default)
        username    : Username (env: CH_USER, default: default)
        password    : Password (env: CH_PASS, default: '')
        table       : Default table for extract/stream
        query       : Custom SQL query (overrides table)
        secure      : Use HTTPS (default: False)
        verify      : Verify TLS cert (default: True)
        chunk_size  : Rows per chunk for stream() (default: 100_000)
        connect_timeout  : Seconds before connect fails (default: 10)
        query_timeout    : Query timeout in seconds (default: 300)
        compression : 'lz4' | 'zstd' | None (default: 'lz4')
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._client = None
        self._chunk_size: int = int(config.get("chunk_size", _DEFAULT_CHUNK))

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _env(self, env_key: str, cfg_key: str, default: str = "") -> str:
        return os.environ.get(env_key, self.config.get(cfg_key, default))

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import clickhouse_connect  # type: ignore

            host     = self._env("CH_HOST", "host", "localhost")
            port     = int(self._env("CH_PORT", "port", str(_DEFAULT_PORT)))
            database = self._env("CH_DB",   "database", "default")
            username = self._env("CH_USER", "username", "default")
            password = self._env("CH_PASS", "password", "")
            secure   = bool(self.config.get("secure", False))
            verify   = bool(self.config.get("verify", True))
            connect_timeout = int(self.config.get("connect_timeout", 10))
            query_timeout   = int(self.config.get("query_timeout",   300))
            compression     = self.config.get("compression", "lz4")

            self._client = clickhouse_connect.get_client(
                host=host,
                port=port,
                database=database,
                username=username,
                password=password,
                secure=secure,
                verify=verify,
                connect_timeout=connect_timeout,
                query_timeout=query_timeout,
                compress=compression,
            )
            logger.info(
                "ClickHouseConnector: connected to %s:%d/%s", host, port, database
            )
            return self._client
        except ImportError as exc:
            raise ConnectorError(
                "clickhouse-connect is required: pip install clickhouse-connect"
            ) from exc
        except Exception as exc:
            raise ConnectorError(
                f"ClickHouseConnector: connect failed — {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        try:
            client = self._get_client()
            client.ping()
            logger.info("ClickHouseConnector: connection test PASSED")
            return True
        except Exception as exc:
            logger.error("ClickHouseConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            table  = self.config.get("table")
            if not table:
                # List tables in current database
                result = client.query("SHOW TABLES")
                tables = [row[0] for row in result.result_rows]
                return {"tables": tables, "description": "All tables in ClickHouse database"}

            result = client.query(f"DESCRIBE TABLE {table}")
            columns, types = [], {}
            for row in result.result_rows:
                col_name = row[0]
                col_type = row[1]
                columns.append(col_name)
                types[col_name] = col_type

            # Estimated row count via system.tables
            est_count = -1
            try:
                db   = self._env("CH_DB", "database", "default")
                row  = client.query(
                    f"SELECT total_rows FROM system.tables "
                    f"WHERE database='{db}' AND name='{table}'"
                ).result_rows
                est_count = int(row[0][0]) if row else -1
            except Exception:
                pass

            return {
                "table": table,
                "columns": columns,
                "dtypes": types,
                "estimated_row_count": est_count,
                "description": f"ClickHouse DESCRIBE TABLE {table}",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """Run SQL query → DataFrame using clickhouse-connect native DataFrame output."""
        sql = query or self._build_query()
        try:
            client = self._get_client()
            df     = client.query_df(sql)
            logger.info("ClickHouseConnector: extracted %d rows", len(df))
            return df
        except Exception as exc:
            raise ConnectorError(
                f"ClickHouseConnector: extract failed — {exc}"
            ) from exc

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Stream ClickHouse result in LIMIT/OFFSET chunks."""
        size   = chunk_size or self._chunk_size
        base   = kwargs.get("query") or self._build_query()
        offset = 0
        client = self._get_client()
        while True:
            try:
                sql   = f"{base} LIMIT {size} OFFSET {offset}"
                chunk = client.query_df(sql)
                if chunk.empty:
                    break
                yield chunk
                offset += size
                if len(chunk) < size:
                    break
            except Exception as exc:
                raise ConnectorError(
                    f"ClickHouseConnector: stream failed at offset {offset} — {exc}"
                ) from exc

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_query(self) -> str:
        query = self.config.get("query")
        if query:
            return query
        table = self.config.get("table")
        if table:
            return f"SELECT * FROM {table}"
        raise ConnectorError(
            "ClickHouseConnector: 'table' or 'query' must be set in config"
        )
