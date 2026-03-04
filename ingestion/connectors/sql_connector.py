"""
ingestion/connectors/sql_connector.py
----------------------------------------
Production-grade SQL connector using SQLAlchemy.

Supports: PostgreSQL, MySQL, SQLite, SQL Server, Oracle
- Connection pooling (QueuePool)
- Credential injection from env vars only (never hardcoded)
- Incremental sync via configurable watermark column
- Schema extraction (column names + types + nullability)
- Chunked reading for large tables
- Retry with exponential backoff on transient failures
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.sql")

_RETRY_ATTEMPTS: int = 3
_RETRY_BASE_DELAY: float = 1.0  # seconds


def _get_env(key: str, fallback: Optional[str] = None) -> Optional[str]:
    """Fetch from env — credentials must never be in config literals."""
    return os.environ.get(key, fallback)


class SQLConnector(BaseConnector):
    """
    Universal SQL connector via SQLAlchemy.

    Config keys (all optional if DSN is set):
        dsn                : Full SQLAlchemy DSN (takes priority)
        dialect            : "postgresql" | "mysql" | "sqlite" | "mssql" | "oracle"
        host               : DB host (env: DB_HOST)
        port               : DB port (env: DB_PORT)
        database           : Database name (env: DB_NAME)
        username           : Username (env: DB_USER)
        password           : Password (env: DB_PASS)
        table              : Default table to query
        schema             : DB schema (default: None)
        watermark_col      : Column for incremental sync
        watermark_value    : Last synced value for incremental
        pool_size          : Connection pool size (default: 5)
        chunk_size         : Rows per chunk for stream() (default: 10000)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._engine = None
        self._pool_size: int = int(config.get("pool_size", 5))
        self._chunk_size: int = int(config.get("chunk_size", 10_000))

    def _get_dsn(self) -> str:
        """Build DSN from config + env vars. Credentials come from env."""
        if self.config.get("dsn"):
            return self.config["dsn"]

        dialect = self.config.get("dialect", "sqlite")
        host = _get_env("DB_HOST", self.config.get("host", "localhost"))
        port = _get_env("DB_PORT", str(self.config.get("port", "")))
        db = _get_env("DB_NAME", self.config.get("database", ":memory:"))
        user = _get_env("DB_USER", self.config.get("username", ""))
        pwd = _get_env("DB_PASS", self.config.get("password", ""))

        if dialect == "sqlite":
            return f"sqlite:///{db}"

        port_str = f":{port}" if port else ""
        auth = f"{user}:{pwd}@" if user else ""
        return f"{dialect}://{auth}{host}{port_str}/{db}"

    def _get_engine(self):
        """Lazy-create SQLAlchemy engine with connection pool."""
        if self._engine is not None:
            return self._engine
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool

            dsn = self._get_dsn()
            self._engine = create_engine(
                dsn,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=2,
                pool_timeout=30,
                pool_pre_ping=True,
            )
            return self._engine
        except ImportError as exc:
            raise ConnectorError("sqlalchemy is required: pip install sqlalchemy") from exc
        except Exception as exc:
            raise ConnectorError(f"SQLConnector: failed to create engine — {exc}") from exc

    def test_connection(self) -> bool:
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                conn.execute(self._text("SELECT 1"))
            logger.info("SQLConnector: connection test PASSED")
            return True
        except Exception as exc:
            logger.error("SQLConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        try:
            from sqlalchemy import inspect

            insp = inspect(self._get_engine())
            table = self.config.get("table")
            if not table:
                tables = insp.get_table_names(schema=self.config.get("schema"))
                return {"tables": tables, "description": "No table specified; listing all tables"}

            schema = self.config.get("schema")
            cols = insp.get_columns(table, schema=schema)
            col_meta = {
                c["name"]: {"type": str(c["type"]), "nullable": c.get("nullable", True)}
                for c in cols
            }
            # Approximate row count
            try:
                with self._get_engine().connect() as conn:
                    row = conn.execute(
                        self._text(f"SELECT COUNT(*) FROM {table}")
                    ).fetchone()
                    est_rows = row[0] if row else -1
            except Exception:
                est_rows = -1

            return {
                "table": table,
                "columns": list(col_meta.keys()),
                "dtypes": col_meta,
                "estimated_row_count": est_rows,
                "description": f"SQLAlchemy schema for {table}",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Extract data. Uses SQL query if provided, else reads the full
        configured table with optional incremental sync.
        """
        sql = query or self._build_query()
        engine = self._get_engine()

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                df = pd.read_sql(sql, engine)
                self._record_watermark(df)
                logger.info("SQLConnector: extracted %d rows", len(df))
                return df
            except Exception as exc:
                if attempt == _RETRY_ATTEMPTS:
                    raise ConnectorError(f"SQL extract failed after {attempt} attempts: {exc}") from exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("SQLConnector: attempt %d failed, retrying in %.1fs — %s", attempt, delay, exc)
                time.sleep(delay)
        return pd.DataFrame()

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Chunked streaming read for large tables."""
        size = chunk_size or self._chunk_size
        sql = kwargs.get("query") or self._build_query()
        engine = self._get_engine()
        try:
            for chunk in pd.read_sql(sql, engine, chunksize=size):
                yield chunk
        except Exception as exc:
            raise ConnectorError(f"SQL stream failed: {exc}") from exc

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def _build_query(self) -> str:
        """Build SELECT query with optional incremental watermark."""
        table = self.config.get("table")
        if not table:
            raise ConnectorError("SQLConnector: 'table' must be specified in config")

        wm_col = self.config.get("watermark_col")
        wm_val = self.config.get("watermark_value")
        base = f"SELECT * FROM {table}"
        if wm_col and wm_val is not None:
            base += f" WHERE {wm_col} > {repr(wm_val)}"
            base += f" ORDER BY {wm_col} ASC"
        return base

    def _record_watermark(self, df: pd.DataFrame) -> None:
        """Update watermark with max value from freshly extracted data."""
        wm_col = self.config.get("watermark_col")
        if wm_col and wm_col in df.columns and not df.empty:
            self.config["watermark_value"] = df[wm_col].max()

    @staticmethod
    def _text(sql: str):
        """Wrap raw SQL for SQLAlchemy 2.x compatibility."""
        try:
            from sqlalchemy import text
            return text(sql)
        except ImportError:
            return sql
