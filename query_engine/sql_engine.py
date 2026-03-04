"""
query_engine/sql_engine.py
--------------------------
DuckDB-backed in-memory SQL engine for Pandas DataFrames.

Features:
  - Register any number of DataFrames as named DuckDB views
  - Execute parameterized SQL and return pd.DataFrame results
  - Query timing and row-count metadata
  - Connection pool singleton per engine instance
  - Safe query validation (read-only guard)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("dipex.query_engine.sql_engine")

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    sql: str
    rows: int
    columns: List[str]
    data: pd.DataFrame
    elapsed_ms: float
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "rows": self.rows,
            "columns": self.columns,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "error": self.error,
            "success": self.success,
        }

    def to_records(self) -> List[Dict[str, Any]]:
        """Return result as list-of-dicts (JSON-serializable)."""
        return self.data.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Mutation guard
# ─────────────────────────────────────────────────────────────────────────────

_MUTATION_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|COPY)\b",
    re.IGNORECASE,
)


def _is_readonly(sql: str) -> bool:
    return not _MUTATION_PATTERN.match(sql.strip())


# ─────────────────────────────────────────────────────────────────────────────
# SQLEngine
# ─────────────────────────────────────────────────────────────────────────────

class SQLEngine:
    """
    In-memory SQL engine backed by DuckDB.

    Usage::

        engine = SQLEngine()
        engine.register("sales", sales_df)
        engine.register("customers", customers_df)
        result = engine.execute("SELECT s.*, c.name FROM sales s JOIN customers c ON s.cid = c.id")
        print(result.data.head())
    """

    def __init__(self, read_only_guard: bool = True) -> None:
        try:
            import duckdb
            self._con = duckdb.connect(database=":memory:")
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for the SQL engine. "
                "Install it with: pip install duckdb"
            ) from exc

        self._registered: Dict[str, str] = {}   # name → fingerprint
        self._read_only_guard = read_only_guard
        logger.info("SQLEngine initialised (DuckDB in-memory).")

    def register(self, name: str, df: pd.DataFrame) -> None:
        """Register a DataFrame as a named DuckDB view."""
        import duckdb  # noqa: F401
        self._con.register(name, df)
        self._registered[name] = f"{name}({len(df)}r×{len(df.columns)}c)"
        logger.debug("Registered view '%s' (%d rows, %d cols).", name, len(df), len(df.columns))

    def unregister(self, name: str) -> None:
        """Remove a named view."""
        self._con.execute(f"DROP VIEW IF EXISTS {name}")
        self._registered.pop(name, None)

    def list_tables(self) -> List[str]:
        """List all registered view names."""
        return list(self._registered.keys())

    def execute(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        allow_mutations: bool = False,
    ) -> QueryResult:
        """
        Execute SQL and return a QueryResult.

        Parameters
        ----------
        sql           : SQL query string (DuckDB dialect, supports $1 $2 … or ? placeholders)
        params        : Optional parameter list for parameterized queries
        allow_mutations : If True, bypass the read-only guard (use with care)
        """
        if self._read_only_guard and not allow_mutations and not _is_readonly(sql):
            return QueryResult(
                sql=sql, rows=0, columns=[], data=pd.DataFrame(),
                elapsed_ms=0.0, error="Mutation queries are blocked (read-only guard).",
                success=False,
            )

        t0 = time.perf_counter()
        try:
            if params:
                result_rel = self._con.execute(sql, params)
            else:
                result_rel = self._con.execute(sql)
            result_df: pd.DataFrame = result_rel.df()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "SQL executed in %.1f ms → %d rows, %d cols.",
                elapsed, len(result_df), len(result_df.columns),
            )
            return QueryResult(
                sql=sql,
                rows=len(result_df),
                columns=list(result_df.columns),
                data=result_df,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error("SQL execution failed: %s", exc)
            return QueryResult(
                sql=sql, rows=0, columns=[], data=pd.DataFrame(),
                elapsed_ms=elapsed, error=str(exc), success=False,
            )

    def execute_named(
        self, query_registry: "QueryRegistry", name: str, params: Optional[List[Any]] = None
    ) -> QueryResult:
        """Execute a named query from a QueryRegistry."""
        sql = query_registry.get(name)
        if sql is None:
            return QueryResult(
                sql="", rows=0, columns=[], data=pd.DataFrame(),
                elapsed_ms=0.0, error=f"Named query '{name}' not found.", success=False,
            )
        return self.execute(sql, params=params)

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        self.close()

    # ── Convenience: analytics shortcuts ─────────────────────────────────────

    def profile_view(self, view_name: str) -> QueryResult:
        """Return DuckDB's built-in SUMMARIZE for a registered view."""
        return self.execute(f"SUMMARIZE {view_name}")

    def sample(self, view_name: str, n: int = 10) -> QueryResult:
        return self.execute(f"SELECT * FROM {view_name} LIMIT {n}")

    def count(self, view_name: str) -> int:
        res = self.execute(f"SELECT COUNT(*) AS n FROM {view_name}")
        if res.success and not res.data.empty:
            return int(res.data.iloc[0, 0])
        return -1
