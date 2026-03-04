"""
api/routes/query.py
---------------------
SQL query endpoint — execute DuckDB SQL against the latest approved dataset.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter(prefix="/query", tags=["SQL Query Engine"])
logger = logging.getLogger("dipex.api.query")


class QueryRequest(BaseModel):
    sql: str
    run_id: Optional[str] = None      # if provided, load that run's dataset
    params: Optional[List[Any]] = None


class SaveQueryRequest(BaseModel):
    name: str
    sql: str
    description: str = ""


@router.post("")
async def execute_query(req: QueryRequest):
    """Execute SQL against the most recent dataset."""
    from query_engine.sql_engine import SQLEngine
    from query_engine.lineage_tracker import LineageTracker

    df = _load_df(req.run_id)
    engine = SQLEngine()
    engine.register("df", df)

    result = engine.execute(req.sql, params=req.params)
    if not result.success:
        raise HTTPException(400, detail=result.error)

    # Record SQL lineage
    try:
        tracker = LineageTracker()
        tracker.record_sql(
            run_id=req.run_id or "adhoc",
            sql=req.sql,
            input_views=["df"],
            output_columns=result.columns,
        )
    except Exception:
        pass

    return {
        "sql": req.sql,
        "rows": result.rows,
        "columns": result.columns,
        "elapsed_ms": result.elapsed_ms,
        "data": result.to_records()[:500],  # Cap at 500 rows for API
    }


@router.post("/save")
async def save_named_query(req: SaveQueryRequest):
    """Save a named query to the registry."""
    from query_engine.query_registry import QueryRegistry
    registry = QueryRegistry()
    registry.save(req.name, req.sql, req.description)
    return {"status": "saved", "name": req.name}


@router.get("/named")
async def list_named_queries():
    """List all saved named queries."""
    from query_engine.query_registry import QueryRegistry
    registry = QueryRegistry()
    return {"queries": registry.list()}


def _load_df(run_id: Optional[str]) -> pd.DataFrame:
    """Load cleaned dataset if available, else sample."""
    if run_id:
        for suffix in ["_cleaned.csv", "_sample.csv"]:
            path = f"data/uploads/{run_id}{suffix}"
            if os.path.exists(path):
                return pd.read_csv(path)
    # Fallback: use most recent upload
    upload_dir = "data/uploads"
    if os.path.exists(upload_dir):
        files = sorted(
            [f for f in os.listdir(upload_dir) if f.endswith(".csv")],
            key=lambda f: os.path.getmtime(os.path.join(upload_dir, f)),
            reverse=True,
        )
        if files:
            return pd.read_csv(os.path.join(upload_dir, files[0]))
    raise HTTPException(404, detail="No dataset available. Run the pipeline first.")
