"""
api/routes/explorer.py
-----------------------
Raw data explorer endpoints — used by the Data Explorer frontend page.
Supports PostgreSQL and MongoDB preview with pagination.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/explorer", tags=["Data Explorer"])
logger = logging.getLogger("dipex.api.explorer")


class ConnectRequest(BaseModel):
    backend: str          # postgresql | mongodb
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    table: Optional[str] = None       # table / collection name
    limit: int = 200
    offset: int = 0
    search: Optional[str] = None      # simple text search across all columns


# ── helpers ───────────────────────────────────────────────────────────────────

def _pg_preview(req: ConnectRequest) -> Dict[str, Any]:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise HTTPException(503, "psycopg2 not installed. Run: pip install psycopg2-binary")

    dsn = f"host={req.host} port={req.port} dbname={req.database} user={req.username or ''} password={req.password or ''} connect_timeout=5"
    try:
        conn = psycopg2.connect(dsn)
    except Exception as exc:
        raise HTTPException(503, f"PostgreSQL connection failed: {exc}")

    cur = conn.cursor()

    # list tables if none selected
    if not req.table:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()
        return {"tables": tables, "rows": [], "columns": [], "total": 0}

    # count total rows
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{req.table}"')
        total = cur.fetchone()[0]
    except Exception:
        total = 0

    # fetch page with optional text search
    try:
        if req.search:
            cur.execute(f'SELECT * FROM "{req.table}" LIMIT 1')
            cols = [d[0] for d in cur.description]
            conditions = " OR ".join([f'"{c}"::text ILIKE %s' for c in cols])
            pattern = f"%{req.search}%"
            params = [pattern] * len(cols) + [req.limit, req.offset]
            cur.execute(f'SELECT * FROM "{req.table}" WHERE {conditions} LIMIT %s OFFSET %s', params)
        else:
            cur.execute(f'SELECT * FROM "{req.table}" LIMIT %s OFFSET %s', (req.limit, req.offset))

        cols = [d[0] for d in cur.description]
        rows = []
        for row in cur.fetchall():
            rows.append({cols[i]: (str(v) if v is not None else None) for i, v in enumerate(row)})
    except Exception as exc:
        conn.close()
        raise HTTPException(500, f"Query failed: {exc}")

    conn.close()
    return {"tables": [], "rows": rows, "columns": cols, "total": total}


def _mongo_preview(req: ConnectRequest) -> Dict[str, Any]:
    try:
        from pymongo import MongoClient
    except ImportError:
        raise HTTPException(503, "pymongo not installed. Run: pip install pymongo")

    uri = f"mongodb://{req.username}:{req.password}@{req.host}:{req.port}/" if req.username else f"mongodb://{req.host}:{req.port}/"
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        client.server_info()
    except Exception as exc:
        raise HTTPException(503, f"MongoDB connection failed: {exc}")

    db = client[req.database]

    if not req.table:
        tables = db.list_collection_names()
        client.close()
        return {"tables": tables, "rows": [], "columns": [], "total": 0}

    col = db[req.table]
    total = col.estimated_document_count()

    query = {}
    if req.search:
        query = {"$text": {"$search": req.search}}

    try:
        cursor = col.find(query, {"_id": 0}).skip(req.offset).limit(req.limit)
        docs = list(cursor)
    except Exception:
        # fallback without text index
        cursor = col.find({}, {"_id": 0}).skip(req.offset).limit(req.limit)
        docs = list(cursor)

    # flatten and stringify for JSON
    flat_rows = []
    all_cols: List[str] = []
    for doc in docs:
        row = {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
               for k, v in doc.items()}
        flat_rows.append(row)
        for k in row:
            if k not in all_cols:
                all_cols.append(k)

    client.close()
    return {"tables": [], "rows": flat_rows, "columns": all_cols, "total": total}


# ── routes ────────────────────────────────────────────────────────────────────

@router.post("/connect", summary="List tables/collections in a database")
async def explorer_connect(req: ConnectRequest) -> Dict[str, Any]:
    """Returns available tables for the given database connection."""
    req.table = None  # force listing mode
    if req.backend == "postgresql":
        return _pg_preview(req)
    elif req.backend == "mongodb":
        return _mongo_preview(req)
    raise HTTPException(400, f"Unsupported backend '{req.backend}'. Use: postgresql | mongodb")


@router.post("/preview", summary="Preview raw rows from a table/collection")
async def explorer_preview(req: ConnectRequest) -> Dict[str, Any]:
    """Returns paginated raw rows from the selected table/collection."""
    if not req.table:
        raise HTTPException(400, "Provide a table name in the request body")
    if req.backend == "postgresql":
        return _pg_preview(req)
    elif req.backend == "mongodb":
        return _mongo_preview(req)
    raise HTTPException(400, f"Unsupported backend '{req.backend}'. Use: postgresql | mongodb")
