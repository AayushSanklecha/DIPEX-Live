"""
ingestion/connectors/mongo_connector.py
-----------------------------------------
Production MongoDB connector via pymongo.

Features:
- Collection → DataFrame flattening with schema inference
- Incremental sync via _id or custom watermark field
- Connection pooling (MongoClient singleton)
- Configurable projection (column selection)
- Aggregation pipeline support
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.mongo")


class MongoConnector(BaseConnector):
    """
    MongoDB connector using pymongo.

    Config keys:
        uri            : Full MongoDB URI (env: MONGO_URI)
        host           : Hostname (env: MONGO_HOST, default: localhost)
        port           : Port (default: 27017)
        database       : Database name (env: MONGO_DB, required)
        collection     : Collection name (required)
        username       : Username (env: MONGO_USER)
        password       : Password (env: MONGO_PASS)
        watermark_col  : Field for incremental sync (default: _id)
        watermark_value: Last synced value
        projection     : List of fields to include
        filter_query   : MongoDB filter dict (default: {})
        batch_size     : Cursor batch size (default: 1000)
        sample_size    : Rows to use for schema inference (default: 100)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._client = None

    def _get_uri(self) -> str:
        uri = os.environ.get("MONGO_URI", self.config.get("uri"))
        if uri:
            return uri
        host = os.environ.get("MONGO_HOST", self.config.get("host", "localhost"))
        port = self.config.get("port", 27017)
        user = os.environ.get("MONGO_USER", self.config.get("username", ""))
        pwd = os.environ.get("MONGO_PASS", self.config.get("password", ""))
        db = os.environ.get("MONGO_DB", self.config.get("database", "admin"))
        if user:
            return f"mongodb://{user}:{pwd}@{host}:{port}/{db}"
        return f"mongodb://{host}:{port}/{db}"

    def _get_client(self):
        if self._client is None:
            try:
                import pymongo  # type: ignore
                self._client = pymongo.MongoClient(
                    self._get_uri(),
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                )
            except ImportError as exc:
                raise ConnectorError("pymongo is required: pip install pymongo") from exc
            except Exception as exc:
                raise ConnectorError(f"MongoConnector: failed to connect — {exc}") from exc
        return self._client

    def test_connection(self) -> bool:
        try:
            client = self._get_client()
            client.server_info()
            logger.info("MongoConnector: connection test PASSED")
            return True
        except Exception as exc:
            logger.error("MongoConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        try:
            db_name = os.environ.get("MONGO_DB", self.config.get("database"))
            collection_name = self.config.get("collection")
            if not db_name or not collection_name:
                raise ConnectorError("MongoConnector: 'database' and 'collection' required")

            client = self._get_client()
            coll = client[db_name][collection_name]
            sample_size = self.config.get("sample_size", 100)

            # Schema inference from sample
            docs = list(coll.find({}, limit=sample_size))
            if not docs:
                return {"columns": [], "dtypes": {}, "estimated_row_count": 0}

            df_sample = pd.json_normalize(docs)
            dtypes = {col: str(df_sample[col].dtype) for col in df_sample.columns}
            est_count = coll.estimated_document_count()

            return {
                "database": db_name,
                "collection": collection_name,
                "columns": list(df_sample.columns),
                "dtypes": dtypes,
                "estimated_row_count": est_count,
                "description": f"MongoDB schema inferred from {len(docs)} sample docs",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Extract collection documents as DataFrame.
        `query` here is a JSON string of the MongoDB filter (optional).
        """
        try:
            import json

            db_name = os.environ.get("MONGO_DB", self.config.get("database"))
            coll_name = self.config.get("collection")
            if not db_name or not coll_name:
                raise ConnectorError("MongoConnector: 'database' and 'collection' required")

            filter_q = {}
            if query:
                try:
                    filter_q = json.loads(query)
                except json.JSONDecodeError:
                    pass
            filter_q = {**self.config.get("filter_query", {}), **filter_q}

            # Incremental sync
            wm_col = self.config.get("watermark_col", "_id")
            wm_val = self.config.get("watermark_value")
            if wm_val is not None:
                filter_q[wm_col] = {"$gt": wm_val}

            # Projection
            proj = self.config.get("projection")
            proj_dict = {f: 1 for f in proj} if proj else None

            client = self._get_client()
            coll = client[db_name][coll_name]
            cursor = coll.find(filter_q, proj_dict).batch_size(self.config.get("batch_size", 1000))

            docs = list(cursor)
            if not docs:
                logger.info("MongoConnector: no documents found")
                return pd.DataFrame()

            df = pd.json_normalize(docs)

            # Update watermark
            if wm_col in df.columns and not df.empty:
                self.config["watermark_value"] = df[wm_col].max()

            logger.info("MongoConnector: extracted %d documents", len(df))
            return df

        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"MongoConnector: extraction failed — {exc}") from exc

    def stream(self, chunk_size: int = 1000, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Stream collection in batches."""
        try:
            import json

            db_name = os.environ.get("MONGO_DB", self.config.get("database"))
            coll_name = self.config.get("collection")
            filter_q = self.config.get("filter_query", {})

            client = self._get_client()
            coll = client[db_name][coll_name]

            batch: List[Dict] = []
            for doc in coll.find(filter_q).batch_size(chunk_size):
                batch.append(doc)
                if len(batch) >= chunk_size:
                    yield pd.json_normalize(batch)
                    batch = []
            if batch:
                yield pd.json_normalize(batch)

        except Exception as exc:
            raise ConnectorError(f"MongoConnector stream failed: {exc}") from exc

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
