"""
ingestion/connectors/elasticsearch_connector.py
-------------------------------------------------
Production Elasticsearch / OpenSearch document store connector for DIPEX.

Uses the official elasticsearch-py client v8+ (pip install elasticsearch).
Also compatible with OpenSearch via the OpenSearch client (duck-typed API).

Features:
- Index search via ES Query DSL → DataFrame from _source fields
- Scroll API for large index scans (millions of documents)
- Index mapping introspection for schema extraction
- HTTPS + API key / basic auth support
- Env-var credential isolation
- Index aliases and multi-index patterns supported
- Auto-detects ES v7/v8 response format differences
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.elasticsearch")

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 9200
_SCROLL_TTL   = "2m"    # Keep scroll context alive for 2 minutes per page
_SCROLL_CHUNK = 1_000   # Documents per scroll page


class ElasticsearchConnector(BaseConnector):
    """
    Elasticsearch / OpenSearch document store connector.

    Config keys:
        host            : ES host (env: ES_HOST, default: localhost)
        port            : ES port (env: ES_PORT, default: 9200)
        index           : Index name or pattern (e.g. 'logs-*')
        query           : ES DSL query dict (default: match_all)
        username        : Username for basic auth (env: ES_USER)
        password        : Password for basic auth (env: ES_PASS)
        api_key         : API key string (env: ES_API_KEY) — preferred over user/pass
        use_ssl         : Enable HTTPS (default: False)
        verify_certs    : Verify TLS certificate (default: True)
        ca_certs        : Path to CA bundle for self-signed certs
        max_results     : Max documents to return in extract() (default: 10_000)
        scroll_size     : Documents per scroll page for stream() (default: 1_000)
        source_fields   : List of _source fields to include (default: all)
        scroll_ttl      : Scroll context TTL (default: '2m')
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._client = None
        self._max_results: int   = int(config.get("max_results", 10_000))
        self._scroll_size: int   = int(config.get("scroll_size", _SCROLL_CHUNK))
        self._scroll_ttl: str    = config.get("scroll_ttl", _SCROLL_TTL)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _env(self, env_key: str, cfg_key: str, default: str = "") -> str:
        return os.environ.get(env_key, self.config.get(cfg_key, default))

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from elasticsearch import Elasticsearch  # type: ignore

            host     = self._env("ES_HOST", "host",     _DEFAULT_HOST)
            port     = int(self._env("ES_PORT", "port", str(_DEFAULT_PORT)))
            use_ssl  = bool(self.config.get("use_ssl", False))
            verify   = bool(self.config.get("verify_certs", True))
            ca_certs = self.config.get("ca_certs")

            scheme   = "https" if use_ssl else "http"
            hosts    = [{"host": host, "port": port, "scheme": scheme}]

            kwargs: Dict[str, Any] = {
                "hosts":        hosts,
                "verify_certs": verify,
                "request_timeout": int(self.config.get("request_timeout", 60)),
            }
            if ca_certs:
                kwargs["ca_certs"] = ca_certs

            # Auth priority: API key > basic auth
            api_key  = self._env("ES_API_KEY", "api_key")
            username = self._env("ES_USER",    "username")
            password = self._env("ES_PASS",    "password")

            if api_key:
                # ES8 API key format: single base64-encoded "id:api_key"
                kwargs["api_key"] = api_key
            elif username:
                kwargs["basic_auth"] = (username, password)

            self._client = Elasticsearch(**kwargs)
            logger.info("ElasticsearchConnector: client created for %s:%d", host, port)
            return self._client

        except ImportError as exc:
            raise ConnectorError(
                "elasticsearch>=8.0.0 is required: pip install elasticsearch"
            ) from exc
        except Exception as exc:
            raise ConnectorError(
                f"ElasticsearchConnector: client creation failed — {exc}"
            ) from exc

    def _get_index(self) -> str:
        index = self.config.get("index", "")
        if not index:
            raise ConnectorError(
                "ElasticsearchConnector: 'index' must be specified in config"
            )
        return index

    def _get_query(self) -> Dict[str, Any]:
        return self.config.get("query") or {"query": {"match_all": {}}}

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        try:
            client = self._get_client()
            info   = client.info()
            version = (
                info.get("version", {}).get("number", "unknown")
                if isinstance(info, dict)
                else "unknown"
            )
            logger.info(
                "ElasticsearchConnector: connection PASSED (ES version %s)", version
            )
            return True
        except Exception as exc:
            logger.error("ElasticsearchConnector: connection FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        """
        Fetch index mapping and convert into a flat column/type schema.
        Handles nested objects by flattening with dot notation.
        """
        try:
            client = self._get_client()
            index  = self._get_index()
            mapping_resp = client.indices.get_mapping(index=index)

            # Flatten the mapping tree
            columns: List[str] = []
            dtypes:  Dict[str, str] = {}

            def _flatten(props: Dict, prefix: str = "") -> None:
                for field, meta in props.items():
                    full = f"{prefix}{field}" if not prefix else f"{prefix}.{field}"
                    if "properties" in meta:
                        _flatten(meta["properties"], prefix=full)
                    else:
                        columns.append(full)
                        dtypes[full] = meta.get("type", "object")

            for idx_name, idx_body in mapping_resp.items():
                props = (
                    idx_body.get("mappings", {}).get("properties", {})
                )
                _flatten(props)
                break  # Only inspect first resolved index

            # Estimated doc count
            est_count = -1
            try:
                count_resp = client.count(index=index)
                est_count = (
                    count_resp["count"]
                    if isinstance(count_resp, dict)
                    else count_resp.body.get("count", -1)
                )
            except Exception:
                pass

            return {
                "index": index,
                "columns": columns,
                "dtypes": dtypes,
                "estimated_row_count": est_count,
                "description": f"Elasticsearch index mapping for '{index}'",
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Search the index and return results as a DataFrame.

        For large result sets, use stream() which uses the Scroll API.
        extract() is capped at self._max_results documents.

        Args:
            query : JSON string OR dict of ES DSL query (optional; uses config default)
        """
        import json as _json

        index  = self._get_index()
        dsl    = self._get_query()

        # Allow passing query as JSON string or dict override
        if query:
            try:
                dsl = _json.loads(query) if isinstance(query, str) else query
            except _json.JSONDecodeError:
                logger.warning("ElasticsearchConnector: invalid query JSON, using config default")

        source_fields = self.config.get("source_fields")
        try:
            client = self._get_client()
            resp   = client.search(
                index=index,
                body=dsl,
                size=min(self._max_results, 10_000),
                source=source_fields or True,
            )
            hits   = self._extract_hits(resp)
            if not hits:
                return pd.DataFrame()
            df = pd.json_normalize(hits)
            logger.info("ElasticsearchConnector: extracted %d documents", len(df))
            return df
        except Exception as exc:
            raise ConnectorError(
                f"ElasticsearchConnector: extract failed — {exc}"
            ) from exc

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """
        Stream all matching documents using the ES Scroll API.
        Suitable for large index scans (millions of documents).
        Automatically clears the scroll context on completion.
        """
        size      = chunk_size or self._scroll_size
        index     = self._get_index()
        dsl       = self._get_query()
        if "query" in kwargs:
            dsl = kwargs["query"]

        source_fields = self.config.get("source_fields")
        client  = self._get_client()
        scroll_id = None

        try:
            # Initialize scroll
            resp = client.search(
                index=index,
                body=dsl,
                size=size,
                scroll=self._scroll_ttl,
                source=source_fields or True,
            )
            scroll_id = self._get_scroll_id(resp)
            hits      = self._extract_hits(resp)

            while hits:
                yield pd.json_normalize(hits)
                # Fetch next page
                resp      = client.scroll(scroll_id=scroll_id, scroll=self._scroll_ttl)
                scroll_id = self._get_scroll_id(resp)
                hits      = self._extract_hits(resp)

        except Exception as exc:
            raise ConnectorError(
                f"ElasticsearchConnector: scroll stream failed — {exc}"
            ) from exc
        finally:
            if scroll_id:
                try:
                    client.clear_scroll(scroll_id=scroll_id)
                    logger.debug("ElasticsearchConnector: scroll context cleared")
                except Exception:
                    pass

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hits(resp: Any) -> List[Dict[str, Any]]:
        """Extract _source dicts from an ES response (v7 and v8 compatible)."""
        try:
            if isinstance(resp, dict):
                raw_hits = resp.get("hits", {}).get("hits", [])
            else:
                raw_hits = resp.body.get("hits", {}).get("hits", [])
            return [h.get("_source", {}) for h in raw_hits]
        except Exception:
            return []

    @staticmethod
    def _get_scroll_id(resp: Any) -> Optional[str]:
        """Extract _scroll_id from response (v7/v8 compatible)."""
        try:
            if isinstance(resp, dict):
                return resp.get("_scroll_id")
            return resp.body.get("_scroll_id")
        except Exception:
            return None
