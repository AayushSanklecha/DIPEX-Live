"""
ingestion/connectors/redis_connector.py
-----------------------------------------
Production Redis key-value connector for DIPEX.

Redis is the industry-standard in-memory key-value store.
Used in DIPEX for:
  - Pipeline result caching (avoid re-running expensive steps)
  - Session store / rate-limit counters
  - Pub/Sub event bus between pipeline workers
  - Temporary storage of intermediate DataFrames as JSON/Arrow

Features:
  - Connection pooling via redis.ConnectionPool (thread-safe)
  - TTL (time-to-live) on all set() operations
  - get / set / delete / exists / TTL management
  - Hash operations (hget, hset, hgetall) for structured records
  - List operations (lpush, rpop, lrange) for queues
  - DataFrame ↔ JSON round-trip helpers
  - Pub/Sub channel support
  - Env-based config: REDIS_URL, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.redis")

_DEFAULT_TTL = 3600   # 1 hour default TTL for cached keys
_DEFAULT_DB  = 0


class RedisConnector(BaseConnector):
    """
    Redis key-value connector.

    Config keys:
        url          : Full Redis URL (env: REDIS_URL) — overrides host/port
        host         : Redis hostname (env: REDIS_HOST, default: localhost)
        port         : Redis port (env: REDIS_PORT, default: 6379)
        db           : Redis DB index (default: 0)
        password     : Redis password (env: REDIS_PASSWORD)
        default_ttl  : Default TTL in seconds for set() (default: 3600)
        max_connections: Connection pool size (default: 10)
        socket_timeout : Socket timeout in seconds (default: 5)
        decode_responses: Return str instead of bytes (default: True)

    Usage:
        conn = RedisConnector({"host": "localhost", "port": 6379})
        conn.set("mykey", "myvalue", ttl=300)
        val = conn.get("mykey")
        conn.set_df("df:run_001", df, ttl=1800)
        df2 = conn.get_df("df:run_001")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._pool = None
        self._client = None
        self._default_ttl: int = int(config.get("default_ttl", _DEFAULT_TTL))

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-create thread-safe Redis client with connection pool."""
        if self._client is not None:
            return self._client
        try:
            import redis  # type: ignore

            url      = os.environ.get("REDIS_URL", self.config.get("url"))
            host     = os.environ.get("REDIS_HOST", self.config.get("host", "localhost"))
            port     = int(os.environ.get("REDIS_PORT", self.config.get("port", 6379)))
            db       = int(self.config.get("db", _DEFAULT_DB))
            password = os.environ.get("REDIS_PASSWORD", self.config.get("password"))
            max_conn = int(self.config.get("max_connections", 10))
            timeout  = float(self.config.get("socket_timeout", 5.0))
            decode   = bool(self.config.get("decode_responses", True))

            if url:
                self._pool = redis.ConnectionPool.from_url(
                    url, max_connections=max_conn,
                    socket_timeout=timeout, decode_responses=decode,
                )
            else:
                self._pool = redis.ConnectionPool(
                    host=host, port=port, db=db, password=password,
                    max_connections=max_conn, socket_timeout=timeout,
                    decode_responses=decode,
                )

            self._client = redis.Redis(connection_pool=self._pool)
            logger.info(
                "RedisConnector: connected to %s",
                url or f"{host}:{port}/{db}",
            )
            return self._client

        except ImportError as exc:
            raise ConnectorError("redis is required: pip install redis") from exc
        except Exception as exc:
            raise ConnectorError(f"RedisConnector: connect failed — {exc}") from exc

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        try:
            self._get_client().ping()
            logger.info("RedisConnector: PING OK")
            return True
        except Exception as exc:
            logger.error("RedisConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        """Return Redis server info as schema."""
        try:
            info = self._get_client().info()
            return {
                "redis_version":    info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_days":   info.get("uptime_in_days"),
                "db_keys":          info.get(f"db{self.config.get('db', 0)}", {}).get("keys", 0),
                "description":      "Redis server info",
            }
        except Exception as exc:
            return {"error": str(exc)}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Scan keys matching a pattern and return as DataFrame.
        query = Redis key glob pattern (default: '*')
        """
        pattern = query or self.config.get("key_pattern", "*")
        try:
            client = self._get_client()
            keys   = list(client.scan_iter(pattern, count=500))
            if not keys:
                return pd.DataFrame(columns=["key", "value", "type", "ttl"])

            rows = []
            for k in keys:
                try:
                    ktype = client.type(k)
                    ttl   = client.ttl(k)
                    val   = None
                    if ktype == "string":
                        val = client.get(k)
                    elif ktype == "hash":
                        val = str(client.hgetall(k))
                    elif ktype == "list":
                        val = str(client.lrange(k, 0, 9))  # first 10
                    rows.append({"key": k, "value": val, "type": ktype, "ttl": ttl})
                except Exception:
                    pass

            return pd.DataFrame(rows)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector: extract failed — {exc}") from exc

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """Yield keys in scan chunks."""
        size    = chunk_size or int(self.config.get("chunk_size", 200))
        pattern = kwargs.get("query") or self.config.get("key_pattern", "*")
        client  = self._get_client()
        buf: List[Dict] = []
        for k in client.scan_iter(pattern):
            try:
                ttl = client.ttl(k)
                val = client.get(k)
                buf.append({"key": k, "value": val, "ttl": ttl})
                if len(buf) >= size:
                    yield pd.DataFrame(buf)
                    buf = []
            except Exception:
                pass
        if buf:
            yield pd.DataFrame(buf)

    def close(self) -> None:
        if self._pool:
            try:
                self._pool.disconnect()
            except Exception:
                pass
            self._pool = None
            self._client = None

    # ------------------------------------------------------------------
    # Key-value operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """Get a string value by key. Returns None if missing."""
        try:
            return self._get_client().get(key)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.get: {exc}") from exc

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key with optional TTL (seconds). Serialises non-str values to JSON."""
        if not isinstance(value, str):
            value = json.dumps(value)
        ex = ttl if ttl is not None else self._default_ttl
        try:
            self._get_client().set(key, value, ex=ex)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.set: {exc}") from exc

    def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns count deleted."""
        try:
            return self._get_client().delete(*keys)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.delete: {exc}") from exc

    def exists(self, key: str) -> bool:
        """Return True if key exists."""
        try:
            return bool(self._get_client().exists(key))
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.exists: {exc}") from exc

    def ttl(self, key: str) -> int:
        """Return TTL in seconds (-1 = no expire, -2 = does not exist)."""
        try:
            return self._get_client().ttl(key)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.ttl: {exc}") from exc

    # ------------------------------------------------------------------
    # Hash operations (structured records)
    # ------------------------------------------------------------------

    def hset(self, name: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple fields in a Redis hash. Optionally set TTL on the hash key."""
        try:
            client = self._get_client()
            client.hset(name, mapping={k: json.dumps(v) if not isinstance(v, str) else v
                                        for k, v in mapping.items()})
            if ttl is not None:
                client.expire(name, ttl)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.hset: {exc}") from exc

    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all fields from a Redis hash. Attempts JSON parse on each value."""
        try:
            raw = self._get_client().hgetall(name)
            result = {}
            for k, v in raw.items():
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            return result
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.hgetall: {exc}") from exc

    # ------------------------------------------------------------------
    # DataFrame helpers (cache DataFrames as JSON)
    # ------------------------------------------------------------------

    def set_df(self, key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> None:
        """Store a DataFrame as JSON string in Redis."""
        self.set(key, df.to_json(orient="split"), ttl=ttl)
        logger.debug("RedisConnector: cached DataFrame key=%s rows=%d", key, len(df))

    def get_df(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame previously stored with set_df(). Returns None if missing."""
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return pd.read_json(raw, orient="split")
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.get_df: JSON parse failed — {exc}") from exc

    # ------------------------------------------------------------------
    # Pub/Sub helpers
    # ------------------------------------------------------------------

    def publish(self, channel: str, message: Any) -> None:
        """Publish a message to a Redis pub/sub channel."""
        if not isinstance(message, str):
            message = json.dumps(message)
        try:
            self._get_client().publish(channel, message)
        except Exception as exc:
            raise ConnectorError(f"RedisConnector.publish: {exc}") from exc
