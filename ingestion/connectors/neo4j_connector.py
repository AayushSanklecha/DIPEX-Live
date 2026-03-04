"""
ingestion/connectors/neo4j_connector.py
-----------------------------------------
Production Neo4j graph database connector for DIPEX.

Uses the official neo4j Python driver (pip install neo4j).
Neo4j is the leading property graph database — nodes, relationships, and
properties modeled as a graph and queried via Cypher.

DIPEX integration:
- Cypher queries → flat DataFrame (via pd.DataFrame on record list)
- SKIP/LIMIT streaming for large result sets
- Schema introspection via `db.schema.nodeTypeProperties()`
- Env-var credential isolation (NEO4J_URI, NEO4J_USER, NEO4J_PASS)
- Supports Aura (Cloud), community, and enterprise Neo4j
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.neo4j")

_DEFAULT_URI  = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"
_CHUNK_SIZE   = 10_000


class Neo4jConnector(BaseConnector):
    """
    Neo4j graph database connector.

    Config keys:
        uri         : Bolt/Neo4j URI (env: NEO4J_URI, default: bolt://localhost:7687)
        username    : Username (env: NEO4J_USER, default: neo4j)
        password    : Password (env: NEO4J_PASS, required)
        database    : Neo4j database name (default: neo4j)
        query       : Cypher query for extract()
        params      : Dict of Cypher query parameters (optional)
        chunk_size  : Records per chunk for stream() (default: 10_000)
        max_conn_lifetime : Driver connection lifetime in seconds (default: 3600)
        connection_timeout : Connect timeout seconds (default: 30)
        encrypted   : Use TLS (default: False for bolt://, True for neo4j+s://)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._driver = None
        self._chunk_size: int = int(config.get("chunk_size", _CHUNK_SIZE))

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _env(self, env_key: str, cfg_key: str, default: str = "") -> str:
        return os.environ.get(env_key, self.config.get(cfg_key, default))

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase, basic_auth  # type: ignore

            uri      = self._env("NEO4J_URI",  "uri",      _DEFAULT_URI)
            user     = self._env("NEO4J_USER", "username", _DEFAULT_USER)
            password = self._env("NEO4J_PASS", "password", "")

            if not password:
                raise ConnectorError(
                    "Neo4jConnector: password required — set NEO4J_PASS env var "
                    "or config 'password'"
                )

            conn_timeout = int(self.config.get("connection_timeout", 30))
            max_lifetime = int(self.config.get("max_conn_lifetime", 3600))

            self._driver = GraphDatabase.driver(
                uri,
                auth=basic_auth(user, password),
                connection_timeout=conn_timeout,
                max_connection_lifetime=max_lifetime,
            )
            logger.info("Neo4jConnector: driver created for %s", uri)
            return self._driver
        except ImportError as exc:
            raise ConnectorError(
                "neo4j driver is required: pip install neo4j"
            ) from exc
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(
                f"Neo4jConnector: driver creation failed — {exc}"
            ) from exc

    def _get_database(self) -> Optional[str]:
        return self.config.get("database") or os.environ.get("NEO4J_DATABASE", "neo4j")

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            logger.info("Neo4jConnector: connection test PASSED")
            return True
        except Exception as exc:
            logger.error("Neo4jConnector: connection test FAILED — %s", exc)
            return False

    def get_schema(self) -> Dict[str, Any]:
        """
        Returns node labels, relationship types, and property schemas
        from Neo4j's built-in schema procedures.
        """
        try:
            driver = self._get_driver()
            db     = self._get_database()

            with driver.session(database=db) as session:
                # Node labels
                label_result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = [r["label"] for r in label_result]

                # Relationship types
                rel_result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN relationshipType"
                )
                rel_types = [r["relationshipType"] for r in rel_result]

                # Node property keys (best-effort schema)
                prop_result = session.run(
                    "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey LIMIT 200"
                )
                properties = [r["propertyKey"] for r in prop_result]

            return {
                "node_labels": labels,
                "relationship_types": rel_types,
                "property_keys": properties,
                "description": (
                    f"Neo4j graph schema: {len(labels)} labels, "
                    f"{len(rel_types)} relationship types"
                ),
            }
        except Exception as exc:
            return {"error": str(exc), "columns": [], "dtypes": {}, "estimated_row_count": -1}

    def extract(self, query: Optional[str] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Run a Cypher query and return results as a flat DataFrame.

        Args:
            query  : Cypher query string
            params : Dict of Cypher parameters (passed as **kwargs or config)
        """
        cypher = query or self.config.get("query")
        if not cypher:
            raise ConnectorError(
                "Neo4jConnector: 'query' (Cypher) must be provided in extract() or config"
            )
        params = kwargs.get("params", self.config.get("params", {}))

        try:
            driver = self._get_driver()
            db     = self._get_database()
            with driver.session(database=db) as session:
                result = session.run(cypher, parameters=params or {})
                records = [dict(r) for r in result]

            if not records:
                logger.info("Neo4jConnector: Cypher returned 0 records")
                return pd.DataFrame()

            df = pd.json_normalize(records)
            logger.info("Neo4jConnector: extracted %d records", len(df))
            return df
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(
                f"Neo4jConnector: extract failed — {exc}"
            ) from exc

    def stream(self, chunk_size: Optional[int] = None, **kwargs: Any) -> Iterator[pd.DataFrame]:
        """
        Stream large Cypher result sets via SKIP/LIMIT pagination.
        The query must NOT already contain SKIP/LIMIT.
        """
        size   = chunk_size or self._chunk_size
        base   = kwargs.get("query") or self.config.get("query")
        if not base:
            raise ConnectorError(
                "Neo4jConnector: 'query' must be set in config for stream()"
            )
        params = kwargs.get("params", self.config.get("params", {})) or {}
        skip   = 0
        driver = self._get_driver()
        db     = self._get_database()

        while True:
            paginated = f"{base} SKIP {skip} LIMIT {size}"
            try:
                with driver.session(database=db) as session:
                    result  = session.run(paginated, parameters=params)
                    records = [dict(r) for r in result]

                if not records:
                    break
                yield pd.json_normalize(records)
                skip += size
                if len(records) < size:
                    break
            except Exception as exc:
                raise ConnectorError(
                    f"Neo4jConnector: stream failed at skip={skip} — {exc}"
                ) from exc

    def close(self) -> None:
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
