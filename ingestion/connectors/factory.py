"""
ingestion/connectors/factory.py
---------------------------------
ConnectorFactory — create any DIPEX connector by source_type.

Usage:
    connector = ConnectorFactory.create("sql", config)
    connector = ConnectorFactory.create("mongodb", config)
    connector = ConnectorFactory.create("kafka", config)
    connector = ConnectorFactory.create("api", config)
    connector = ConnectorFactory.create("redis", config)
    connector = ConnectorFactory.create("parquet", config)
    connector = ConnectorFactory.create("duckdb", config)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .base_connector import BaseConnector, ConnectorError

logger = logging.getLogger("dipex.connectors.factory")

_REGISTRY: Dict[str, str] = {
    # SQL / Relational
    "sql":           "ingestion.connectors.sql_connector.SQLConnector",
    "postgresql":    "ingestion.connectors.sql_connector.SQLConnector",
    "postgres":      "ingestion.connectors.sql_connector.SQLConnector",
    "mysql":         "ingestion.connectors.sql_connector.SQLConnector",
    "sqlite":        "ingestion.connectors.sql_connector.SQLConnector",
    "mssql":         "ingestion.connectors.sql_connector.SQLConnector",
    "oracle":        "ingestion.connectors.sql_connector.SQLConnector",
    # NoSQL / Document
    "mongodb":       "ingestion.connectors.mongo_connector.MongoConnector",
    "mongo":         "ingestion.connectors.mongo_connector.MongoConnector",
    # Key-Value
    "redis":         "ingestion.connectors.redis_connector.RedisConnector",
    "cache":         "ingestion.connectors.redis_connector.RedisConnector",
    # Streaming
    "kafka":         "ingestion.connectors.kafka_connector.KafkaConnector",
    # API
    "api":           "ingestion.connectors.api_connector.APIConnector",
    "rest":          "ingestion.connectors.api_connector.APIConnector",
    "graphql":       "ingestion.connectors.api_connector.APIConnector",
    # Analytical / Columnar
    "duckdb":        "ingestion.connectors.duckdb_connector.DuckDBConnector",
    "clickhouse":    "ingestion.connectors.clickhouse_connector.ClickHouseConnector",
    # Columnar file
    "parquet":       "ingestion.connectors.parquet_connector.ParquetConnector",
    "arrow":         "ingestion.connectors.parquet_connector.ParquetConnector",
    # Graph
    "neo4j":         "ingestion.connectors.neo4j_connector.Neo4jConnector",
    "graph":         "ingestion.connectors.neo4j_connector.Neo4jConnector",
    # Document / Search
    "elasticsearch": "ingestion.connectors.elasticsearch_connector.ElasticsearchConnector",
    "elastic":       "ingestion.connectors.elasticsearch_connector.ElasticsearchConnector",
    "opensearch":    "ingestion.connectors.elasticsearch_connector.ElasticsearchConnector",
}


class ConnectorFactory:
    """
    Factory for instantiating DIPEX data connectors.

    Connector config must include a 'source_type' key (used as lookup).
    You may also pass source_type explicitly to create().
    """

    @classmethod
    def create(cls, source_type: str, config: Dict[str, Any]) -> BaseConnector:
        """
        Instantiate and return a connector for the given source type.

        Args:
            source_type : One of: sql, postgresql, mysql, sqlite, mssql, oracle,
                          mongodb, mongo, kafka, api, rest, graphql
            config      : Source-specific config dict

        Returns:
            BaseConnector instance

        Raises:
            ConnectorError: if source_type is unknown or import fails
        """
        key = (source_type or "").lower().strip()
        class_path = _REGISTRY.get(key)
        if not class_path:
            raise ConnectorError(
                f"Unknown connector source_type: '{source_type}'. "
                f"Supported: {sorted(_REGISTRY.keys())}"
            )

        module_path, class_name = class_path.rsplit(".", 1)
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls_ = getattr(module, class_name)
            instance = cls_(config)
            logger.info("ConnectorFactory: created %s for source_type=%s", class_name, source_type)
            return instance
        except (ImportError, AttributeError) as exc:
            raise ConnectorError(f"Failed to import connector '{class_path}': {exc}") from exc

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseConnector:
        """
        Instantiate connector from a config dict with 'source_type' key.

        Args:
            config : Must contain 'source_type' key

        Returns:
            BaseConnector instance
        """
        source_type = config.get("source_type", config.get("type", ""))
        if not source_type:
            raise ConnectorError("ConnectorFactory.from_config: 'source_type' key required in config")
        return cls.create(source_type, config)

    @classmethod
    def supported_types(cls):
        """Return list of supported source type keys."""
        return sorted(_REGISTRY.keys())
