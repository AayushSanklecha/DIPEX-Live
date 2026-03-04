"""
ingestion/connectors/__init__.py
----------------------------------
Connector abstraction layer for DIPEX.

Every connector must implement the BaseConnector ABC.
Use ConnectorFactory.create(source_type, config) to instantiate.
"""

from .base_connector import BaseConnector
from .sql_connector import SQLConnector
from .mongo_connector import MongoConnector
from .kafka_connector import KafkaConnector
from .api_connector import APIConnector
from .duckdb_connector import DuckDBConnector
from .clickhouse_connector import ClickHouseConnector
from .neo4j_connector import Neo4jConnector
from .elasticsearch_connector import ElasticsearchConnector
from .factory import ConnectorFactory

__all__ = [
    "BaseConnector",
    "SQLConnector",
    "MongoConnector",
    "KafkaConnector",
    "APIConnector",
    "DuckDBConnector",
    "ClickHouseConnector",
    "Neo4jConnector",
    "ElasticsearchConnector",
    "ConnectorFactory",
]
