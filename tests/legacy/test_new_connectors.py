"""
tests/test_new_connectors.py
------------------------------
Unit tests for the four new DIPEX database connectors:
  - DuckDBConnector    (analytical/columnar SQL)
  - ClickHouseConnector (columnar OLAP)
  - Neo4jConnector      (graph DB)
  - ElasticsearchConnector (document search)

All tests use mocks — no live database required.
Tests verify:
  - BaseConnector interface compliance (extract, test_connection, get_schema, stream)
  - Factory registration (ConnectorFactory.create(source_type, config))
  - Error handling on missing library / wrong config
  - Credential loading from environment variables
"""

from __future__ import annotations

import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ══════════════════════════════════════════════════════════════════════════════
# DuckDB Connector
# ══════════════════════════════════════════════════════════════════════════════

class TestDuckDBConnector:

    def test_importable(self):
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        assert DuckDBConnector is not None

    def test_factory_registration(self):
        from ingestion.connectors.factory import ConnectorFactory
        types = ConnectorFactory.supported_types()
        assert "duckdb" in types

    def test_creates_instance_from_factory(self):
        from ingestion.connectors.factory import ConnectorFactory
        conn = ConnectorFactory.create("duckdb", {"duckdb_path": ":memory:"})
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        assert isinstance(conn, DuckDBConnector)

    def test_extract_in_memory(self):
        """DuckDB in-memory: create table, insert, extract → DataFrame."""
        try:
            import duckdb  # skip if not installed
        except ImportError:
            pytest.skip("duckdb not installed")

        from ingestion.connectors.duckdb_connector import DuckDBConnector
        conn = DuckDBConnector({"duckdb_path": ":memory:"})
        # Bootstrap a table directly via internal connection
        dbc = conn._get_conn()
        dbc.execute("CREATE TABLE test (id INT, val FLOAT)")
        dbc.execute("INSERT INTO test VALUES (1, 1.5), (2, 2.5)")

        df = conn.extract("SELECT * FROM test")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "id" in df.columns
        conn.close()

    def test_stream_in_memory(self):
        """Stream in chunks smaller than table size."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        from ingestion.connectors.duckdb_connector import DuckDBConnector
        conn = DuckDBConnector({"duckdb_path": ":memory:", "chunk_size": 2})
        dbc = conn._get_conn()
        dbc.execute("CREATE TABLE big (x INT)")
        dbc.execute("INSERT INTO big VALUES (1),(2),(3),(4),(5)")

        chunks = list(conn.stream(query="SELECT * FROM big"))
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 5
        conn.close()

    def test_test_connection_passes(self):
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        conn = DuckDBConnector({"duckdb_path": ":memory:"})
        assert conn.test_connection() is True
        conn.close()

    def test_missing_library_raises_connector_error(self):
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        from ingestion.connectors.base_connector import ConnectorError
        with patch("builtins.__import__", side_effect=ImportError("no duckdb")):
            conn = DuckDBConnector({"duckdb_path": ":memory:"})
            conn._conn = None  # ensure no cached connection
            with pytest.raises(ConnectorError, match="duckdb is required"):
                conn._get_conn()

    def test_env_var_overrides_path(self, monkeypatch):
        monkeypatch.setenv("DUCKDB_PATH", ":memory:")
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        conn = DuckDBConnector({})
        assert conn._get_path() == ":memory:"

    def test_build_query_requires_table_or_query(self):
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        from ingestion.connectors.base_connector import ConnectorError
        conn = DuckDBConnector({})
        with pytest.raises(ConnectorError):
            conn._build_query()

    def test_schema_extraction(self):
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")
        from ingestion.connectors.duckdb_connector import DuckDBConnector
        conn = DuckDBConnector({"duckdb_path": ":memory:", "table": "info"})
        dbc = conn._get_conn()
        dbc.execute("CREATE TABLE info (a INT, b VARCHAR)")
        schema = conn.get_schema()
        assert "columns" in schema
        assert "a" in schema["columns"]
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# ClickHouse Connector
# ══════════════════════════════════════════════════════════════════════════════

class TestClickHouseConnector:

    def test_importable(self):
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        assert ClickHouseConnector is not None

    def test_factory_registration(self):
        from ingestion.connectors.factory import ConnectorFactory
        assert "clickhouse" in ConnectorFactory.supported_types()

    def test_creates_instance_from_factory(self):
        from ingestion.connectors.factory import ConnectorFactory
        conn = ConnectorFactory.create("clickhouse", {"host": "localhost", "table": "t"})
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        assert isinstance(conn, ClickHouseConnector)

    def test_missing_library_raises_connector_error(self):
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        from ingestion.connectors.base_connector import ConnectorError
        conn = ClickHouseConnector({"host": "localhost"})
        with patch.dict("sys.modules", {"clickhouse_connect": None}):
            conn._client = None
            with pytest.raises(ConnectorError, match="clickhouse-connect"):
                conn._get_client()

    def test_extract_with_mocked_client(self):
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        conn = ClickHouseConnector({"table": "events"})

        mock_client = MagicMock()
        expected_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_client.query_df.return_value = expected_df
        conn._client = mock_client

        df = conn.extract()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        mock_client.query_df.assert_called_once_with("SELECT * FROM events")

    def test_test_connection_with_mock(self):
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        conn = ClickHouseConnector({})
        mock_client = MagicMock()
        conn._client = mock_client
        assert conn.test_connection() is True
        mock_client.ping.assert_called_once()

    def test_env_var_credentials(self, monkeypatch):
        monkeypatch.setenv("CH_HOST", "ch-prod.internal")
        monkeypatch.setenv("CH_DB", "analytics")
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        conn = ClickHouseConnector({})
        assert conn._env("CH_HOST", "host") == "ch-prod.internal"

    def test_stream_paginates(self):
        from ingestion.connectors.clickhouse_connector import ClickHouseConnector
        conn = ClickHouseConnector({"table": "data", "chunk_size": 2})

        call_count = 0
        def fake_query_df(sql: str):
            nonlocal call_count
            call_count += 1
            if "OFFSET 0" in sql:
                return pd.DataFrame({"x": [1, 2]})
            if "OFFSET 2" in sql:
                return pd.DataFrame({"x": [3]})
            return pd.DataFrame()

        mock_client = MagicMock()
        mock_client.query_df.side_effect = fake_query_df
        conn._client = mock_client

        chunks = list(conn.stream(chunk_size=2))
        assert sum(len(c) for c in chunks) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j Connector
# ══════════════════════════════════════════════════════════════════════════════

class TestNeo4jConnector:

    def test_importable(self):
        from ingestion.connectors.neo4j_connector import Neo4jConnector
        assert Neo4jConnector is not None

    def test_factory_registration(self):
        from ingestion.connectors.factory import ConnectorFactory
        assert "neo4j" in ConnectorFactory.supported_types()

    def test_extract_with_mocked_driver(self):
        from ingestion.connectors.neo4j_connector import Neo4jConnector

        mock_record1 = {"name": "Alice", "age": 30}
        mock_record2 = {"name": "Bob",   "age": 25}

        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([mock_record1, mock_record2])
        )

        mock_session        = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__  = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        conn = Neo4jConnector({"query": "MATCH (n) RETURN n.name AS name, n.age AS age"})
        conn._driver = mock_driver

        df = conn.extract()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_requires_password(self):
        from ingestion.connectors.neo4j_connector import Neo4jConnector
        from ingestion.connectors.base_connector import ConnectorError

        conn = Neo4jConnector({"uri": "bolt://localhost", "username": "neo4j", "password": ""})
        with pytest.raises(ConnectorError, match="password required"):
            # Mock the neo4j import but allow ConnectorError to propagate
            with patch.dict("sys.modules", {"neo4j": MagicMock()}):
                conn._driver = None
                conn._get_driver()

    def test_test_connection_with_mock(self):
        from ingestion.connectors.neo4j_connector import Neo4jConnector
        conn = Neo4jConnector({})
        mock_driver = MagicMock()
        conn._driver = mock_driver
        assert conn.test_connection() is True
        mock_driver.verify_connectivity.assert_called_once()

    def test_stream_skips_and_limits(self):
        from ingestion.connectors.neo4j_connector import Neo4jConnector

        call_cypher = []

        def fake_run(cypher, **kwargs):
            call_cypher.append(cypher)
            if "SKIP 0" in cypher:
                results = [{"id": 1}, {"id": 2}]
            elif "SKIP 2" in cypher:
                results = [{"id": 3}]
            else:
                results = []
            mock_r = MagicMock()
            mock_r.__iter__ = MagicMock(return_value=iter(results))
            return mock_r

        mock_session        = MagicMock()
        mock_session.run    = fake_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__  = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        conn = Neo4jConnector({"query": "MATCH (n) RETURN n.id AS id", "chunk_size": 2})
        conn._driver = mock_driver

        chunks = list(conn.stream(chunk_size=2))
        assert len(chunks) >= 1
        assert sum(len(c) for c in chunks) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Elasticsearch Connector
# ══════════════════════════════════════════════════════════════════════════════

class TestElasticsearchConnector:

    def test_importable(self):
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector
        assert ElasticsearchConnector is not None

    def test_factory_registration(self):
        from ingestion.connectors.factory import ConnectorFactory
        types = ConnectorFactory.supported_types()
        assert "elasticsearch" in types
        assert "elastic" in types
        assert "opensearch" in types

    def test_extract_with_mocked_client(self):
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector

        mock_resp = {
            "hits": {
                "hits": [
                    {"_source": {"user": "alice", "score": 0.9}},
                    {"_source": {"user": "bob",   "score": 0.7}},
                ]
            }
        }

        mock_client = MagicMock()
        mock_client.search.return_value = mock_resp

        conn = ElasticsearchConnector({"index": "logs"})
        conn._client = mock_client

        df = conn.extract()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "user" in df.columns

    def test_extract_hits_from_dict_response(self):
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector
        resp = {
            "hits": {
                "hits": [
                    {"_source": {"a": 1}},
                    {"_source": {"a": 2}},
                ]
            }
        }
        hits = ElasticsearchConnector._extract_hits(resp)
        assert len(hits) == 2
        assert hits[0]["a"] == 1

    def test_scroll_id_extraction(self):
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector
        resp = {"_scroll_id": "test-scroll-abc123", "hits": {"hits": []}}
        sid = ElasticsearchConnector._get_scroll_id(resp)
        assert sid == "test-scroll-abc123"

    def test_missing_index_raises_error(self):
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector
        from ingestion.connectors.base_connector import ConnectorError
        conn = ElasticsearchConnector({})
        conn._client = MagicMock()
        with pytest.raises(ConnectorError, match="'index' must be specified"):
            conn.extract()

    def test_env_var_api_key(self, monkeypatch):
        monkeypatch.setenv("ES_API_KEY", "my-api-key-123")
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector
        conn = ElasticsearchConnector({"index": "logs"})
        assert conn._env("ES_API_KEY", "api_key") == "my-api-key-123"

    def test_stream_clears_scroll_on_completion(self):
        from ingestion.connectors.elasticsearch_connector import ElasticsearchConnector

        first_resp  = {
            "_scroll_id": "scroll-1",
            "hits": {"hits": [{"_source": {"x": 1}}, {"_source": {"x": 2}}]},
        }
        second_resp = {
            "_scroll_id": "scroll-2",
            "hits": {"hits": []},  # empty → stop
        }

        mock_client          = MagicMock()
        mock_client.search.return_value  = first_resp
        mock_client.scroll.return_value  = second_resp

        conn = ElasticsearchConnector({"index": "test-idx", "scroll_size": 2})
        conn._client = mock_client

        chunks = list(conn.stream())
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
        mock_client.clear_scroll.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Factory: all source types registered
# ══════════════════════════════════════════════════════════════════════════════

class TestFactoryRegistrations:

    def test_all_new_types_in_factory(self):
        from ingestion.connectors.factory import ConnectorFactory
        types = ConnectorFactory.supported_types()
        for expected in ("duckdb", "clickhouse", "neo4j", "elasticsearch",
                         "elastic", "opensearch", "graph"):
            assert expected in types, f"'{expected}' not in ConnectorFactory registry"

    def test_unknown_type_raises_connector_error(self):
        from ingestion.connectors.factory import ConnectorFactory
        from ingestion.connectors.base_connector import ConnectorError
        with pytest.raises(ConnectorError, match="Unknown connector"):
            ConnectorFactory.create("unknown_db_type_xyz", {})

    def test_existing_types_still_registered(self):
        from ingestion.connectors.factory import ConnectorFactory
        types = ConnectorFactory.supported_types()
        for expected in ("sql", "postgresql", "mongodb", "kafka", "api"):
            assert expected in types
